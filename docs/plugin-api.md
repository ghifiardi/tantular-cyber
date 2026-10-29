# Scanner plugin API (contract only)

Plugin API **1.0** · report schema **1.3.0** · code: `src/tantular_cyber/plugins.py`

This document defines how a scanner can plug into Tantular Cyber. In POC-006 it is a
**contract and a validator, not an execution engine**: the two built-in scanners are the only
code that runs, and third-party plugins are **never executed** until an enforceable sandbox
exists (see [Execution policy](#execution-policy)).

```bash
tantular-cyber plugins list                    # API version, report schema, built-ins, policy
tantular-cyber plugins validate plugin.json    # validate + negotiate; never runs anything
```

## Design rules

1. **Scanner evidence and model output are separate.** A plugin returns findings only. It
   cannot create, read or modify `analysis` (advisory LLM output), report `status`,
   `status_reasons` or other scanners' findings. Findings are immutable once normalized.
2. **Plugins are untrusted.** They are described declaratively; nothing is imported from
   them, no entry points are discovered, and no plugin code runs in-process.
3. **Fail closed.** Any invalid manifest or output field rejects the whole plugin result.
4. **Report compatibility.** Normalized plugin findings become ordinary `Finding` objects
   in report schema 1.3.0 with `tool = "plugin:<id>"`; no schema change is needed.

## Manifest (`manifest_version: 1`)

```json
{
  "manifest_version": 1,
  "plugin_api": "1.0",
  "id": "example-scanner",
  "name": "Example scanner",
  "version": "0.1.0",
  "publisher": "example",
  "license": "Apache-2.0",
  "languages": ["python"],
  "cwe": ["CWE-89"],
  "rule_id_prefix": "plugin.example-scanner.",
  "permissions": {
    "read_roots": ["target_repo"],
    "network": false,
    "write": "none",
    "spawn_processes": false,
    "environment": [],
    "limits": {"max_wall_s": 300, "max_cpu_s": 300, "max_memory_mb": 1024,
               "max_output_bytes": 16000000}
  },
  "entrypoint": {"kind": "subprocess-json", "command": ["example-scanner", "--json"],
                 "protocol": "tantular-plugin-output/1"},
  "trust": "third_party"
}
```

* Unknown fields are rejected. `id` is lowercase (`[a-z][a-z0-9-]{2,47}`); rule ids must use
  the namespace `plugin.<id>.`, so a plugin can never impersonate built-in rules.
* Permissions can only express the safe values: read access to the scanned repository only,
  no network, no writes, no child processes, bounded wall/CPU time, memory and output, and
  an environment allowlist that rejects secret-like names (`*TOKEN*`, `*KEY*`, ...).
* `trust` is always `third_party`; built-ins are not loaded from manifests.

## Compatibility negotiation

`plugin_api` is `MAJOR.MINOR`. A plugin is compatible when its major equals the host's and
its minor is not newer than the host's. Host 1.0 accepts `1.0`, rejects `1.1` ("plugin needs
API 1.1; host provides 1.0") and `2.0`. Minor versions only add optional fields; a major
version change may remove or rename fields.

## Output protocol (`tantular-plugin-output/1`)

```json
{
  "protocol": "tantular-plugin-output/1",
  "plugin_id": "example-scanner",
  "plugin_version": "0.1.0",
  "findings": [
    {"rule_id": "plugin.example-scanner.sqli", "path": "app.py", "start_line": 14,
     "end_line": 14, "message": "query built from input", "severity": "high",
     "cwe": ["CWE-89"], "confidence": "medium", "uncertainty": []}
  ]
}
```

`normalize_plugin_output` rejects the whole output when the identity does not match the
manifest, a rule id is outside the namespace, a CWE was not declared, a path is absolute,
contains `..` or `\`, or was not in the scanned inventory, a line range is inverted, a
severity is unknown, an extra field (for example `analysis`) appears, or there are more than
10,000 findings. Accepted findings then go through the same redaction, evidence-context and
de-duplication pipeline as built-in findings (Semgrep keeps precedence on duplicates).

## Built-in scanners

| id | Trust | Execution | Rule prefix | CWEs |
|---|---|---|---|---|
| `semgrep` | builtin | external Semgrep CLI (user-installed), bundled local rules, `--metrics=off` | `tantular.` | 89, 78, 22, 79, 798 |
| `tantular-callgraph` | builtin | in-process AST analysis; parses files, never imports or runs them | `tantular.callgraph.` | 89, 78, 22 |

## Execution policy

`execute_plugin()` always raises `PluginExecutionDisabled`. Running a third-party plugin in a
plain subprocess with a timeout is **not** a sandbox: it can still open sockets, read your
home directory, write to the repository, spawn processes and read secrets from the
environment. Execution will only be enabled when an isolation boundary can **enforce**, on
each supported OS:

* no network (network namespace / sandbox profile denying sockets);
* read-only access limited to the scanned repository, no other filesystem access, no writes
  except a host-provided output pipe;
* no child processes, bounded CPU time, memory and wall time, bounded output size;
* a scrubbed environment (allowlisted variables only) and no inherited credentials;
* deterministic audit logs of what ran, with which manifest hash, limits and exit status.

Candidates to evaluate in a later milestone: Linux namespaces + seccomp (e.g. bubblewrap),
macOS sandbox profiles, WebAssembly (WASI) runtimes, or containers with networking disabled.
Until one is implemented and tested on every supported platform, plugin execution stays
disabled and the CLI has no command to run a plugin.
