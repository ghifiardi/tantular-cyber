# Provenance and licensing audit

Generated from the manifests (POC-005). Application code: Apache-2.0. Model weights: their own licenses.

## Synthetic corpus (`benchmarks/corpus`)

- 33 cases: {'synthetic': 29, 'cve_informed': 4}; third-party code copied: none.
- All cases are independently written; credentials are synthetic (`tc-synthetic-…`) or the AWS documentation example key.

| Case | Kind | License | References |
|---|---|---|---|
| accounts-lookup | synthetic | Apache-2.0 | - |
| orders-reports | synthetic | Apache-2.0 | - |
| admin-maintenance | synthetic | Apache-2.0 | - |
| auth-login | synthetic | Apache-2.0 | - |
| near-miss-logging | synthetic | Apache-2.0 | - |
| known-fp-constants | synthetic | Apache-2.0 | - |
| known-fp-task-runner | synthetic | Apache-2.0 | - |
| known-fn-helpers | synthetic | Apache-2.0 | - |
| injection-en | synthetic | Apache-2.0 | - |
| injection-id | synthetic | Apache-2.0 | - |
| unicode-bidi | synthetic | Apache-2.0 | - |
| secrets-nearby | synthetic | Apache-2.0 | - |
| cmdi-tools | synthetic | Apache-2.0 | - |
| cmdi-launcher | cve_informed | Apache-2.0 | [CVE-2024-29189](https://nvd.nist.gov/vuln/detail/CVE-2024-29189) |
| cmdi-limits | synthetic | Apache-2.0 | - |
| path-files | synthetic | Apache-2.0 | - |
| path-archives | cve_informed | Apache-2.0 | [CVE-2007-4559](https://nvd.nist.gov/vuln/detail/CVE-2007-4559) |
| path-limits | synthetic | Apache-2.0 | - |
| xss-flask | synthetic | Apache-2.0 | - |
| xss-jinja | synthetic | Apache-2.0 | - |
| xss-django | synthetic | Apache-2.0 | - |
| xss-template-debug | cve_informed | Apache-2.0 | [CVE-2017-12794](https://nvd.nist.gov/vuln/detail/CVE-2017-12794) |
| xss-template-article | synthetic | Apache-2.0 | - |
| creds-flask | cve_informed | Apache-2.0 | [CVE-2022-25510](https://nvd.nist.gov/vuln/detail/CVE-2022-25510) |
| creds-clients | synthetic | Apache-2.0 | - |
| context-report-builder | synthetic | Apache-2.0 | - |
| context-deploy | synthetic | Apache-2.0 | - |
| interproc-service | synthetic | Apache-2.0 | - |
| interproc-queries | synthetic | Apache-2.0 | - |
| interproc-package | synthetic | Apache-2.0 | - |
| interproc-repository | synthetic | Apache-2.0 | - |
| interproc-deep-chain | synthetic | Apache-2.0 | - |
| chains-schema-browser | synthetic | Apache-2.0 | - |

## Real-world pointers (`benchmarks/realworld`) — pointer-only, no upstream code stored

| Pointer | CWE | Repository @ commit | Upstream license | Advisory | Verification | Held out |
|---|---|---|---|---|---|---|
| rw-wsgidav-mysql-select | CWE-89 | mar10/wsgidav @ `67abdbd7d622` | MIT | [CVE-2026-55509](https://github.com/advisories/GHSA-p6gw-4frg-j7jw) | upstream_label | no |
| rw-wsgidav-mysql-describe | CWE-89 | mar10/wsgidav @ `67abdbd7d622` | MIT | [CVE-2026-55509](https://github.com/advisories/GHSA-p6gw-4frg-j7jw) | upstream_label | yes |
| rw-scitokens-keycache-delete | CWE-89 | scitokens/scitokens @ `66b075c35da1` | Apache-2.0 | [CVE-2026-32714](https://github.com/advisories/GHSA-rh5m-2482-966c) | upstream_label | yes |
| rw-mem0-databricks-delete | CWE-89 | mem0ai/mem0 @ `3a9fcdbec25f` | Apache-2.0 | [CVE-2026-115324](https://github.com/advisories/GHSA-7587-9684-52r6) | upstream_label | yes |
| rw-djust-fixture-path | CWE-22 | djust-org/djust @ `ce5b5151eb10` | MIT | [CVE-2026-113731](https://github.com/advisories/GHSA-r57g-mc79-8hj2) | upstream_label | yes |
| rw-setuptools-namespace-dir | CWE-22 | pypa/setuptools @ `8419f006d4e3` | MIT | [GHSA-xmj3-9gf7-h52w](https://github.com/advisories/GHSA-xmj3-9gf7-h52w) | upstream_label | yes |
| rw-glances-action-command | CWE-78 | nicolargo/glances @ `65b27142f523` | NOASSERTION | [CVE-2026-68519](https://github.com/advisories/GHSA-59fj-m2j6-hcxh) | upstream_label | yes |

License values come from the GitHub repository API (`license.spdx_id`, checked 2026-10-29); `NOASSERTION` means GitHub could not classify the license — such sources stay pointer-only. Fetched files live only in the local cache and are never redistributed.
