# Panduan cepat

[English](quickstart.md) · Panduan instalasi lengkap (bahasa Inggris): [install.md](install.md)

Tantular Cyber menemukan masalah keamanan (SQL injection CWE-89, command injection CWE-78,
path traversal CWE-22, XSS CWE-79, kredensial tertanam CWE-798) di repositori **Python**
langsung di komputer Anda. Alat ini hanya membaca berkas: kode yang dipindai tidak pernah
dijalankan, di-import, diubah, maupun diunggah.

## 1. Instalasi (sekitar 5 menit, tanpa model AI)

```bash
brew install semgrep          # Linux/WSL2: pipx install semgrep
pipx install ./tantular_cyber-0.3.0-py3-none-any.whl   # wheel rilis yang sudah diverifikasi, lihat install.md
tantular-cyber doctor
```

Pastikan muncul `scan capability: FULL`. Jika tertulis `PARTIAL`, Semgrep belum terpasang;
pemindaian tetap berjalan tetapi ditandai *degraded* (cakupan sebagian).

Semgrep adalah dependensi eksternal berlisensi LGPL-2.1 yang Anda pasang sendiri; Tantular
Cyber tidak menyertakannya. Windows: gunakan WSL2 lalu ikuti langkah Linux.

## 2. Coba pada contoh bawaan

```bash
tantular-cyber sample --dest ./tc-demo/sample-app
tantular-cyber scan --repo ./tc-demo/sample-app --output ./tc-demo/reports
```

```
status: complete
coverage: full (scanners succeeded: semgrep, tantular-callgraph)
ai explanations: not requested
findings: 2 {'CWE-89': 2}
```

Contoh ini sintetis dan sengaja dibuat rentan; tidak pernah dijalankan. Buka
`./tc-demo/reports/report.md` (laporan yang mudah dibaca) atau `report.json` (untuk mesin,
skema 1.3.0).

## 3. Pindai repositori Anda

```bash
tantular-cyber scan --repo ~/code/proyek-saya --output ~/laporan-tantular/proyek-saya
```

* Pindai hanya repositori yang Anda berwenang untuk menilai.
* Laporan harus ditulis **di luar** repositori yang dipindai.
* Kode keluar: `0` lengkap, `4` *degraded* (cakupan sebagian atau tahap AI tidak tersedia),
  `3` gagal, `2` masukan tidak valid. Alasan *degraded* selalu tercantum di
  `status_reasons`.
* Temuan adalah bukti pemindai (`status: unverified`); tinjau sebelum bertindak. Tantular
  Cyber tidak pernah mengubah kode Anda.

## 4. Opsional: penjelasan AI

Jika [Ollama](https://ollama.com) berjalan dan model lokal sudah terpasang (lihat
[install.md §4](install.md#4-optional-local-ai-explanations-ollama)):

```bash
tantular-cyber scan --repo ~/code/proyek-saya --output ~/laporan-tantular/proyek-saya --ai \
    --model ghifidanukusumo/tantular:latest
```

Penjelasan berbahasa Indonesia dan bersifat **saran**: tidak pernah mengubah temuan, statusnya,
maupun status laporan. Cuplikan kode disamarkan (redaksi) dan hanya dikirim ke server Ollama
lokal. Tantular Cyber tidak pernah mengunduh model; `ollama pull` adalah keputusan Anda
(unduhan sekitar 10 GB untuk bobot Tantular, lisensi mengikuti halaman model). Model *cloud*
(`*:cloud`) berjalan di server jarak jauh dan ditolak kecuali Anda memberi
`--allow-remote-ollama`.

## Keterbatasan

* Hanya Python; lima kelompok CWE. Presisi 0,84 / recall 0,91 pada benchmark sintetis adalah
  batas regresi, **bukan** akurasi dunia nyata.
* Pemindai call-graph mengikuti panggilan hingga 3 tingkat di dalam repositori dan tidak
  mengikuti *dynamic dispatch*; sebagian alur akan terlewat.
* Kualitas penjelasan AI baru diukur dengan pemeriksaan proksi pada korpus sintetis kecil.

## Privasi

Tanpa telemetri, tanpa login, tanpa cloud. Detail: [privacy.md](privacy.md).

## Menghapus

```bash
pipx uninstall tantular-cyber
rm -rf ~/.cache/tantular-cyber
```

Laporan hanya ada di folder `--output` yang Anda pilih.
