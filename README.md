<div align="center">

# Edge Vision Counter

Sistem penghitung orang multi-kamera berbasis edge untuk pemantauan real-time melalui kamera IP atau RTSP.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-111827?style=flat-square)](https://www.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Supervision](https://img.shields.io/badge/Supervision-0.19%2B-00B3FF?style=flat-square)](https://supervision.roboflow.com/)
[![SQLite](https://img.shields.io/badge/SQLite-Built%20in-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.6%2B-0071C5?style=flat-square&logo=intel&logoColor=white)](https://docs.openvino.ai/)

</div>

## Ringkasan

Repository ini berisi aplikasi counting orang yang menjalankan pipeline vision, penyimpanan state, dan dashboard web dalam satu proses utama. Alurnya dimulai dari `main.py`, lalu menginisialisasi `StateManager`, `VisionModel`, dan `CameraManager`, kemudian menyalakan server Flask pada port 5000.

Fokus utamanya adalah stabil pada kamera Wi-Fi/RTSP yang tidak selalu konsisten. Implementasi di repo ini memakai buffer capture satu frame, normalisasi frame dengan letterbox, deteksi orang dengan YOLOv8n, tracking dengan ByteTrack dari `supervision`, lalu pencatatan event IN/OUT ke SQLite.

## Fitur Utama

- Multi-kamera dengan konfigurasi per kamera: `orientation`, `line_ratio`, `in_direction`, dan `frame_skip`.
- Pipeline inference yang bisa berjalan dengan backend `auto`, `cuda`, `openvino`, atau `cpu`.
- Normalisasi frame berbasis letterbox agar rasio gambar tidak rusak sebelum tracking dan inference.
- State counting yang dipersistenkan ke SQLite dan snapshot state JSON.
- Dashboard web real-time dengan SSE, grafik tren, ringkasan per kamera, dan panel kesehatan sistem.
- Halaman history untuk analitik event dan ekspor CSV.
- Endpoint admin untuk kalibrasi manual dan reset sesi dengan password.
- mDNS opsional agar dashboard bisa diakses via hostname lokal jika jaringan mendukung.

## Arsitektur Singkat

1. `main.py` mengatur logging, memuat state, memulai model, lalu menambah semua kamera yang aktif dari `config.py`.
2. `CameraManager` membaca stream kamera, menjalankan tracking, dan mengirim event crossing ke `StateManager`.
3. `StateManager` menyimpan total masuk/keluar dan menulis event ke SQLite.
4. `web/app.py` menyediakan dashboard, history page, SSE stream, JSON API, ekspor CSV, dan health check.

## Prasyarat

- Python 3.9 atau lebih baru.
- Kamera IP, RTSP, atau stream HTTP yang bisa diakses dari mesin ini.
- Ruang disk untuk database SQLite dan log aplikasi.
- Jika ingin memakai OpenVINO atau CUDA, pastikan environment dan driver sesuai dengan perangkat.

## Instalasi

### 1. Buat dan aktifkan virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Command Prompt:

```bat
venv\Scripts\activate.bat
```

Linux atau macOS:

```bash
source venv/bin/activate
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

File `requirements.txt` sudah mengarah ke indeks wheel PyTorch yang sesuai untuk build di repo ini.

## Konfigurasi Kamera

Semua kamera aktif diatur di `config.py` pada variabel `CAMERAS`.

Contoh struktur konfigurasi:

```python
CAMERAS = [
    {
        "id": "Pintu_Utama",
        "url": "rtsp://user:password@192.168.1.10:554/stream2",
        "orientation": "vertical",
        "line_ratio": 0.5,
        "in_direction": "left_to_right",
        "frame_skip": 3,
        "enabled": True,
    }
]
```

Arti field utama:

- `id`: nama unik kamera yang muncul di dashboard dan log.
- `url`: alamat stream `rtsp://` atau `http://`.
- `orientation`: `vertical` atau `horizontal`.
- `line_ratio`: posisi garis crossing dari 0.0 sampai 1.0.
- `in_direction`: `left_to_right`, `right_to_left`, `top_to_bottom`, atau `bottom_to_top`.
- `frame_skip`: proses inference setiap N frame.
- `enabled`: aktif atau tidak.

## Backend Inference dan Model

Repo ini memakai model `yolov8n.pt` untuk deteksi, dengan opsi OpenVINO yang sudah disediakan di folder `yolov8n_openvino_model/`.

Perilaku backend inference:

- `auto`: mencoba CUDA, lalu OpenVINO, lalu CPU.
- `cuda`: memaksa GPU NVIDIA.
- `openvino`: memaksa OpenVINO.
- `cpu`: memaksa CPU.

Default konfigurasi berada di `config.py` dan bisa diubah tanpa menyentuh kode utama. Jika Anda ingin menjalankan OpenVINO, pastikan file IR yang dirujuk oleh `OPENVINO_XML_PATH` tersedia.

## Menjalankan Aplikasi

```bash
python main.py
```

Setelah aplikasi hidup, buka:

- `http://localhost:5000`
- atau `http://counter-3wifi.local:5000` jika mDNS aktif dan hostname tidak diubah

## Endpoint yang Tersedia

### Halaman Web

- `GET /` - Dashboard utama.
- `GET /history` - Dashboard history dan analitik event.

### Streaming dan Statistik

- `GET /stream` - SSE untuk update real-time.
- `GET /api/stats` - Statistik agregat dan ringkasan tren.
- `GET /api/camera/<camera_id>/stats` - Statistik per kamera.

### History dan Ekspor

- `GET /api/history` - Data history mentah dan agregasi analitik dari SQLite.
- `GET /api/history/export` - Ekspor CSV history dengan filter tanggal dan scope.
- `GET /api/export` - Ekspor CSV log sesi aktif, atau semua log jika `all=1`.

### Admin dan Kesehatan

- `POST /api/set_count` - Set jumlah orang di dalam ruangan secara manual.
- `POST /api/reset` - Reset sesi dengan password admin.
- `GET /health` - Status kesehatan sistem.

Contoh request untuk kalibrasi manual:

```bash
curl -X POST http://localhost:5000/api/set_count \
  -H "Content-Type: application/json" \
  -d "{\"target_inside\": 12}"
```

Contoh request reset sesi:

```bash
curl -X POST http://localhost:5000/api/reset \
  -H "Content-Type: application/json" \
  -d "{\"password\": \"admin\"}"
```

## Data yang Disimpan

- `state.json` - snapshot state lama dan kompatibilitas balik.
- `edge_vision_counter.db` - SQLite untuk state aktif dan event history.
- `logs/edge_vision_counter.log` - log runtime dengan rotasi file.

## Struktur Proyek

```text
edge-vision-counter/
├── main.py
├── config.py
├── requirements.txt
├── state.json
├── edge_vision_counter.db
├── yolov8n.pt
├── yolov8n_openvino_model/
│   └── yolov8n.xml
├── core/
│   ├── analytics_db.py
│   ├── detector.py
│   ├── state_manager.py
│   ├── tracker_manager.py
│   ├── video_reader.py
│   └── vision_model.py
└── web/
    ├── app.py
    └── templates/
        ├── index.html
        └── history.html
```

## Catatan Operasional

- Password reset default adalah `admin`, dan bisa diubah lewat environment variable `ADMIN_RESET_PASSWORD`.
- Port web default adalah `5000`.
- SSE update interval default berada di `0.25` detik.
- Hostname mDNS default adalah `counter-3wifi`, dan bisa diubah lewat `COUNTER_MDNS_HOSTNAME`.
- Frame normalization default menggunakan mode `letterbox`.
- Jika kamera sering putus, cek nilai `frame_skip`, timeout, dan kualitas jaringan kamera.

## Troubleshooting

Jika aplikasi tidak mau jalan, urutan cek yang paling efektif adalah:

1. Pastikan semua kamera di `config.py` punya URL yang valid dan `enabled` bernilai `True`.
2. Pastikan model `yolov8n.pt` masih ada.
3. Periksa apakah `python main.py` gagal karena dependensi belum terpasang atau backend inference tidak cocok dengan hardware.
4. Buka `http://localhost:5000/health` untuk melihat status kamera dan device inference.
5. Cek `logs/edge_vision_counter.log` jika aplikasi berhenti atau kamera gagal tersambung.

## Teknologi Inti

- YOLOv8n untuk deteksi orang.
- ByteTrack dari `supervision` untuk tracking.
- OpenCV untuk capture dan pemrosesan frame.
- Flask untuk dashboard dan API.
- SQLite untuk riwayat event dan state aplikasi.
