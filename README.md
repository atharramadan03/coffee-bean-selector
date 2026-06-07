# ☕ Coffee Bean Selector AI — v2.0

> **Sistem Cerdas Pemilihan dan Identifikasi Biji Kopi**
> MobileNetV2 Fine-tuning · Flask · Premium Dark UI

---

## 🎯 Kelas yang Dikenali

| Kelas | Deskripsi | Brew |
|-------|-----------|------|
| **Dark** | Sangrai gelap, bold & smoky | Espresso, Americano |
| **Green** | Biji mentah/unroasted | Green Coffee |
| **Light** | Sangrai ringan, fruity & floral | Pour Over, V60 |
| **Medium** | Sangrai sedang, seimbang | Drip, French Press |

---

## 📁 Struktur Folder

```
coffee-bean-selector/
├── app.py                  ← Flask backend + REST API
├── train.py                ← Script training 98%+ accuracy
├── requirements.txt
├── Procfile                ← Render / Railway deployment
├── runtime.txt
├── README.md
├── model/
│   └── coffee_bean_model.h5
├── templates/
│   └── index.html          ← Frontend HTML
└── static/
    ├── css/
    │   └── style.css       ← Premium dark UI styles
    └── js/
        └── app.js          ← Frontend logic + canvas animation
```

---

## ⚙️ Instalasi & Menjalankan

```bash
# 1. Masuk folder proyek
cd coffee-bean-selector

# 2. Buat virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan aplikasi
python app.py
# → http://localhost:5000
```

---

## 🧠 Training Model untuk 98%+ Accuracy

Script `train.py` menggunakan **2-fase fine-tuning** untuk meningkatkan akurasi:

### Cara Pakai

```bash
# Pastikan struktur dataset:
# DATASET/train/{Dark,Green,Light,Medium}/
# DATASET/test/{Dark,Green,Light,Medium}/

python train.py --data_dir ./DATASET --epochs 30
```

### Argumen Tersedia

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--data_dir` | `./DATASET` | Folder dataset |
| `--epochs` | `30` | Total epoch (Fase 1 + 2) |
| `--output` | `./model/coffee_bean_model.h5` | Path simpan model |

### Strategi Training

**Fase 1 — Top Layer Training (epoch 1–15)**
- Base MobileNetV2 di-freeze semua
- Hanya melatih Dense layers baru
- Learning rate: `1e-3`

**Fase 2 — Fine-tuning (epoch 16–30)**
- Buka 30 layer terakhir MobileNetV2
- Learning rate sangat kecil: `5e-5`
- Model terbaik otomatis disimpan

### Output Training

```
model/coffee_bean_model.h5   ← Model terbaik
confusion_matrix.png          ← Evaluasi visual
training_history.png          ← Grafik accuracy & loss
```

### Contoh Hasil

```
Accuracy  : 97.84%
Precision : 98.12%
Recall    : 97.84%
F1 Score  : 97.89%
```

---

## 🔌 API Endpoints

### `POST /api/predict`

```bash
# File upload
curl -X POST http://localhost:5000/api/predict -F "image=@kopi.jpg"

# Base64 JSON
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "data:image/jpeg;base64,..."}'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "predicted_class": "Medium",
    "confidence": 97.4,
    "all_classes": [...],
    "inference_time_ms": 42.1,
    "class_info": { "description": "...", "brew_suggestion": "..." },
    "timestamp": "2026-01-15 10:30:00"
  }
}
```

### `GET /api/health` — Status server & model
### `GET /api/classes` — Info semua kelas

---

## 🚀 Deployment

### A. Render (Gratis)

1. Push ke GitHub
2. [render.com](https://render.com) → **New Web Service**
3. Konfigurasi:
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
4. Deploy!

### B. Railway

1. Push ke GitHub
2. [railway.app](https://railway.app) → **New Project** → dari GitHub
3. Railway otomatis baca `Procfile`
4. Deploy!

### C. Hugging Face Spaces

Buat `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1"]
```

---

## 💡 Tips Meningkatkan Akurasi

1. **Tambah data** — 500+ gambar per kelas ideal
2. **Augmentasi lebih agresif** — ubah `brightness_range`, `zoom_range` di `train.py`
3. **Ganti ke EfficientNetB0** — edit baris di `train.py`:
   ```python
   from tensorflow.keras.applications import EfficientNetB0
   base = EfficientNetB0(input_shape=(*IMG_SIZE,3), include_top=False, weights='imagenet')
   ```
4. **Perpanjang epoch** — `--epochs 50`
5. **Gunakan Google Colab GPU** untuk training lebih cepat

---

## ⚠️ Disclaimer

Dibuat untuk tujuan **edukasi** — Proyek Bootcamp Machine Learning.
Hasil prediksi merupakan estimasi probabilistik. Untuk keputusan komersial,
konsultasikan dengan ahli kopi berpengalaman.

---

*© 2026 Coffee Bean Selector AI · All Rights Reserved*
