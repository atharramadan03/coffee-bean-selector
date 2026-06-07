"""
Coffee Bean Selector AI - Flask Backend
Sistem Cerdas Pemilihan dan Identifikasi Biji Kopi
"""

import os
import io
import time
import base64
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

# ============================================================
# Inisialisasi Flask
# ============================================================
app = Flask(__name__)
CORS(app)

# Konfigurasi
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'coffee_bean_model.h5')
IMG_SIZE = (224, 224)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Label kelas (urutan sesuai training: alphabetical)
CLASS_NAMES = ['Dark', 'Green', 'Light', 'Medium']

# Deskripsi tiap kelas
CLASS_INFO = {
    'Dark': {
        'description': 'Biji kopi sangrai gelap dengan rasa bold, pahit kuat, dan aroma smoky. Cocok untuk espresso dan americano.',
        'characteristics': ['Warna coklat sangat gelap hingga hitam', 'Permukaan berminyak', 'Rasa pahit dominan', 'Kafein lebih rendah'],
        'brew_suggestion': 'Espresso, Americano, Cold Brew',
        'emoji': '☕'
    },
    'Green': {
        'description': 'Biji kopi mentah (unroasted) yang belum melalui proses sangrai. Mengandung klorofil dan antioksidan tinggi.',
        'characteristics': ['Warna hijau keabuan', 'Tekstur keras', 'Aroma seperti rumput segar', 'Kandungan antioksidan tinggi'],
        'brew_suggestion': 'Green Coffee Extract, Supplement',
        'emoji': '🌱'
    },
    'Light': {
        'description': 'Biji kopi sangrai ringan dengan rasa fruity, asam cerah, dan aroma floral. Ideal untuk pour-over dan filter.',
        'characteristics': ['Warna coklat terang', 'Permukaan kering', 'Rasa asam dan fruity', 'Kafein lebih tinggi'],
        'brew_suggestion': 'Pour Over, V60, Chemex, Aeropress',
        'emoji': '🫖'
    },
    'Medium': {
        'description': 'Biji kopi sangrai sedang dengan keseimbangan sempurna antara rasa, keasaman, dan aroma. Pilihan universal.',
        'characteristics': ['Warna coklat sedang', 'Permukaan sedikit berminyak', 'Rasa seimbang', 'Kafein moderat'],
        'brew_suggestion': 'Drip Coffee, French Press, Cappuccino',
        'emoji': '☕'
    }
}

# ============================================================
# Load Model
# ============================================================
print("⏳ Memuat model Coffee Bean AI...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model berhasil dimuat dari: {MODEL_PATH}")
    print(f"   Input shape  : {model.input_shape}")
    print(f"   Output shape : {model.output_shape}")
    print(f"   Total params : {model.count_params():,}")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model = None


# ============================================================
# Helper Functions
# ============================================================
def preprocess_image(image_bytes):
    """Preprocess gambar sebelum dimasukkan ke model."""
    img = Image.open(io.BytesIO(image_bytes))
    
    # Konversi ke RGB jika RGBA atau grayscale
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize ke ukuran input model
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    
    # Normalisasi piksel ke [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Tambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_coffee(image_bytes):
    """Melakukan prediksi jenis biji kopi dari bytes gambar."""
    if model is None:
        raise RuntimeError("Model belum dimuat. Pastikan file model tersedia.")
    
    # Preprocess
    img_array = preprocess_image(image_bytes)
    
    # Prediksi
    start_time = time.time()
    predictions = model.predict(img_array, verbose=0)
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Ambil probabilitas tiap kelas
    probs = predictions[0].tolist()
    
    # Kelas dengan probabilitas tertinggi
    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx]) * 100
    
    # Semua kelas dengan probabilitas
    all_classes = [
        {
            'class': CLASS_NAMES[i],
            'probability': round(float(probs[i]) * 100, 2),
            'emoji': CLASS_INFO[CLASS_NAMES[i]]['emoji']
        }
        for i in range(len(CLASS_NAMES))
    ]
    # Sort by probability descending
    all_classes.sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        'predicted_class': predicted_class,
        'confidence': round(confidence, 2),
        'all_classes': all_classes,
        'inference_time_ms': round(inference_time, 1),
        'class_info': CLASS_INFO[predicted_class],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    """Halaman utama."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint prediksi biji kopi.
    
    Accepts:
        - Form data dengan field 'image' (file upload)
        - JSON dengan field 'image_base64' (base64 encoded image)
    
    Returns:
        JSON dengan hasil prediksi
    """
    try:
        image_bytes = None
        filename = 'unknown'
        
        # Cek apakah ada file upload
        if 'image' in request.files:
            file = request.files['image']
            
            if file.filename == '':
                return jsonify({'error': 'Tidak ada file yang dipilih.'}), 400
            
            # Validasi ekstensi
            allowed_ext = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}
            ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
            if ext not in allowed_ext:
                return jsonify({'error': f'Format file tidak didukung. Gunakan: {", ".join(allowed_ext)}'}), 400
            
            image_bytes = file.read()
            filename = file.filename
            
            # Validasi ukuran file
            if len(image_bytes) > MAX_FILE_SIZE:
                return jsonify({'error': 'Ukuran file terlalu besar. Maksimal 10MB.'}), 400
        
        # Cek apakah ada base64 image (dari kamera)
        elif request.json and 'image_base64' in request.json:
            data_url = request.json['image_base64']
            
            # Strip data URL prefix jika ada
            if ',' in data_url:
                data_url = data_url.split(',')[1]
            
            image_bytes = base64.b64decode(data_url)
            filename = 'camera_capture.jpg'
        
        else:
            return jsonify({'error': 'Tidak ada gambar yang diberikan. Kirim file atau base64 image.'}), 400
        
        # Lakukan prediksi
        result = predict_coffee(image_bytes)
        result['filename'] = filename
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Terjadi kesalahan saat prediksi: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_input_shape': str(model.input_shape) if model else None,
        'classes': CLASS_NAMES,
        'version': '1.0.0'
    })


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Mendapatkan informasi semua kelas."""
    return jsonify({
        'success': True,
        'classes': [
            {
                'name': name,
                'info': info
            }
            for name, info in CLASS_INFO.items()
        ]
    })


# ============================================================
# Error Handlers
# ============================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint tidak ditemukan.'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Terjadi kesalahan pada server.'}), 500


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print(f"\n🚀 Coffee Bean Selector AI berjalan di http://localhost:{port}")
    print(f"   Debug mode: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)
