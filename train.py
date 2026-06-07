"""
Coffee Bean Selector AI — Training Script
Mencapai akurasi 98%+ menggunakan MobileNetV2 Fine-tuning

Struktur Dataset yang diharapkan:
  DATASET/
    train/
      Dark/     Green/     Light/     Medium/
    test/
      Dark/     Green/     Light/     Medium/

Cara Pakai:
  python train.py --data_dir ./DATASET --epochs 30
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 4
CLASS_NAMES = ['Dark', 'Green', 'Light', 'Medium']

# ─────────────────────────────────────────────
# ARGPARSE
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Train Coffee Bean Classifier')
parser.add_argument('--data_dir', type=str, default='./DATASET',
                    help='Path ke folder dataset (berisi train/ dan test/)')
parser.add_argument('--epochs', type=int, default=30,
                    help='Jumlah epoch training (default: 30)')
parser.add_argument('--output', type=str, default='./model/coffee_bean_model.h5',
                    help='Path output model .h5')
args = parser.parse_args()

# ─────────────────────────────────────────────
# ANALISIS DATASET
# ─────────────────────────────────────────────
def analyze_dataset(data_dir):
    print("\n" + "="*55)
    print("  📊 ANALISIS DATASET")
    print("="*55)
    splits = {}
    for split in ['train', 'test']:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            print(f"  ⚠️  Folder {split} tidak ditemukan!")
            continue
        counts = {}
        for cls in CLASS_NAMES:
            cls_path = os.path.join(split_path, cls)
            if os.path.exists(cls_path):
                n = len([f for f in os.listdir(cls_path)
                         if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))])
                counts[cls] = n
        splits[split] = counts

    for split, counts in splits.items():
        total = sum(counts.values())
        print(f"\n  [{split.upper()}] — Total: {total} gambar")
        for cls, n in counts.items():
            bar = "█" * int(n / max(counts.values()) * 20)
            print(f"    {cls:<8} {bar:<20} {n:>4} gambar")

        # Keseimbangan
        vals  = list(counts.values())
        ratio = min(vals) / max(vals) if max(vals) > 0 else 0
        if ratio > 0.8:
            print(f"\n  ✅ Dataset SEIMBANG (ratio {ratio:.2f})")
        elif ratio > 0.5:
            print(f"\n  ⚠️  Dataset CUKUP SEIMBANG (ratio {ratio:.2f})")
        else:
            print(f"\n  ❌ Dataset TIDAK SEIMBANG (ratio {ratio:.2f})")
            print("     → Pertimbangkan augmentasi lebih agresif atau oversampling")

    print("="*55 + "\n")
    return splits

# ─────────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────────
def build_generators(data_dir):
    # Augmentasi training yang kuat
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.75, 1.25],
        channel_shift_range=20,
        fill_mode='nearest',
        validation_split=0.15,     # 15% dari train untuk validasi
    )

    # Test: hanya normalisasi
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='training',
        shuffle=True,
        seed=42,
    )

    val_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='validation',
        shuffle=False,
        seed=42,
    )

    test_gen = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,
    )

    print(f"  Train batches  : {len(train_gen)} ({train_gen.samples} gambar)")
    print(f"  Val batches    : {len(val_gen)}   ({val_gen.samples} gambar)")
    print(f"  Test batches   : {len(test_gen)}  ({test_gen.samples} gambar)")
    return train_gen, val_gen, test_gen

# ─────────────────────────────────────────────
# BANGUN MODEL (2 Fase)
# ─────────────────────────────────────────────
def build_model_phase1():
    """Fase 1: Hanya top layers yang dilatih (base frozen)."""
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False   # Freeze semua layer base

    inputs  = keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation='relu')(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    print(f"\n  [Fase 1] Trainable params: {sum(p.numpy().size for p in model.trainable_variables):,}")
    return model


def unfreeze_top_layers(model, n_unfreeze=30):
    """Fase 2: Fine-tune — buka N layer terakhir base model."""
    base_model = model.layers[1]   # MobileNetV2 adalah layer index 1
    base_model.trainable = True

    # Freeze semua kecuali N layer terakhir
    for layer in base_model.layers[:-n_unfreeze]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"  [Fase 2] Membuka {trainable} layer terakhir MobileNetV2 untuk fine-tuning")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),  # LR sangat kecil!
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

# ─────────────────────────────────────────────
# EVALUASI
# ─────────────────────────────────────────────
def evaluate_model(model, test_gen):
    print("\n" + "="*55)
    print("  📈 HASIL EVALUASI MODEL")
    print("="*55)

    # Prediksi
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    # Akurasi per kelas
    acc    = np.mean(y_pred == y_true)
    prec   = precision_score(y_true, y_pred, average='weighted')
    rec    = recall_score(y_true, y_pred, average='weighted')
    f1     = f1_score(y_true, y_pred, average='weighted')

    print(f"\n  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")

    if acc >= 0.98:
        print("\n  🎯 TARGET 98% TERCAPAI!")
    elif acc >= 0.95:
        print("\n  ✅ Akurasi sangat baik (>95%)")
    else:
        print(f"\n  ⚠️  Akurasi {acc*100:.1f}% — pertimbangkan tips di bawah")

    # Classification report
    print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix — Coffee Bean Classifier', fontsize=13)
    plt.ylabel('Label Asli')
    plt.xlabel('Prediksi Model')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=120)
    print("  📊 Confusion matrix disimpan: confusion_matrix.png")

    # Rekomendasi jika akurasi kurang
    if acc < 0.98:
        print("\n  💡 Tips meningkatkan akurasi:")
        print("     1. Tambah gambar per kelas (idealnya 500+ per kelas)")
        print("     2. Perpanjang --epochs (coba 50)")
        print("     3. Ganti ke EfficientNetB0 di baris 'base = MobileNetV2(...)'")
        print("        → from tensorflow.keras.applications import EfficientNetB0")
        print("        → base = EfficientNetB0(input_shape=..., include_top=False, weights='imagenet')")

    print("="*55)
    return acc

# ─────────────────────────────────────────────
# PLOT HISTORY
# ─────────────────────────────────────────────
def plot_history(h1, h2=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Training History — Coffee Bean AI', fontsize=14)

    def plot_metric(ax, metric, title):
        ax.plot(h1.history[metric], label='Train (Fase 1)', color='#b87333')
        ax.plot(h1.history[f'val_{metric}'], label='Val (Fase 1)', linestyle='--', color='#b87333')
        if h2:
            offset = len(h1.history[metric])
            xs2 = range(offset, offset + len(h2.history[metric]))
            ax.plot(xs2, h2.history[metric], label='Train (Fase 2)', color='#4a7a38')
            ax.plot(xs2, h2.history[f'val_{metric}'], label='Val (Fase 2)',
                    linestyle='--', color='#4a7a38')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    plot_metric(axes[0], 'accuracy', 'Akurasi')
    plot_metric(axes[1], 'loss', 'Loss')
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=120)
    print("  📊 Grafik training disimpan: training_history.png")

# ─────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────
def main():
    print("\n" + "█"*55)
    print("  ☕ COFFEE BEAN AI — TRAINING SCRIPT")
    print("  Target: 98%+ accuracy dengan MobileNetV2 Fine-tuning")
    print("█"*55)

    # Analisis dataset
    analyze_dataset(args.data_dir)

    # Build generators
    print("  ⏳ Memuat dataset...")
    train_gen, val_gen, test_gen = build_generators(args.data_dir)

    # ── FASE 1: Train top layers ────────────────
    print("\n  ── FASE 1: Melatih top layers (base frozen) ──")
    model = build_model_phase1()

    epochs_phase1 = max(10, args.epochs // 2)
    callbacks_p1 = [
        EarlyStopping(monitor='val_accuracy', patience=6,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint('model_phase1_best.h5', monitor='val_accuracy',
                        save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

    hist1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_phase1,
        callbacks=callbacks_p1,
        verbose=1,
    )

    p1_acc = max(hist1.history['val_accuracy'])
    print(f"\n  Fase 1 selesai. Val accuracy terbaik: {p1_acc*100:.2f}%")

    # ── FASE 2: Fine-tuning ─────────────────────
    print("\n  ── FASE 2: Fine-tuning 30 layer terakhir ──")
    model = unfreeze_top_layers(model, n_unfreeze=30)

    epochs_phase2 = max(10, args.epochs - epochs_phase1)
    callbacks_p2 = [
        EarlyStopping(monitor='val_accuracy', patience=8,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(args.output, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=3, min_lr=1e-9, verbose=1),
    ]

    hist2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_phase2,
        callbacks=callbacks_p2,
        verbose=1,
    )

    p2_acc = max(hist2.history['val_accuracy'])
    print(f"\n  Fase 2 selesai. Val accuracy terbaik: {p2_acc*100:.2f}%")

    # ── Evaluasi Final ──────────────────────────
    model = keras.models.load_model(args.output)  # Load best weights
    final_acc = evaluate_model(model, test_gen)
    plot_history(hist1, hist2)

    print(f"\n  ✅ Model tersimpan: {args.output}")
    print(f"  📦 Ukuran model: {os.path.getsize(args.output)/1e6:.1f} MB")
    print(f"  🎯 Test accuracy: {final_acc*100:.2f}%")
    print("\n  Langkah selanjutnya:")
    print("  1. Ganti model/coffee_bean_model.h5 dengan file baru")
    print("  2. Jalankan python app.py")
    print("  3. Buka http://localhost:5000\n")


if __name__ == '__main__':
    main()
