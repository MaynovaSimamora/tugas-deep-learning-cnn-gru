# Tugas Deep Learning — CNN & GRU

Implementasi **dua arsitektur deep learning** dalam satu repositori:

| # | Arsitektur | Tugas | Dataset |
|---|---|---|---|
| 1 | **CNN** (Convolutional Neural Network) | Klasifikasi citra digit 0–9 | [MNIST](http://yann.lecun.com/exdb/mnist/) (70.000 citra 28×28) |
| 2 | **GRU** (Gated Recurrent Unit, keluarga RNN) | Klasifikasi teks SMS spam / bukan spam | [SMS Spam Collection, UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5.574 pesan) |

> **Identitas**
> Nama: _isi nama_
> NIM: _isi NIM_
> Mata kuliah: Deep Learning

---

## Hasil Singkat

| Model | Akurasi Test | Precision (macro) | Recall (macro) | F1 (macro) | Waktu Training* |
|---|---|---|---|---|---|
| CNN — MNIST | **98,55%** | 98,56% | 98,54% | 98,54% | ±132 detik |
| GRU — SMS Spam | **98,26%** | 96,92% | 95,06% | 95,96% | ±6 detik |

\*CPU (tanpa GPU), diukur otomatis oleh masing-masing skrip.

Grafik dan metrik lengkap: [`results/`](results)
- `results/cnn_metrics.json`, `cnn_curves.png`, `cnn_confusion_matrix.png`, `cnn_predictions.png`
- `results/gru_metrics.json`, `gru_curves.png`, `gru_confusion_matrix.png`

---

## Struktur Repositori

```
.
├── src/
│   ├── common.py        # seeding, evaluasi, confusion matrix, plotting
│   ├── train_cnn.py     # arsitektur + pipeline CNN/MNIST
│   └── train_gru.py     # arsitektur + pipeline GRU/SMS Spam
├── results/             # metrik JSON + grafik (dihasilkan saat training)
├── requirements.txt
└── README.md
```

Dataset **tidak** di-commit; keduanya diunduh otomatis saat skrip pertama kali dijalankan.

---

## Cara Menjalankan

```bash
# 1. (opsional) buat virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# 2. install dependensi
pip install -r requirements.txt

# 3. latih CNN (MNIST, ~1 menit di CPU)
python src/train_cnn.py --epochs 2

# 4. latih GRU (SMS Spam, ~30 detik di CPU)
python src/train_gru.py --epochs 6
```

Semua argumen penting tersedia lewat CLI, misalnya
`python src/train_cnn.py --epochs 3 --batch-size 64 --lr 5e-4`.

### Opsi tanpa install lokal: Google Colab

Buka [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) di Colab
(GitHub → file notebook → *Open in Colab*), lalu **Runtime → Run all**.
PyTorch sudah tersedia di Colab; kedua model selesai dilatih ±3 menit.

### Hasil otomatis lewat GitHub Actions

Workflow [`.github/workflows/train.yml`](.github/workflows/train.yml) menjalankan
kedua skrip setiap kali ada push ke `main`, lalu meng-commit metrik & grafik ke
folder `results/`. Progresnya bisa dilihat di tab **Actions** repositori.

---

## 1. CNN — Klasifikasi Digit MNIST

### Dataset
MNIST: 60.000 citra training + 10.000 citra test, grayscale 28×28, 10 kelas.
Dari data training, 5.000 citra dipisahkan sebagai **validation set** (test set tetap murni untuk evaluasi akhir).

### Arsitektur
```
Input (1×28×28)
 └─ Conv2d(1→32, 3×3) + BatchNorm + ReLU
 └─ Conv2d(32→32, 3×3) + BatchNorm + ReLU
 └─ MaxPool2d(2)                      → 32×14×14
 └─ Conv2d(32→64, 3×3) + BatchNorm + ReLU
 └─ MaxPool2d(2)                      → 64×7×7
 └─ Flatten → Linear(3136→128) + ReLU + Dropout(0.3)
 └─ Linear(128→10) → Softmax
```

### Hyperparameter
| Parameter | Nilai |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Epoch | 2 |
| Loss | CrossEntropyLoss |
| Normalisasi | mean 0.1307, std 0.3081 (statistik MNIST) |

### Hasil
Dihasilkan otomatis oleh `src/train_cnn.py` (lihat `results/cnn_metrics.json`).

| Metrik | Nilai |
|---|---|
| Akurasi test | **98,55%** (9.855 / 10.000 citra) |
| Test loss | 0,0425 |
| Akurasi validasi (epoch 2) | 98,12% |
| Waktu training (CPU) | ±132 detik |

Akurasi melompat dari 94,92% (epoch 1) menjadi 98,28% (epoch 2) pada data
training, dan validasi stabil di ±98%. Kelas tersulit adalah digit **9**
(F1 97,49%) dan **8** (recall 96,0%) — keduanya paling sering tertukar dengan
digit berekor serupa. Detail per kelas ada di `cnn_metrics.json`, sedangkan
confusion matrix dan contoh prediksi divisualisasikan di `results/`.

---

## 2. GRU — Klasifikasi SMS Spam

### Dataset
SMS Spam Collection (UCI): 5.574 pesan unik berlabel `ham` (4.825) atau `spam` (747).
Pembagian **stratified** 80% train / 10% validasi / 10% test.

### Preprocessing
1. Case folding + tokenisasi regex `[a-z0-9']+`.
2. Vocabulary dibangun **hanya dari data training** (kata langka → `<unk>`, padding → `<pad>`).
3. Setiap pesan dipotong/di-pad menjadi 40 token.

### Arsitektur
```
Input token IDs (batch × 40)
 └─ Embedding(vocab × 64), padding_idx=0 + Dropout(0.3)
 └─ GRU(64 → 64, batch_first=True)
 └─ Ambil hidden state terakhir (batch × 64) + Dropout(0.3)
 └─ Linear(64 → 2)
```

### Hyperparameter
| Parameter | Nilai |
|---|---|
| Optimizer | Adam |
| Learning rate | 2e-3 |
| Batch size | 64 |
| Epoch | 6 |
| Loss | CrossEntropyLoss dengan **class weight** (menangani ketidakseimbangan ham/spam) |
| Gradient clipping | max norm 1.0 |

### Hasil
Dihasilkan otomatis oleh `src/train_gru.py` (lihat `results/gru_metrics.json`).

| Metrik | Nilai |
|---|---|
| Akurasi test | **98,26%** (508 / 517 pesan) |
| Test loss | 0,1205 |
| Akurasi validasi (epoch 6) | 97,68% |
| Waktu training (CPU) | ±6 detik (vocab 7.783 kata) |

| Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ham | 98,68% | 99,34% | 99,01% | 452 |
| spam | 95,16% | 90,77% | 92,91% | 65 |

Dari 65 SMS spam di data uji, **59 terdeteksi** dan 6 lolos sebagai `ham`;
sebaliknya hanya 3 dari 452 SMS normal yang salah ditandai spam. Artinya model
cukup konservatif: presisi spam tinggi (95,16%) sehingga pengguna jarang
mendapat false alarm, dengan konsekuensi recall spam sedikit lebih rendah.
Grafik learning curve memperlihatkan val loss mulai mendatar setelah epoch 4,
pertanda training sudah konvergen dan epoch tambahan berisiko overfitting.

---

## Analisis & Catatan

- **CNN** unggul untuk data spasial karena convolution memanfaatkan lokasi piksel bertetangga dan weight sharing, sehingga jumlah parameter jauh lebih efisien dibanding MLP dengan input 784 dimensi.
- **GRU** cocok untuk data sekuensial: hidden state membawa konteks kata-kata sebelumnya, sedangkan gate (`update`, `reset`) mengatur informasi yang dipertahankan/dilupakan. GRU dipilih daripada LSTM karena parameter lebih sedikit dan latihannya lebih cepat pada dataset kecil.
- **Ketidakseimbangan kelas** pada data SMS ditangani dengan class weighting agar model tidak bias memprediksi `ham`; recall kelas `spam` menjadi perhatian utama.
- **Validasi** hanya dipakai untuk memantau learning curve; angka akhir dilaporkan pada test set yang tidak pernah dilihat model.

## Referensi

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.
2. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *EMNLP*.
3. Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering. *DOCENG*.
