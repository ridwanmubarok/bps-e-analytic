# Sistem Analisis Clustering Ekspor Hasil Pertanian Indonesia

## 📋 Deskripsi
Aplikasi berbasis web untuk menganalisis pola dan karakteristik ekspor hasil pertanian Indonesia menggunakan metode clustering. 
Menggunakan data historis dari BPS (Badan Pusat Statistik) dan teknik machine learning untuk mengidentifikasi kelompok komoditas dengan karakteristik serupa.

## 🚀 Fitur Utama
- Visualisasi data historis ekspor pertanian
- Analisis clustering komoditas ekspor
- Visualisasi interaktif cluster (2D dan 3D)
- Perbandingan karakteristik antar cluster
- Rekomendasi berdasarkan karakteristik cluster
- Tampilan interaktif dengan Streamlit

## 💻 Teknologi yang Digunakan
- Python 3.12.6
- Streamlit
- Scikit-learn (KMeans Clustering)
- Pandas
- NumPy
- Plotly
- Requests

## 📊 Materi Presentasi

### 1. Pendahuluan
- Latar Belakang: Pentingnya analisis ekspor pertanian
- Tujuan: Identifikasi pola dan karakteristik ekspor
- Manfaat: Pengambilan keputusan berbasis data

### 2. Metodologi
#### A. Data
- Sumber: API BPS
- Periode: Data historis ekspor pertanian
- Variabel: Nilai ekspor bulanan per komoditas

#### B. Teknik Analisis
- Metode: K-Means Clustering
- Karakteristik yang dianalisis:
  * Rata-rata nilai ekspor
  * Volatilitas (standar deviasi)
  * Tren pertumbuhan
  * Nilai maksimum dan minimum

#### C. Visualisasi
- Plot Cluster 2D dan 3D
- Box Plot karakteristik
- Analisis perbandingan

### 3. Hasil Analisis
#### A. Pembentukan Cluster
- Jumlah optimal cluster
- Karakteristik tiap cluster
- Distribusi komoditas

#### B. Interpretasi
- Pola ekspor per cluster
- Tren dan volatilitas
- Potensi pengembangan

#### C. Rekomendasi
- Strategi per cluster
- Manajemen risiko
- Pengembangan pasar

## 📦 Cara Instalasi

### 1. Clone Repository
```bash
git clone [URL_REPOSITORY_ANDA]
cd [NAMA_FOLDER_PROJECT]
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
```bash
streamlit run main.py
```

## 📊 Struktur Data
Data yang digunakan mencakup:
- Nilai ekspor bulanan per komoditas
- Karakteristik yang diekstrak:
  * Rata-rata ekspor
  * Volatilitas
  * Tren pertumbuhan
  * Nilai ekstrem

## 📈 Metodologi
### 1. K-Means Clustering
- Algoritma pengelompokan unsupervised
- Mengelompokkan komoditas berdasarkan karakteristik serupa
- Optimasi dengan metode elbow dan silhouette score

### 2. Fitur yang Dianalisis
- Rata-rata nilai ekspor: mengukur skala ekspor
- Volatilitas: mengukur stabilitas
- Tren pertumbuhan: mengukur perkembangan
- Nilai maksimum/minimum: mengukur rentang nilai

### 3. Visualisasi
- Plot interaktif 2D/3D
- Box plot karakteristik
- Analisis komparatif

## 📱 Tampilan Aplikasi
Aplikasi terdiri dari 3 tab utama:
1. Analisis Cluster
   - Visualisasi cluster
   - Pengaturan parameter
   - Detail karakteristik
2. Perbandingan Karakteristik
   - Analisis komparatif
   - Distribusi nilai
3. Rekomendasi
   - Strategi per cluster
   - Manajemen risiko
   - Pengembangan pasar

## ⚙️ Requirements
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
requests>=2.31.0
plotly>=5.18.0
```

## 🤝 Kontribusi
Silakan berkontribusi dengan:
1. Fork repository
2. Buat branch baru (`git checkout -b fitur-baru`)
3. Commit perubahan (`git commit -m 'Menambah fitur baru'`)
4. Push ke branch (`git push origin fitur-baru`)
5. Buat Pull Request

## 🙏 Ucapan Terima Kasih
- Badan Pusat Statistik (BPS) untuk penyediaan data
- Streamlit untuk framework visualisasi
- Scikit-learn untuk tools machine learning

## 📚 Teori dan Metodologi

### 1. K-Means Clustering
Algoritma clustering yang mengelompokkan data berdasarkan kesamaan karakteristik.

#### Prinsip Dasar:
```
1. Inisialisasi k centroid secara acak
2. Assign setiap data ke centroid terdekat
3. Update posisi centroid
4. Ulangi langkah 2-3 hingga konvergen
```

### 2. Evaluasi Cluster

#### 1. Silhouette Score
```
s = (b - a) / max(a, b)
```
Dimana:
- a = rata-rata jarak ke point dalam cluster yang sama
- b = rata-rata jarak ke point di cluster terdekat
- Rentang nilai: -1 hingga 1

#### 2. Karakteristik Cluster
- Kohesi internal
- Separasi antar cluster
- Distribusi anggota

### 3. Analisis Karakteristik

#### 1. Rata-rata Ekspor
- Mengukur skala operasi
- Indikator kapasitas ekspor
- Benchmark antar komoditas

#### 2. Volatilitas
- Mengukur stabilitas
- Indikator risiko
- Basis manajemen risiko

#### 3. Tren Pertumbuhan
- Arah perkembangan
- Potensi masa depan
- Dasar strategi pengembangan

### 4. Implikasi untuk Kebijakan

#### 1. Strategi Pengembangan
- Fokus per cluster
- Alokasi sumber daya
- Target pengembangan

#### 2. Manajemen Risiko
- Identifikasi risiko per cluster
- Strategi mitigasi
- Monitoring dan evaluasi

#### 3. Pengembangan Pasar
- Target pasar per cluster
- Strategi penetrasi
- Diversifikasi tujuan ekspor
