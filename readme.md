# Analisis dan Pengelompokan Pola Bencana Alam di Indonesia Menggunakan Metode K-Means Clustering

## 📋 Deskripsi
Aplikasi berbasis web untuk menganalisis pola dan karakteristik bencana alam di Indonesia menggunakan metode clustering. 
Menggunakan data historis dari BPS (Badan Pusat Statistik) dan teknik machine learning untuk mengidentifikasi kelompok provinsi dengan karakteristik kebencanaan yang serupa.

## 🚀 Fitur Utama
- Visualisasi data historis bencana alam
- Analisis clustering provinsi berdasarkan pola bencana
- Peta interaktif persebaran bencana
- Perbandingan karakteristik antar cluster
- Rekomendasi mitigasi berdasarkan karakteristik cluster
- Tampilan interaktif dengan Streamlit

## 💻 Teknologi yang Digunakan
- Python 3.12.6
- Streamlit
- Scikit-learn (KMeans Clustering)
- Pandas
- NumPy
- Plotly
- Folium
- Requests

## 📊 Materi Presentasi

### 1. Pendahuluan
- Latar Belakang: Indonesia sebagai negara rawan bencana
- Tujuan: Identifikasi pola dan karakteristik bencana
- Manfaat: Peningkatan kesiapsiagaan dan mitigasi bencana

### 2. Metodologi
#### A. Data
- Sumber: API BPS
- Periode: 2018-2024
- Variabel: Jumlah kejadian per jenis bencana

#### B. Teknik Analisis
- Metode: K-Means Clustering
- Karakteristik yang dianalisis:
  * Frekuensi kejadian bencana
  * Jenis bencana dominan
  * Pola spasial
  * Tren temporal

#### C. Visualisasi
- Peta interaktif persebaran bencana
- Analisis cluster
- Tren temporal
- Insight detail per cluster

### 3. Hasil Analisis
#### A. Pembentukan Cluster
- Jumlah optimal cluster
- Karakteristik tiap cluster
- Distribusi provinsi

#### B. Interpretasi
- Pola kebencanaan per cluster
- Tren temporal
- Korelasi antar jenis bencana

#### C. Rekomendasi
- Strategi mitigasi per cluster
- Manajemen risiko
- Peningkatan kesiapsiagaan

## 📦 Cara Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/ridwanmubarok/bps-e-analytic
cd bps-e-analytic
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
- Jumlah kejadian per jenis bencana
- Karakteristik yang diekstrak:
  * Frekuensi kejadian
  * Jenis bencana dominan
  * Pola spasial
  * Tren temporal

## 📈 Metodologi
### 1. K-Means Clustering
- Algoritma pengelompokan unsupervised
- Mengelompokkan provinsi berdasarkan karakteristik bencana
- Optimasi dengan silhouette score

### 2. Fitur yang Dianalisis
- Frekuensi kejadian: mengukur intensitas bencana
- Jenis bencana: mengidentifikasi karakteristik wilayah
- Pola spasial: menganalisis distribusi geografis
- Tren temporal: mengukur perubahan pola

### 3. Visualisasi
- Peta interaktif
- Analisis cluster
- Tren temporal
- Detail insight

## 📱 Tampilan Aplikasi
Aplikasi terdiri dari 6 tab utama:
1. Business Understanding
   - Latar belakang
   - Tujuan
   - Manfaat
2. Data Understanding
   - Sumber data
   - Struktur data
   - Statistik deskriptif
3. Data Preparation
   - Pembersihan data
   - Transformasi
   - Normalisasi
4. Modeling
   - Proses clustering
   - Parameter optimal
   - Hasil pengelompokan
5. Evaluation
   - Metrik evaluasi
   - Analisis cluster
   - Interpretasi hasil
6. Deployment
   - Visualisasi interaktif
   - Insight detail
   - Rekomendasi tindakan

## ⚙️ Requirements
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
requests>=2.31.0
plotly>=5.18.0
folium>=0.14.0
streamlit-folium>=0.15.0
```

## 🤝 Kontribusi
Silakan berkontribusi dengan:
1. Fork repository
2. Buat branch baru (`git checkout -b fitur-baru`)
3. Commit perubahan (`git commit -m 'Menambah fitur baru'`)
4. Push ke branch (`git push origin fitur-baru`)
5. Buat Pull Request

## 👨‍💻 Developer
Ridwan Mubarok (230401010053)
- Website: [amubhya.com](https://amubhya.com)
- LinkedIn: [Ridwan Mubarok](https://www.linkedin.com/in/ridwan-mubarok/)
- GitHub: [ridwanmubarok](https://github.com/ridwanmubarok)
- Instagram: [@amubhya](https://www.instagram.com/amubhya/)

## 🙏 Ucapan Terima Kasih
- Badan Pusat Statistik (BPS) untuk penyediaan data
- Streamlit untuk framework visualisasi
- Scikit-learn untuk tools machine learning

## 📚 Teori dan Metodologi

### 1. K-Means Clustering
Algoritma clustering yang mengelompokkan provinsi berdasarkan kesamaan karakteristik bencana.

#### Prinsip Dasar:
```
1. Inisialisasi k centroid secara acak
2. Assign setiap provinsi ke centroid terdekat
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
- Distribusi provinsi

### 3. Analisis Karakteristik

#### 1. Frekuensi Bencana
- Mengukur intensitas
- Indikator kerawanan
- Basis mitigasi

#### 2. Jenis Bencana
- Karakteristik wilayah
- Pola kejadian
- Fokus penanganan

#### 3. Pola Spasial
- Distribusi geografis
- Korelasi wilayah
- Zonasi risiko

### 4. Implikasi untuk Kebijakan

#### 1. Strategi Mitigasi
- Fokus per cluster
- Alokasi sumber daya
- Prioritas penanganan

#### 2. Manajemen Risiko
- Identifikasi risiko per cluster
- Strategi mitigasi
- Sistem peringatan dini

#### 3. Kesiapsiagaan
- Perencanaan evakuasi
- Penguatan infrastruktur
- Edukasi masyarakat
