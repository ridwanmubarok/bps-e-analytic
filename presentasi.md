# Analisis dan Pengelompokan Pola Ekspor Hasil Pertanian Indonesia Menggunakan Metode K-Means Clustering

**Oleh:**
- Nama: Ridwan Mubarok
- NIM: 230401010053

## Daftar Isi
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

---

## 1. Business Understanding

### Latar Belakang
- Ekspor hasil pertanian merupakan sektor penting ekonomi Indonesia
- Perlu pemahaman pola dan karakteristik ekspor tiap komoditas
- Pengelompokan komoditas membantu pengambilan kebijakan

### Tujuan
- Mengidentifikasi pola ekspor komoditas pertanian
- Mengelompokkan komoditas berdasarkan karakteristik
- Menganalisis performa ekspor tiap kelompok

---

## 2. Data Understanding

### Sumber Data
- API Badan Pusat Statistik (BPS)
- Data ekspor bulanan hasil pertanian
- Periode: 2022-2023

### Karakteristik Data
- Time series bulanan
- Multiple komoditas
- Nilai dalam satuan USD

### Struktur Data
- Variabel: Nilai ekspor per bulan
- Observasi: Komoditas pertanian
- Format: JSON terstruktur

---

## 3. Data Preparation

### Ekstraksi Fitur
1. Rata-rata nilai ekspor
2. Volatilitas (standar deviasi)
3. Tren pertumbuhan
4. Nilai maksimum
5. Nilai minimum

### Pra-pemrosesan
- Penanganan missing values
- Standardisasi fitur
- Persiapan data untuk clustering

---

## 4. Modeling

### Metode: K-Means Clustering

#### Parameter
- Jumlah cluster: 2-5
- Random state: 42
- Standardized features

#### Implementasi
- Scikit-learn KMeans
- Visualisasi interaktif dengan Plotly
- Analisis multi-cluster

---

## 5. Evaluation

### Metrik Evaluasi
1. Silhouette Score
   - Mengukur kualitas cluster
   - Range: -1 hingga 1

2. Inertia
   - Within-cluster sum of squares
   - Mengukur kohesi cluster

### Hasil Evaluasi
- Perbandingan performa berbagai jumlah cluster
- Analisis karakteristik tiap cluster
- Interpretasi hasil clustering

---

## 6. Deployment

### Implementasi
- Aplikasi web interaktif dengan Streamlit
- Visualisasi dinamis
- Analisis real-time

### Fitur Aplikasi
1. Data Awal
   - Visualisasi data mentah
   - Statistik deskriptif

2. Analisis Cluster
   - Visualisasi 2D/3D
   - Detail karakteristik

3. Perbandingan Karakteristik
   - Analisis komparatif
   - Distribusi nilai

4. Evaluasi
   - Metrik performa
   - Analisis hasil

---

## Kesimpulan

### Temuan Utama
- Identifikasi pola ekspor yang distinktif
- Pengelompokan komoditas berdasarkan karakteristik
- Evaluasi performa model clustering

### Implikasi
- Pemahaman lebih baik tentang pola ekspor
- Dasar pengambilan kebijakan
- Potensi pengembangan sektor pertanian

---

## Terima Kasih

**Kontak:**
- Nama: Ridwan Mubarok
- NIM: 230401010053 