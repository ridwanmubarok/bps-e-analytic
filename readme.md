# Sistem Prediksi Nilai Ekspor Hasil Pertanian Indonesia

## 📋 Deskripsi
Aplikasi berbasis web untuk memprediksi nilai ekspor hasil pertanian Indonesia per komoditas dari tahun 2022 hingga 2045. 
Menggunakan data historis dari BPS (Badan Pusat Statistik) dan model machine learning untuk menghasilkan prediksi.

## 🚀 Fitur Utama
- Visualisasi data historis ekspor pertanian
- Prediksi nilai ekspor hingga tahun 2045
- Analisis tren per komoditas
- Evaluasi model dengan metrik R²
- Rekomendasi berdasarkan hasil analisis
- Tampilan interaktif dengan Streamlit

## 💻 Teknologi yang Digunakan
- Python 3.9+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly
- Requests

## �� Demo Aplikasi

### Video Demo
Anda dapat melihat demonstrasi aplikasi melalui link berikut:
[Video Demo Aplikasi](https://drive.google.com/file/d/1r1XesSGJR6Vz9fH_fRJOJvOaRvaaws7r/view?usp=sharing)

### Fitur yang Ditampilkan dalam Video
1. Data Historis
   - Visualisasi data ekspor 2022-2023
   - Tabel dan grafik per komoditas

2. Hasil Prediksi
   - Prediksi nilai ekspor 2024-2045
   - Grafik prediksi interaktif
   - Analisis tren per komoditas

3. Kesimpulan
   - Analisis performa model
   - Rekomendasi berdasarkan tren
   - Metrik evaluasi R²

4. Teori dan Metodologi
   - Penjelasan model regresi linear
   - Metrik evaluasi
   - Proses analisis data


## 📦 Cara Instalasi

### 1. Clone Repository
```bash
git clone [URL_REPOSITORY_ANDA]
cd [NAMA_FOLDER_PROJECT]
```

### 2. Buat Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
streamlit run main.py
```

## 🔑 Konfigurasi API
Untuk menggunakan API BPS, Anda perlu:
1. Mendaftar di [Website BPS](https://webapi.bps.go.id)
2. Dapatkan API key
3. Ganti API key di `main.py`

## 📊 Struktur Data
Data yang digunakan mencakup:
- Nilai ekspor bulanan per komoditas
- Periode data historis: 2022-2023
- Prediksi: 2024-2045

## 📈 Metodologi
Aplikasi menggunakan pendekatan CRISP-DM:
1. Pemahaman Bisnis
2. Pemahaman Data
3. Persiapan Data
4. Pemodelan (Linear Regression)
5. Evaluasi
6. Penerapan

## 📱 Tampilan Aplikasi
Aplikasi terdiri dari 4 tab utama:
1. Data Historis
2. Hasil Prediksi
3. Kesimpulan
4. Teori dan Metodologi

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

### 1. Regresi Linear
Model regresi linear digunakan untuk memprediksi nilai ekspor berdasarkan tren waktu. Model ini cocok untuk prediksi time series karena dapat menangkap tren linear dalam data.

#### Rumus Dasar:
```
Y = β₀ + β₁X + ε
```
Dimana:
- Y = Nilai ekspor (variabel dependen)
- X = Waktu (variabel independen)
- β₀ = Intercept (nilai Y ketika X = 0)
- β₁ = Slope (perubahan Y untuk setiap unit perubahan X)
- ε = Error term (perbedaan antara nilai prediksi dan aktual)

#### Estimasi Parameter:
Menggunakan metode Ordinary Least Squares (OLS):
```
β₁ = Σ((x - x̄)(y - ȳ)) / Σ(x - x̄)²
β₀ = ȳ - β₁x̄
```

### 2. Evaluasi Model

#### 1. R² (Koefisien Determinasi)
```
R² = 1 - (SSres / SStot)
```
Dimana:
- SSres (Jumlah Kuadrat Residual) = Σ(y - ŷ)²
- SStot (Jumlah Kuadrat Total) = Σ(y - ȳ)²
- Rentang nilai: 0-1
  * R² = 1: model sempurna
  * R² = 0: model tidak lebih baik dari rata-rata

#### 2. MSE (Mean Squared Error)
```
MSE = (1/n) * Σ(y - ŷ)²
```
Mengukur rata-rata kesalahan kuadrat prediksi

### 3. Asumsi Model

#### 1. Linearitas
- Hubungan antara X dan Y harus bersifat linear
- Dapat diverifikasi dengan scatter plot dan residual plot

#### 2. Independensi
- Setiap observasi harus independen
- Penting untuk data time series
- Dapat diuji dengan Durbin-Watson test

#### 3. Homoskedastisitas
- Varians error harus konstan
- Diperiksa dengan residual plot
- Pelanggaran dapat menyebabkan estimasi tidak efisien

#### 4. Normalitas
- Residual harus berdistribusi normal
- Diuji dengan:
  * Q-Q plot
  * Uji Shapiro-Wilk
  * Uji Kolmogorov-Smirnov

### 4. Implikasi untuk Prediksi Ekspor

#### 1. Interpretasi Koefisien
- β₁ positif: tren ekspor meningkat
- β₁ negatif: tren ekspor menurun
- Besaran β₁: kecepatan perubahan

#### 2. Keterbatasan Model
- Asumsi tren linear mungkin tidak selalu tepat
- Tidak dapat menangkap perubahan musiman kompleks
- Sensitif terhadap outlier

#### 3. Penggunaan R²
- R² tinggi: prediksi lebih dapat diandalkan
- R² rendah: perlu pertimbangan faktor lain
- Berguna untuk membandingkan reliabilitas prediksi antar komoditas

### 5. Proses Analisis Data

#### 1. Pengumpulan Data
- Sumber: API BPS
- Periode: 2022-2023
- Format: JSON terstruktur

#### 2. Pra-pemrosesan
- Pembersihan data
- Penanganan nilai hilang
- Standardisasi format

#### 3. Pemodelan
- Pembuatan model per komoditas
- Validasi asumsi
- Kalibrasi parameter

#### 4. Evaluasi dan Validasi
- Pengujian akurasi
- Validasi silang
- Analisis residual

#### 5. Visualisasi dan Pelaporan
- Grafik tren
- Tabel prediksi
- Metrik performa
