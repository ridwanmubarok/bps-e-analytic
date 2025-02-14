# Analisis dan Pengelompokan Pola Ekspor Hasil Pertanian Indonesia Menggunakan Metode K-Means Clustering

**Identitas:**
- Nama: Ridwan Mubarok
- NIM: 230401010053
- Mata Kuliah: Data Mining
- Program Studi: Teknik Informatika
- Universitas: Universitas Siber Asia

## Slide 1: Pendahuluan

### Judul Penelitian
"Analisis dan Pengelompokan Pola Ekspor Hasil Pertanian Indonesia Menggunakan Metode K-Means Clustering"

### Identitas Peneliti
- Nama: Ridwan Mubarok
- NIM: 230401010053
- Mata Kuliah: Data Mining
- Program Studi: Teknik Informatika
- Universitas: Universitas Siber Asia

### Latar Belakang Singkat
Sektor pertanian merupakan salah satu pilar utama perekonomian Indonesia, dengan kontribusi signifikan terhadap ekspor nasional. Dalam upaya mengoptimalkan strategi pengembangan ekspor pertanian, diperlukan pemahaman mendalam tentang pola dan karakteristik ekspor setiap komoditas. Analisis clustering menggunakan metode K-Means dapat membantu mengidentifikasi kelompok-kelompok komoditas dengan karakteristik serupa, yang dapat menjadi dasar pengambilan kebijakan yang lebih terarah.

### Metodologi
Penelitian ini menggunakan pendekatan CRISP-DM (Cross-Industry Standard Process for Data Mining) dengan tahapan:
1. Business Understanding: Memahami tujuan dan kebutuhan analisis
2. Data Understanding: Eksplorasi data ekspor dari BPS
3. Data Preparation: Penyiapan dan transformasi data
4. Modeling: Implementasi K-Means Clustering
5. Evaluation: Pengujian dan validasi hasil
6. Deployment: Implementasi sistem analisis

### Ruang Lingkup
- Data: Nilai ekspor hasil pertanian Indonesia
- Periode: 2022-2024
- Sumber: API Badan Pusat Statistik (BPS)
- Metode: K-Means Clustering
- Tools: Python, Streamlit, Scikit-learn

### Manfaat Penelitian
1. **Bagi Pemerintah:**
   - Dasar pengambilan kebijakan pengembangan ekspor
   - Identifikasi komoditas unggulan dan potensial
   - Strategi peningkatan daya saing ekspor

2. **Bagi Pelaku Usaha:**
   - Pemahaman karakteristik pasar ekspor
   - Identifikasi peluang pengembangan
   - Strategi pengelolaan risiko

3. **Bagi Akademisi:**
   - Kontribusi metodologi analisis data ekspor
   - Referensi penelitian terkait
   - Pengembangan model analisis serupa

## Slide 2-4: Business Understanding

### Slide 2: Latar Belakang
**A. Kondisi Saat Ini**
- Indonesia memiliki potensi besar dalam ekspor hasil pertanian
- Data ekspor pertanian tersedia namun belum dimanfaatkan optimal
- Perlu analisis mendalam untuk pengambilan keputusan

**B. Permasalahan yang Ada**
- Belum ada pengelompokan komoditas berdasarkan karakteristik
- Sulit mengidentifikasi pola ekspor tiap komoditas
- Kebijakan pengembangan belum berdasarkan analisis data

**C. Peluang Pengembangan**
- Tersedianya data ekspor dari BPS
- Metode clustering dapat membantu pengelompokan
- Teknologi memungkinkan analisis data yang lebih baik

### Slide 3: Permasalahan
**A. Masalah Utama**
1. Identifikasi Pola:
   - Bagaimana pola ekspor setiap komoditas?
   - Apakah ada tren tertentu dalam ekspor?
   - Bagaimana volatilitas nilai ekspor?

2. Pengelompokan:
   - Bagaimana mengelompokkan komoditas serupa?
   - Apa karakteristik setiap kelompok?
   - Berapa jumlah kelompok yang optimal?

3. Pengembangan:
   - Strategi apa yang sesuai untuk tiap kelompok?
   - Bagaimana meningkatkan performa ekspor?
   - Apa rekomendasi untuk setiap kelompok?

### Slide 4: Tujuan
**A. Tujuan Umum**
- Mengembangkan sistem analisis ekspor hasil pertanian
- Memberikan dasar pengambilan keputusan berbasis data
- Meningkatkan efektivitas strategi pengembangan ekspor

**B. Tujuan Khusus**
1. Analisis Data:
   - Mengidentifikasi pola ekspor setiap komoditas
   - Menghitung karakteristik statistik ekspor
   - Menganalisis tren dan volatilitas

2. Pengelompokan:
   - Menerapkan K-Means Clustering
   - Menentukan jumlah cluster optimal
   - Menganalisis karakteristik setiap cluster

3. Rekomendasi:
   - Menyusun strategi per kelompok
   - Memberikan rekomendasi pengembangan
   - Mengidentifikasi peluang peningkatan

**C. Target Pencapaian**
1. Sistem Analisis:
   - Aplikasi web interaktif
   - Visualisasi hasil clustering
   - Dashboard analisis ekspor

2. Hasil Analisis:
   - Pola ekspor teridentifikasi
   - Kelompok komoditas terbentuk
   - Rekomendasi tersusun

3. Dokumentasi:
   - Laporan analisis lengkap
   - Panduan penggunaan sistem
   - Rekomendasi tindak lanjut

## Slide 5-8: Data Understanding

### Slide 5: Sumber dan Pengumpulan Data
**A. Sumber Data Utama**
- Data bersumber dari API resmi Badan Pusat Statistik (BPS) Indonesia
- Mencakup periode tahun 2022 hingga 2024 (data terkini)
- Fokus pada nilai ekspor bulanan untuk setiap komoditas pertanian

**B. Proses Pengambilan Data**
```python
def get_data_from_api():
    """
    Mengambil data ekspor dari API BPS dengan tahapan:
    1. Koneksi ke endpoint API
    2. Autentikasi dengan API key
    3. Pengambilan data dalam format JSON
    """
    url = "https://webapi.bps.go.id/v1/api/list/..."
    response = requests.get(url)
    return response.json()
```

**C. Karakteristik Data**
- Data time series bulanan
- Nilai dalam satuan ribu USD
- Mencakup berbagai komoditas pertanian

### Slide 6: Struktur dan Organisasi Data
**A. Format Data**
- Data dalam struktur JSON terorganisir
- Hierarki data yang terstruktur dan sistematis
- Kemudahan dalam pengolahan dan analisis

**B. Komponen Data Utama**
1. Informasi Komoditas (vervar):
   - Kode komoditas
   - Nama komoditas
   - Kategori produk

2. Informasi Waktu:
   - Tahun ekspor
   - Bulan ekspor
   - Periode pelaporan

3. Nilai Ekspor (datacontent):
   - Nilai ekspor bulanan
   - Historical data
   - Trend perubahan

### Slide 7: Eksplorasi dan Analisis Awal
**A. Analisis Statistik Deskriptif**
1. Ukuran Data:
   - Jumlah total komoditas yang dianalisis
   - Periode waktu yang tercakup
   - Kelengkapan data per periode

2. Distribusi Nilai:
   - Rentang nilai ekspor
   - Rata-rata dan median
   - Standar deviasi

**B. Visualisasi Data Awal**
1. Tren Temporal:
   - Grafik time series nilai ekspor
   - Pola musiman jika ada
   - Tren jangka panjang

2. Distribusi:
   - Histogram nilai ekspor
   - Box plot per komoditas
   - Scatter plot hubungan antar variabel

### Slide 8: Analisis Kualitas Data
**A. Identifikasi Missing Values**
1. Analisis Kuantitatif:
   - Jumlah data kosong
   - Persentase missing values
   - Pola kemunculan data kosong

2. Strategi Penanganan:
   - Metode imputasi yang sesuai
   - Justifikasi penanganan missing values
   - Impact analysis

**B. Deteksi dan Analisis Outliers**
1. Metodologi:
   - Metode statistik (z-score, IQR)
   - Visualisasi box plot
   - Analisis kontekstual

2. Penanganan Outliers:
   - Verifikasi data ekstrem
   - Strategi treatment
   - Dokumentasi keputusan

**C. Evaluasi Kualitas Data**
1. Konsistensi:
   - Format data
   - Satuan pengukuran
   - Penamaan variabel

2. Validitas:
   - Rentang nilai yang masuk akal
   - Konsistensi temporal
   - Kelengkapan informasi

3. Kebutuhan Preprocessing:
   - Standardisasi format
   - Transformasi data
   - Penanganan noise

## Slide 9-12: Data Preparation

### Slide 9: Preprocessing
- Penanganan missing values
- Standardisasi nilai
- Transformasi data
```python
def process_numeric_data(df):
    numeric_df = df.copy()
    numeric_df[col] = pd.to_numeric(numeric_df[col].replace('-', np.nan))
    return numeric_df
```

### Slide 10: Feature Engineering
- Rata-rata ekspor: `mean_export = np.mean(monthly_values)`
- Volatilitas: `std_export = np.std(monthly_values)`
- Pertumbuhan: `growth = (nilai_akhir - nilai_awal) / nilai_awal`
- Nilai maksimum dan minimum

### Slide 11: Standardisasi
```python
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
```
- Tujuan: Menyamakan skala fitur
- Metode: Z-score standardization
- Rumus: z = (x - μ) / σ

### Slide 12: Hasil Preparation
- Matriks fitur final
- Visualisasi data setelah preprocessing
- Statistik deskriptif fitur

## Slide 13-17: Modeling

### Slide 13: K-Means Clustering
- Pengertian K-Means
- Kelebihan dan keterbatasan
- Parameter yang digunakan

### Slide 14: Algoritma K-Means
1. Inisialisasi
   - Pilih K pusat cluster
   - Hitung jarak data ke pusat
2. Iterasi
   - Assignment: data → cluster terdekat
   - Update: hitung ulang pusat cluster
3. Konvergensi
   - Kriteria berhenti
   - Hasil final

### Slide 15: Implementasi
```python
def perform_clustering(features, n_clusters):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    silhouette_avg = silhouette_score(scaled_features, clusters)
    return clusters, centers, silhouette_avg
```

### Slide 16: Visualisasi Cluster
```python
fig = px.scatter_3d(
    cluster_df,
    x='Rata-rata Ekspor',
    y='Volatilitas',
    z='Pertumbuhan',
    color='Cluster',
    title='Hasil Clustering'
)
```

### Slide 17: Analisis Cluster
- Karakteristik tiap cluster
- Distribusi komoditas
- Interpretasi hasil

## Slide 18-20: Evaluation

### Slide 18: Metrik Evaluasi
1. Silhouette Score
   - Rumus: s = (b - a) / max(a, b)
   - Interpretasi nilai
   - Hasil per jumlah cluster

2. Inertia
   - Within-cluster sum of squares
   - Elbow method
   - Optimal number of clusters

### Slide 19: Validasi Hasil
- Perbandingan antar jumlah cluster
- Stabilitas cluster
- Interpretabilitas hasil

### Slide 20: Insight Bisnis
- Karakteristik cluster
- Implikasi kebijakan
- Rekomendasi pengembangan

## Slide 21-23: Deployment

### Slide 21: Implementasi Sistem
```python
def main():
    st.set_page_config(
        page_title="Analisis Clustering Ekspor",
        layout="wide"
    )
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Awal",
        "Analisis Cluster",
        "Perbandingan",
        "Evaluasi"
    ])
```

### Slide 22: Fitur Aplikasi
1. Data Awal
   - Visualisasi data mentah
   - Statistik deskriptif
2. Analisis Cluster
   - Visualisasi interaktif
   - Detail cluster
3. Evaluasi
   - Metrik performa
   - Interpretasi hasil

### Slide 23: Penggunaan Sistem
- Cara mengakses
- Interpretasi output
- Manfaat sistem

## Slide 24: Kesimpulan
- Ringkasan temuan
- Implikasi praktis
- Saran pengembangan

## Slide 25: Terima Kasih
- Kontak
- Referensi 