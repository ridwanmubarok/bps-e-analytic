# Analisis dan Pengelompokan Pola Ekspor Hasil Pertanian Indonesia Menggunakan Metode K-Means Clustering

**Identitas:**
- Nama: Ridwan Mubarok
- NIM: 230401010053

## Slide 1: Pendahuluan
- Judul Penelitian
- Identitas Peneliti
- Outline Presentasi (CRISP-DM Framework)

## Slide 2-4: Business Understanding

### Slide 2: Latar Belakang
- Ekspor pertanian merupakan sektor strategis ekonomi Indonesia
- Perlu analisis mendalam untuk pengembangan sektor
- Clustering dapat membantu memahami karakteristik komoditas

### Slide 3: Permasalahan
- Bagaimana pola ekspor tiap komoditas?
- Bagaimana mengelompokkan komoditas berdasarkan karakteristik?
- Bagaimana strategi pengembangan per kelompok?

### Slide 4: Tujuan
- Mengidentifikasi pola ekspor komoditas
- Mengelompokkan komoditas dengan karakteristik serupa
- Memberikan rekomendasi pengembangan per kelompok

## Slide 5-8: Data Understanding

### Slide 5: Sumber Data
- API Badan Pusat Statistik (BPS)
- Periode: 2022-2024
- Variabel: Nilai ekspor bulanan per komoditas
```python
def get_data_from_api():
    url = "https://webapi.bps.go.id/v1/api/list/..."
    response = requests.get(url)
    return response.json()
```

### Slide 6: Struktur Data
- Format: JSON terstruktur
- Komponen data:
  * Komoditas (vervar)
  * Periode (tahun, bulan)
  * Nilai ekspor (datacontent)

### Slide 7: Eksplorasi Data
- Jumlah komoditas
- Rentang nilai ekspor
- Missing values
- Visualisasi distribusi awal

### Slide 8: Kualitas Data
- Identifikasi missing values
- Outliers
- Konsistensi data
- Kebutuhan preprocessing

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