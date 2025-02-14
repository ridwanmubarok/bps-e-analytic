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

### Implementasi Pengambilan Data
```python
def get_data_from_api():
    """Mengambil data dari API BPS"""
    url = "https://webapi.bps.go.id/v1/api/list/model/data/lang/ind/domain/0000/var/2310/key/[API_KEY]"
    response = requests.get(url)
    return response.json()

def prepare_data(data):
    """Menyiapkan data dari API"""
    formatted_response = {
        "status": "OK",
        "data-availability": "available",
        "var": data.get("var", []),
        "turvar": data.get("turvar", []),
        "labelvervar": data.get("labelvervar", ""),
        "vervar": data.get("vervar", []),
        "tahun": data.get("tahun", []),
        "turtahun": data.get("turtahun", []),
        "metadata": data.get("metadata", {}),
        "datacontent": data.get("datacontent", {})
    }
    return formatted_response
```

### Struktur Data
- Variabel: Nilai ekspor per bulan
- Observasi: Komoditas pertanian
- Format: JSON terstruktur

---

## 3. Data Preparation

### Ekstraksi Fitur
```python
def create_feature_matrix(formatted_response, years):
    """Membuat matriks fitur untuk clustering"""
    features = []
    commodities = []
    
    for komoditas in formatted_response['vervar'][:-1]:
        komoditas_name = komoditas['label']
        komoditas_id = str(komoditas['val'])
        
        # Mengumpulkan nilai bulanan
        monthly_values = []
        for year in years:
            year_suffix = str(year)[-2:]
            for month_num in range(1, 13):
                key = f"{komoditas_id}231001{year_suffix}{month_num}"
                val = formatted_response['datacontent'].get(key)
                if val is not None:
                    monthly_values.append(float(val))
        
        if len(monthly_values) > 0:
            # Menghitung fitur
            mean_export = np.mean(monthly_values)
            std_export = np.std(monthly_values)
            growth_trend = (monthly_values[-1] - monthly_values[0]) / monthly_values[0]
            max_export = np.max(monthly_values)
            min_export = np.min(monthly_values)
            
            features.append([mean_export, std_export, growth_trend, max_export, min_export])
            commodities.append(komoditas_name)
    
    return np.array(features), commodities
```

### Pra-pemrosesan
```python
def process_numeric_data(df):
    """Konversi data ke numeric dengan handling missing values"""
    numeric_df = df.copy()
    for col in numeric_df.columns[1:]:
        numeric_df[col] = pd.to_numeric(numeric_df[col].replace('-', np.nan))
    return numeric_df
```

---

## 4. Modeling

### Slide 1: Pengenalan K-Means Clustering
- **Pengertian**
  * Metode pengelompokan data menjadi K kelompok berdasarkan kemiripan
  * K = jumlah kelompok yang diinginkan
  * Means = pusat kelompok dari rata-rata anggotanya

### Slide 2: Tujuan dan Karakteristik
- **Tujuan**
  * Mengelompokkan data yang mirip dalam satu cluster
  * Memaksimalkan perbedaan antar cluster
  * Menemukan pola dalam data ekspor pertanian

- **Karakteristik**
  * Mudah diimplementasikan
  * Cepat untuk data besar
  * Hasil mudah diinterpretasi

### Slide 3: Perhitungan Utama
- **Fitur yang Dihitung**
  1. Rata-rata Ekspor
     ```python
     mean_export = np.mean(monthly_values)
     # Rumus: Rata-rata = (Jumlah semua nilai) / (Banyak data)
     ```
  
  2. Volatilitas
     ```python
     std_export = np.std(monthly_values)
     # Rumus: Simpangan = √[(Jumlah(nilai - rata-rata)²) / (n-1)]
     ```
  
  3. Pertumbuhan
     ```python
     growth = (nilai_akhir - nilai_awal) / nilai_awal
     ```

### Slide 4: Proses Clustering
1. **Standardisasi Data**
   ```python
   scaled_features = scaler.fit_transform(features)
   # Membuat skala data seragam: (nilai - rata-rata) / simpangan
   ```

2. **Pengelompokan**
   ```python
   kmeans = KMeans(n_clusters=n_clusters)
   clusters = kmeans.fit_predict(scaled_features)
   ```

3. **Evaluasi Kualitas**
   ```python
   silhouette = silhouette_score(scaled_features, clusters)
   # Mengukur kemiripan data dalam cluster (range: -1 hingga 1)
   ```

### Slide 5: Cara Kerja Algoritma
1. **Inisialisasi**
   - Pilih K titik pusat secara acak
   - Hitung jarak setiap data ke pusat

2. **Iterasi**
   - Masukkan data ke kelompok terdekat
   - Perbarui pusat dengan rata-rata anggota
   - Ulangi sampai stabil

3. **Pengukuran Jarak**
   ```python
   # Jarak Euclidean
   distance = √(selisih_x² + selisih_y² + ... + selisih_n²)
   ```

### Slide 6: Implementasi pada Data Ekspor
- **Fitur yang Dianalisis**
  * Rata-rata nilai ekspor → skala bisnis
  * Volatilitas → stabilitas ekspor
  * Tren pertumbuhan → prospek kedepan
  * Nilai max/min → rentang operasional

- **Proses Analisis**
  1. Normalisasi data
  2. Clustering dengan K = 2-5
  3. Evaluasi hasil
  4. Interpretasi kelompok

### Slide 7: Visualisasi Hasil
```python
# Contoh visualisasi 3D
fig = px.scatter_3d(
    cluster_df,
    x='Rata-rata Ekspor',
    y='Volatilitas',
    z='Pertumbuhan',
    color='Cluster',
    title='Pengelompokan Komoditas Ekspor'
)
```

### Slide 8: Interpretasi Hasil
- **Karakteristik Cluster**
  * Cluster 1: Ekspor tinggi dan stabil
  * Cluster 2: Pertumbuhan tinggi
  * Cluster 3: Volatilitas tinggi
  * Cluster 4/5: Variasi lainnya

- **Implikasi Bisnis**
  * Identifikasi komoditas unggulan
  * Strategi pengembangan per kelompok
  * Fokus perbaikan sesuai karakteristik

---

## 5. Evaluation

### Implementasi Evaluasi
```python
# Evaluasi berbagai jumlah cluster
silhouette_scores = []
for k in range(2, 6):
    _, _, score, _ = perform_clustering(features, k)
    silhouette_scores.append(score)

# Visualisasi perbandingan score
fig = px.line(
    x=list(range(2, 6)),
    y=silhouette_scores,
    title='Perbandingan Silhouette Score',
    labels={'x': 'Jumlah Cluster', 'y': 'Silhouette Score'}
)

# Hitung inertia
inertia = KMeans(n_clusters=optimal_k, random_state=42).fit(scaled_features).inertia_
```

---

## 6. Deployment

### Implementasi Streamlit
```python
def main():
    st.set_page_config(
        page_title="Analisis Clustering Ekspor Hasil Pertanian",
        page_icon="📊",
        layout="wide"
    )
    
    # Tabs untuk analisis berbeda
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Awal",
        "Analisis Cluster",
        "Perbandingan Karakteristik",
        "Evaluasi"
    ])
    
    # Implementasi tiap tab
    with tab1:
        st.header("📋 Data Ekspor Hasil Pertanian")
        # ... kode untuk tab Data Awal
    
    with tab2:
        st.header("🔍 Analisis Cluster")
        # ... kode untuk tab Analisis Cluster
    
    with tab3:
        st.header("📊 Perbandingan Karakteristik")
        # ... kode untuk tab Perbandingan
    
    with tab4:
        st.header("📊 Evaluasi Model")
        # ... kode untuk tab Evaluasi
```

---

## Rumus dan Perhitungan yang Digunakan

### 1. Perhitungan Karakteristik Data

#### a. Nilai Rata-rata
```python
mean_export = np.mean(monthly_values)
```
Rumus: Rata-rata = (Jumlah semua nilai) / (Banyak data)
- Menghitung nilai tengah dari data ekspor
- Mengukur pusat atau nilai umum dari ekspor

#### b. Tingkat Perubahan (Volatilitas)
```python
std_export = np.std(monthly_values)
```
Rumus: Simpangan Baku = √[(Jumlah(nilai - rata-rata)²) / (n-1)]
- Mengukur seberapa besar nilai-nilai berbeda dari rata-rata
- Semakin besar nilainya, semakin tidak stabil ekspor

#### c. Pertumbuhan
```python
growth_trend = (monthly_values[-1] - monthly_values[0]) / monthly_values[0]
```
Rumus: Pertumbuhan = (Nilai akhir - Nilai awal) / Nilai awal
- Menghitung persentase kenaikan/penurunan
- Menunjukkan arah perkembangan ekspor

### 2. Pengukuran Kualitas Pengelompokan

#### a. Nilai Kemiripan (Silhouette Score)
```python
silhouette_avg = silhouette_score(scaled_features, clusters)
```
Rumus: Nilai = (Jarak ke cluster lain - Jarak dalam cluster) / Nilai terbesar
- Mengukur seberapa mirip data dengan kelompoknya sendiri
- Nilai 1: sangat mirip, -1: sangat tidak mirip

#### b. Nilai Kekompakan (Inertia)
```python
inertia = kmeans.inertia_
```
Rumus: Total jarak setiap data ke pusat kelompoknya
- Mengukur seberapa dekat data dalam satu kelompok
- Semakin kecil nilainya semakin baik

### 3. Penyesuaian Skala Data

```python
scaled_features = scaler.fit_transform(features)
```
Rumus: Nilai baru = (Nilai asli - Rata-rata) / Simpangan baku
- Membuat semua data memiliki skala yang sama
- Mencegah dominasi variabel dengan nilai besar

### 4. Cara Kerja Pengelompokan (K-Means)

Langkah-langkah:
1. Tentukan pusat kelompok secara acak
2. Ulangi sampai stabil:
   - Masukkan data ke kelompok terdekat
   - Perbarui pusat kelompok
   
Cara mengukur jarak:
Jarak = √(selisih_x² + selisih_y² + ... + selisih_n²)
- Menghitung seberapa jauh satu data dengan data lain
- Digunakan untuk menentukan kelompok terdekat

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