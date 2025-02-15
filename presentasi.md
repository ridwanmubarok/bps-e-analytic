# Analisis Clustering Kejadian Bencana Alam di Indonesia
## Menggunakan Metodologi CRISP-DM

### 🌏 Latar Belakang
Indonesia, sebagai negara kepulauan terbesar di dunia yang terletak di "Ring of Fire" Pasifik, menghadapi tantangan besar dalam hal kebencanaan. Beberapa fakta penting:
- 127 gunung api aktif
- Terletak di pertemuan tiga lempeng tektonik utama
- 60% wilayah daratan rawan banjir
- 40% wilayah berpotensi longsor

### 💡 Tujuan Proyek
1. **Peningkatan Kesiapsiagaan**
   - Identifikasi daerah rawan bencana
   - Sistem peringatan dini
   - Perencanaan evakuasi

2. **Optimalisasi Sumber Daya**
   - Alokasi anggaran efisien
   - Distribusi logistik
   - Penempatan tim tanggap darurat

3. **Mitigasi Bencana**
   - Strategi berbasis data
   - Infrastruktur yang sesuai
   - Edukasi masyarakat

### 📊 Metodologi CRISP-DM
1. **Business Understanding**
   - Analisis kebutuhan stakeholder (BNPB, Pemerintah Daerah)
   - Identifikasi masalah: pemetaan risiko bencana
   - Penentuan success metrics: akurasi clustering dan interpretabilitas hasil
   - Perencanaan proyek dan timeline

2. **Data Understanding**
   - Sumber Data: API BPS (https://webapi.bps.go.id/v1/api/interoperabilitas/)
   - Periode: 2018-2024
   - Variabel:
     * 10 jenis bencana (features)
     * 34 provinsi
     * Data temporal per tahun
   - Kualitas Data:
     * Missing values ditandai dengan '–'
     * Format data JSON bersarang
     * Skala data bervariasi antar jenis bencana

3. **Data Preparation**
   - Pembersihan Data:
     * Konversi missing values ('–') ke nilai 0
     * Penghapusan data total Indonesia
     * Standardisasi format numerik
   - Transformasi:
     * StandardScaler untuk normalisasi
     ```python
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(X)
     ```
   - Feature Engineering:
     * Perhitungan total bencana per provinsi
     * Normalisasi fitur untuk clustering

4. **Modeling - K-Means Clustering**
   
   **Teori K-Means:**
   K-Means adalah algoritma clustering yang membagi n observasi ke dalam k cluster berdasarkan jarak ke centroid terdekat.
   
   **Algoritma:**
   1. Inisialisasi k centroid secara acak
   2. Iterasi sampai konvergen:
      - Assign setiap titik ke centroid terdekat
      - Update posisi centroid
   
   **Rumus:**
   - Jarak Euclidean: d(x,y) = √(Σ(xi - yi)²)
   - Centroid Update: μj = (1/|Cj|)Σxi∈Cj xi
   - Inertia (WCSS): Σ(xi - μc(i))²
   
   **Implementasi:**
   ```python
   kmeans = KMeans(n_clusters=optimal_k, random_state=42)
   df['cluster'] = kmeans.fit_predict(X_scaled)
   ```
   
   **Optimasi Jumlah Cluster:**
   - Silhouette Score untuk evaluasi kualitas cluster
   - Range pengujian: 3-6 cluster
   ```python
   silhouette_scores = []
   K = range(3, 7)
   for k in K:
       kmeans = KMeans(n_clusters=k, random_state=42)
       score = silhouette_score(X_scaled, kmeans.labels_)
       silhouette_scores.append(score)
   ```

5. **Evaluation**
   
   **Metrics & Implementation:**
   1. Silhouette Score:
      - Mengukur seberapa mirip objek dengan clusternya sendiri
      - Range: [-1, 1], semakin tinggi semakin baik
      - Rumus: (b - a) / max(a, b)
        * a = jarak rata-rata ke titik dalam cluster yang sama
        * b = jarak rata-rata ke titik di cluster terdekat lainnya
      ```python
      silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
      ```

   2. Inertia (Within-Cluster Sum of Squares):
      - Mengukur seberapa dekat titik ke centroidnya
      - Semakin kecil semakin baik
      - Rumus: Σ(xi - μc(i))²
      ```python
      inertia = kmeans.inertia_
      ```

   **Analisis Cluster Detail:**
   1. Statistik per Cluster:
      ```python
      cluster_stats = []
      for i in range(len(np.unique(kmeans.labels_))):
          cluster_data = df[df['cluster'] == i]
          stats = {
              'Cluster': i,
              'Jumlah Provinsi': len(cluster_data),
              'Rata-rata Total Bencana': cluster_data['Total Bencana'].mean(),
              'Provinsi dengan Bencana Terbanyak': cluster_data.loc[cluster_data['Total Bencana'].idxmax(), 'Provinsi'],
              'Max Total Bencana': cluster_data['Total Bencana'].max()
          }
          cluster_stats.append(stats)
      ```

   2. Visualisasi Evaluasi:
      - Scatter Matrix Plot:
        * Menampilkan hubungan antar variabel
        * Pemisahan cluster secara visual
        ```python
        fig = px.scatter_matrix(
            df_clustered, 
            dimensions=features,
            color='cluster'
        )
        ```
      
      - Radar Chart Karakteristik:
        * Profil cluster berdasarkan jenis bencana
        ```python
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=cluster_data[features].mean(),
            theta=features,
            fill='toself'
        ))
        ```

   3. Analisis Spasial:
      - Peta Interaktif:
        * Visualisasi distribusi cluster
        * Marker dinamis dengan animasi
        * Popup informasi detail
        ```python
        def create_indonesia_map(df, selected_disaster, features):
            m = folium.Map(
                location=[center_lat, center_long],
                zoom_start=4
            )
            # Marker dan popup implementation
            for idx, row in df.iterrows():
                folium.CircleMarker(
                    location=province_coords,
                    radius=radius,
                    popup=popup_text,
                    color=color
                ).add_to(m)
        ```

   4. Analisis Temporal:
      - Tren per Cluster:
        * Perubahan komposisi cluster
        * Stabilitas keanggotaan cluster
        * Pola musiman

   5. Validasi Hasil:
      - Interpretabilitas:
        * Kesesuaian dengan pengetahuan domain
        * Kejelasan karakteristik cluster
      - Stabilitas:
        * Konsistensi hasil dengan random_state berbeda
        * Robustness terhadap outliers
      - Actionability:
        * Kegunaan praktis hasil clustering
        * Dasar pengambilan keputusan

   6. Dashboard Evaluasi:
      - Komponen Interaktif:
        * Filter temporal
        * Seleksi jenis bencana
        * Perbandingan cluster
      - Metrik Real-time:
        * Update silhouette score
        * Statistik cluster dinamis
      - Visualisasi Responsif:
        * Peta choropleth
        * Grafik tren
        * Analisis komparatif

6. **Deployment**
   
   **Dashboard Streamlit:**
   - Visualisasi interaktif:
     * Peta choropleth Indonesia
     * Grafik tren temporal
     * Analisis cluster
   
   **Fitur Utama:**
   1. Filter temporal
   2. Pemilihan jenis bencana
   3. Analisis komparatif
   4. Eksplorasi data spasial

### 📈 Hasil dan Temuan

**Analisis Cluster:**
1. **Cluster Karakteristik:**
   - Cluster 0: Provinsi dengan risiko tinggi gempa
   - Cluster 1: Daerah rawan banjir dan longsor
   - Cluster 2: Wilayah dengan risiko kebakaran hutan
   - Cluster n: [sesuai optimal_k]

2. **Pola Spasial:**
   - Visualisasi menggunakan folium
   - Marker dinamis untuk hotspot
   - Animasi untuk kejadian aktif

3. **Tren Temporal:**
   - Analisis seasonality
   - Korelasi antar bencana
   - Prediksi pola

### 🎯 Rekomendasi

1. **Kebijakan:**
   - Prioritisasi wilayah berdasarkan cluster
   - Alokasi sumber daya sesuai karakteristik
   - Program mitigasi spesifik

2. **Operasional:**
   - Penempatan tim sesuai zonasi
   - Sistem logistik adaptif
   - Early warning system

3. **Edukasi:**
   - Program awareness sesuai risiko lokal
   - Pelatihan tanggap bencana
   - Sosialisasi peta risiko

### 📱 Pengembangan Kedepan

1. **Teknis:**
   - Integrasi data real-time
   - Machine learning prediktif
   - API untuk mobile apps

2. **Fungsional:**
   - Sistem notifikasi otomatis
   - Modul pelaporan bencana
   - Dashboard mobile

### 👥 Tim Pengembang
[Isi dengan informasi tim]

### 📝 Referensi
1. Badan Pusat Statistik (BPS)
2. Badan Nasional Penanggulangan Bencana (BNPB)
3. Scikit-learn Documentation: K-Means Clustering
4. CRISP-DM Methodology Guide
5. Python Libraries:
   - Streamlit
   - Folium
   - Scikit-learn
   - Pandas
   - NumPy 