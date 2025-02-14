import requests  
import pandas as pd  
import numpy as np  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score  
from tabulate import tabulate  
import streamlit as st  
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
  
#==============================================================================  
# 1. PEMAHAMAN BISNIS  
#==============================================================================  
# Tujuan: Memprediksi nilai ekspor hasil pertanian per komoditas hingga 2026
# Target:   
# - Menghasilkan prediksi akurat untuk perencanaan jangka pendek
# - Mengidentifikasi tren ekspor untuk setiap komoditas  
# - Memberikan wawasan untuk pengambilan keputusan
  
#==============================================================================  
# 2. PEMAHAMAN DATA  
#==============================================================================  
def get_data_from_api():  
    """  
    Fase Pengumpulan Data:
    - Sumber: API BPS (Badan Pusat Statistik)
    - Jenis Data: Nilai ekspor bulanan hasil pertanian
    - Periode: Data historis 2022-2023  
    - Format: Respons JSON dengan struktur bersarang
    """  
    url = "https://webapi.bps.go.id/v1/api/list/model/data/lang/ind/domain/0000/var/2310/key/b3dc419ec75b4bcd83a5ff680035a99e"  
    response = requests.get(url)  
    return response.json()  
  
#==============================================================================  
# 3. PERSIAPAN DATA  
#==============================================================================  
def prepare_data(data):  
    """  
    Fase Pra-pemrosesan Data:
    - Standarisasi struktur respons API
    - Penanganan nilai yang hilang dan pemformatan data
    - Memastikan konsistensi data antar periode waktu
    - Menyiapkan struktur data untuk pemodelan
    """  
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
  
#==============================================================================  
# 4. PEMODELAN  
#==============================================================================  
def train_linear_regression(X, y):  
    """  
    Fase Pengembangan Model:
    - Algoritma: Regresi Linear Sederhana
    - Fitur: Variabel berbasis waktu (X)
    - Target: Nilai ekspor (y)
    - Tujuan: Menangkap dan memprediksi tren linear dalam data ekspor
    """  
    model = LinearRegression()  
    model.fit(X, y)  
    return model  
  
#==============================================================================  
# 5. EVALUASI  
#==============================================================================  
def evaluate_model(y_true, y_pred):  
    """  
    Fase Evaluasi Model:
    - Metrik: 
      * MSE: Mengukur akurasi prediksi
      * R²: Mengukur kesesuaian model dan keandalan prediksi
    - Tujuan: Menilai performa model dan tingkat kepercayaan prediksi
    """  
    mse = mean_squared_error(y_true, y_pred)  
    r2 = r2_score(y_true, y_pred)  
    return mse, r2  
  
#==============================================================================  
# 6. PENERAPAN  
#==============================================================================  
def create_historical_table(formatted_response, year):  
    """  
    Penyajian Data Historis:
    - Membuat tabel data historis terstruktur
    - Menangani imputasi nilai yang hilang
    - Memformat data untuk visualisasi dan analisis
    """  
    table_data = []  
    for komoditas in formatted_response['vervar'][:-1]:  # Skip last item (Jumlah)  
        komoditas_id = str(komoditas['val'])  
        komoditas_name = komoditas['label']  
        
        # Extract monthly values  
        monthly_values = []  
        year_suffix = str(year)[-2:]  
        
        for month_num in range(1, 13):  
            key = f"{komoditas_id}231001{year_suffix}{month_num}"  
            val = formatted_response['datacontent'].get(key)  
            monthly_values.append(float(val) if val is not None else None)  
        
        # Impute missing values for November and December 2024
        if year == 2024:
            # Find indices of missing values
            missing_indices = [i for i, v in enumerate(monthly_values) if v is None]
            if missing_indices:
                # Prepare data for regression
                valid_indices = [i for i, v in enumerate(monthly_values) if v is not None]
                X_valid = np.array(valid_indices).reshape(-1, 1)
                y_valid = np.array([monthly_values[i] for i in valid_indices])
                
                # Train regression model
                model = LinearRegression()
                model.fit(X_valid, y_valid)
                
                # Predict missing values
                X_missing = np.array(missing_indices).reshape(-1, 1)
                y_missing_pred = model.predict(X_missing)
                
                # Fill in the missing values
                for i, idx in enumerate(missing_indices):
                    monthly_values[idx] = y_missing_pred[i]
        
        # Hanya tambahkan baris jika ada data  
        if any(v is not None for v in monthly_values):  
            row_data = [komoditas_name]  
            row_data.extend([f"{val:.1f}" if val is not None else "-" for val in monthly_values])  
            table_data.append(row_data)  
    
    # Create headers  
    headers = ['Komoditas']  
    headers.extend([f' {month:02d}' for month in range(1, 13)])  
    df = pd.DataFrame(table_data, columns=headers)
    df.index = range(1, len(df) + 1)  
    
    return df
  
def create_prediction_table(formatted_response, years):  
    """  
    Pembuatan dan Penerapan Prediksi:
    - Menghasilkan prediksi untuk setiap komoditas
    - Membuat tabel prediksi terstruktur
    - Menyertakan metrik performa model
    - Menyiapkan data untuk visualisasi dan analisis
    """  
    # Menyimpan data prediksi untuk semua komoditas  
    prediction_data = {}  
    
    # Process each commodity untuk prediksi  
    for komoditas in formatted_response['vervar'][:-1]:  # Skip last item (Jumlah)  
        komoditas_name = komoditas['label']  
        komoditas_id = str(komoditas['val'])  
        
        # Extract historical monthly values  
        monthly_values = []  
        for year in years:  
            year_suffix = str(year)[-2:]  
            for month_num in range(1, 13):  
                # Format key sesuai dengan format API  
                key = f"{komoditas_id}231001{year_suffix}{month_num}"  
                
                val = formatted_response['datacontent'].get(key)  
                if val is not None:  
                    monthly_values.append(float(val))  
                else:  
                    monthly_values.append(None)  
        
        # Filter out None values for modeling  
        valid_values = [v for v in monthly_values if v is not None]  
        if len(valid_values) > 0:  
            X = np.array(range(len(valid_values))).reshape(-1, 1)  
            y = np.array(valid_values)  
            
            model = train_linear_regression(X, y)  
            y_pred = model.predict(X)  
            
            # Prediksi untuk bulan ke depan hingga 2045
            future_months = np.array(range(len(valid_values), len(valid_values) + (2045 - 2024) * 12)).reshape(-1, 1)  
            future_predictions = model.predict(future_months)  
            
            _, r2 = evaluate_model(y, y_pred)  
            
            prediction_data[komoditas_name] = {  
                'predictions': future_predictions.tolist(),  
                'r2': r2  
            }  
    
    # Fungsi helper untuk membuat DataFrame prediksi per tahun  
    def create_year_prediction(year):  
        table_data = []  
        for komoditas in formatted_response['vervar'][:-1]:  # Skip last item (Jumlah)  
            komoditas_name = komoditas['label']  
            if komoditas_name in prediction_data:  
                predictions = prediction_data[komoditas_name]['predictions']  
                monthly_values = []  
                
                for month in range(12):  
                    idx = (year - 2025) * 12 + month  # Mulai dari 2025  
                    if idx < len(predictions):  
                        monthly_values.append(predictions[idx])  
                    else:  
                        monthly_values.append(None)  
                    
                row_data = [komoditas_name]  
                row_data.extend([f"{val:.1f}" if val is not None else "-" for val in monthly_values])  
                row_data.append(f"{prediction_data[komoditas_name]['r2']:.3f}")  
                table_data.append(row_data)  
        
        headers = ['Komoditas']  
        headers.extend([f' {month:02d}' for month in range(1, 13)])  
        headers.append('R²')  
        df = pd.DataFrame(table_data, columns=headers)
        df.index = range(1, len(df) + 1)  
    
        return df
        
    # Membuat dictionary untuk menyimpan semua DataFrame prediksi  
    prediction_tables = {}  
    for year in range(2025, 2046):  # Prediksi hingga 2045  
        prediction_tables[f'Prediksi {year}'] = create_year_prediction(year)  
    
    return prediction_tables  
  
def create_feature_matrix(formatted_response, years):
    """
    Membuat matriks fitur untuk clustering dengan karakteristik berikut:
    1. Rata-rata nilai ekspor
    2. Standar deviasi (volatilitas)
    3. Tren pertumbuhan
    4. Nilai maksimum
    5. Nilai minimum
    """
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
            growth_trend = (monthly_values[-1] - monthly_values[0]) / monthly_values[0] if monthly_values[0] != 0 else 0
            max_export = np.max(monthly_values)
            min_export = np.min(monthly_values)
            
            features.append([
                mean_export,
                std_export,
                growth_trend,
                max_export,
                min_export
            ])
            commodities.append(komoditas_name)
    
    return np.array(features), commodities

def perform_clustering(features, n_clusters=3):
    """Melakukan clustering menggunakan K-Means"""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    silhouette_avg = silhouette_score(scaled_features, clusters)
    
    return clusters, kmeans.cluster_centers_, silhouette_avg, scaled_features
  
def create_historical_table_raw(formatted_response, year):
    """
    Membuat tabel data historis tanpa normalisasi atau imputasi
    untuk menunjukkan data asli
    """
    table_data = []
    for komoditas in formatted_response['vervar'][:-1]:  # Skip last item (Jumlah)
        komoditas_id = str(komoditas['val'])
        komoditas_name = komoditas['label']
        
        # Extract monthly values tanpa imputasi
        monthly_values = []
        year_suffix = str(year)[-2:]
        
        for month_num in range(1, 13):
            key = f"{komoditas_id}231001{year_suffix}{month_num}"
            val = formatted_response['datacontent'].get(key)
            monthly_values.append(val if val is not None else '-')
        
        # Tambahkan data mentah
        row_data = [komoditas_name]
        row_data.extend(monthly_values)
        table_data.append(row_data)
    
    # Create headers
    headers = ['Komoditas']
    headers.extend([f' {month:02d}' for month in range(1, 13)])
    df = pd.DataFrame(table_data, columns=headers)
    df.index = range(1, len(df) + 1)
    
    return df

def process_numeric_data(df):
    """Helper function untuk mengkonversi data ke numeric dengan handling missing values"""
    numeric_df = df.copy()
    for col in numeric_df.columns[1:]:  # Skip kolom Komoditas
        numeric_df[col] = pd.to_numeric(numeric_df[col].replace('-', np.nan))
    return numeric_df

def main():  
    """  
    Alur Aplikasi Utama:
    1. Pengumpulan Data: Mengambil data dari API BPS
    2. Pra-pemrosesan: Membersihkan dan menyusun data
    3. Pemodelan: Melatih model prediksi
    4. Evaluasi: Menilai performa model
    5. Visualisasi: Menyajikan hasil dan wawasan
    6. Analisis: Memberikan rekomendasi dan kesimpulan
    """  
    st.set_page_config(
        page_title="Analisis Clustering Ekspor Hasil Pertanian",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Get and prepare data
    raw_data = get_data_from_api()
    formatted_response = prepare_data(raw_data)
    years = [int(year['label']) for year in formatted_response['tahun']]
    
    st.title("Analisis dan Pengelompokan Pola Ekspor Hasil Pertanian Indonesia Menggunakan Metode K-Means Clustering")
    st.write("=" * 100)
    
    # Subtitle/deskripsi
    st.write("""
    UJIAN AKHIR SEMESTER - DATA MINING
    """)
    
    # Identitas
    st.write("""
    **Identitas:**
    - Nama: Ridwan Mubarok
    - NIM: 230401010053
    """)
    st.write("=" * 100)
    
    # Create historical tables dengan data mentah
    historical_tables_raw = {}
    for year in range(2022, 2025):
        historical_tables_raw[f'Data {year}'] = create_historical_table_raw(formatted_response, year)
    
    # Create feature matrix
    features, commodities = create_feature_matrix(formatted_response, years)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Data Awal", "Analisis Cluster", "Perbandingan Karakteristik", "Evaluasi"])
    
    with tab1:
        st.header("📋 Data Ekspor Hasil Pertanian")
        
        # 1. Business Understanding
        st.subheader("1️⃣ Deskripsi Data")
        st.write("""
        **Informasi Umum:**
        - Data ekspor hasil pertanian Indonesia
        - Periode data: 2022-2023
        - Sumber: Badan Pusat Statistik (BPS)
        - Jenis data: Time series (bulanan)
        
        **Tujuan Analisis:**
        - Mengidentifikasi pola ekspor komoditas pertanian
        - Mengelompokkan komoditas berdasarkan karakteristik
        - Menganalisis performa ekspor tiap kelompok
        """)
        
        # 2. Data Understanding
        st.subheader("2️⃣ Data Mentah")
        
        # Tampilkan data mentah dalam tabs per tahun
        year_tabs = st.tabs([f"Tahun {year}" for year in range(2022, 2025)])
        
        for tab, (year, df) in zip(year_tabs, historical_tables_raw.items()):
            with tab:
                st.write(f"**Nilai Ekspor Bulanan {year}**")
                st.dataframe(df, use_container_width=True)
                
                # Informasi missing values
                missing_count = df.iloc[:, 1:].eq('-').sum().sum()
                total_cells = df.iloc[:, 1:].size
                missing_percentage = (missing_count / total_cells) * 100
                
                st.write(f"""
                **Informasi Data:**
                - Total data: {total_cells} sel
                - Data kosong: {missing_count} sel
                - Persentase missing: {missing_percentage:.2f}%
                """)
        
        # 3. Statistik Deskriptif
        st.subheader("3️⃣ Statistik Deskriptif")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**A. Ringkasan Statistik**")
            all_data = pd.DataFrame()
            for year, df in historical_tables_raw.items():
                numeric_df = process_numeric_data(df)
                year_df = numeric_df.set_index('Komoditas')
                year_df.columns = [f'{col}_{year}' for col in year_df.columns]
                
                if all_data.empty:
                    all_data = year_df
                else:
                    all_data = pd.concat([all_data, year_df], axis=1)
            
            st.write(all_data.describe())
        
        with col2:
            st.write("**B. Informasi Dataset**")
            st.write(f"- Jumlah Komoditas: {len(commodities)}")
            st.write(f"- Periode Data: {min(years)} - {max(years)}")
            st.write(f"- Total Observasi: {len(all_data.columns) * len(all_data)}")
        
        # 4. Visualisasi Data
        st.subheader("4️⃣ Visualisasi Data")
        
        # Trend total ekspor
        st.write("**A. Trend Total Ekspor**")
        yearly_totals = {}
        for year, df in historical_tables_raw.items():
            numeric_df = process_numeric_data(df)
            yearly_totals[year] = numeric_df.iloc[:, 1:].sum().sum()
        
        fig = px.line(
            x=list(yearly_totals.keys()),
            y=list(yearly_totals.values()),
            title='Total Nilai Ekspor per Tahun',
            labels={'x': 'Tahun', 'y': 'Total Nilai Ekspor'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribusi per komoditas
        st.write("**B. Distribusi per Komoditas**")
        all_commodities_data = pd.DataFrame()
        for year, df in historical_tables_raw.items():
            numeric_df = process_numeric_data(df)
            year_data = numeric_df.melt(
                id_vars=['Komoditas'],
                value_vars=numeric_df.columns[1:],
                var_name='Bulan',
                value_name='Nilai Ekspor'
            )
            year_data['Tahun'] = year
            all_commodities_data = pd.concat([all_commodities_data, year_data])
        
        fig = px.box(
            all_commodities_data,
            x='Komoditas',
            y='Nilai Ekspor',
            color='Tahun',
            title='Distribusi Nilai Ekspor per Komoditas'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🔍 Analisis Cluster")
        
        # Pilihan dimensi untuk visualisasi
        st.subheader("Pengaturan Visualisasi")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_dim = st.selectbox(
                "Pilih Dimensi X:",
                ['Rata-rata Ekspor', 'Volatilitas', 'Tren Pertumbuhan', 'Nilai Maksimum', 'Nilai Minimum'],
                index=0
            )
        with col2:
            y_dim = st.selectbox(
                "Pilih Dimensi Y:",
                ['Volatilitas', 'Rata-rata Ekspor', 'Tren Pertumbuhan', 'Nilai Maksimum', 'Nilai Minimum'],
                index=0
            )
        with col3:
            plot_type = st.radio(
                "Jenis Plot:",
                ["3D Scatter", "2D Scatter"]
            )
            if plot_type == "3D Scatter":
                z_dim = st.selectbox(
                    "Pilih Dimensi Z:",
                    ['Tren Pertumbuhan', 'Rata-rata Ekspor', 'Volatilitas', 'Nilai Maksimum', 'Nilai Minimum'],
                    index=0
                )
        
        # Analisis untuk berbagai jumlah cluster
        st.subheader("Analisis Multi-Cluster")
        
        for n_clusters in range(2, 6):
            st.write(f"### Analisis dengan {n_clusters} Cluster")
            
            # Perform clustering
            clusters, centers, silhouette_avg, scaled_features = perform_clustering(features, n_clusters)
            
            # Create DataFrame with cluster information
            cluster_df = pd.DataFrame({
                'Komoditas': commodities,
                'Cluster': clusters,
                'Rata-rata Ekspor': features[:, 0],
                'Volatilitas': features[:, 1],
                'Tren Pertumbuhan': features[:, 2],
                'Nilai Maksimum': features[:, 3],
                'Nilai Minimum': features[:, 4]
            })
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Visualisasi hasil clustering
                if plot_type == "3D Scatter":
                    fig = px.scatter_3d(
                        cluster_df,
                        x=x_dim,
                        y=y_dim,
                        z=z_dim,
                        color='Cluster',
                        hover_name='Komoditas',
                        title=f'Clustering Komoditas Ekspor ({n_clusters} Cluster)',
                        labels={
                            x_dim: x_dim,
                            y_dim: y_dim,
                            z_dim: z_dim
                        }
                    )
                    fig.update_layout(
                        scene=dict(
                            xaxis_title=x_dim,
                            yaxis_title=y_dim,
                            zaxis_title=z_dim
                        ),
                        height=500
                    )
                else:
                    fig = px.scatter(
                        cluster_df,
                        x=x_dim,
                        y=y_dim,
                        color='Cluster',
                        hover_name='Komoditas',
                        title=f'Clustering Komoditas Ekspor ({n_clusters} Cluster)',
                        labels={
                            x_dim: x_dim,
                            y_dim: y_dim
                        }
                    )
                    fig.update_layout(
                        height=500,
                        xaxis_title=x_dim,
                        yaxis_title=y_dim
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write(f"**Silhouette Score: {silhouette_avg:.3f}**")
                
                # Ringkasan cluster
                st.write("**Ringkasan Cluster:**")
                for i in range(n_clusters):
                    cluster_size = len(cluster_df[cluster_df['Cluster'] == i])
                    st.write(f"Cluster {i}: {cluster_size} komoditas")
            
            # Detail cluster dalam expander
            with st.expander(f"Detail Karakteristik {n_clusters} Cluster"):
                for i in range(n_clusters):
                    st.write(f"**Cluster {i}:**")
                    cluster_commodities = cluster_df[cluster_df['Cluster'] == i]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Statistik:**")
                        st.write(f"- Rata-rata Ekspor: {cluster_commodities['Rata-rata Ekspor'].mean():.2f}")
                        st.write(f"- Volatilitas: {cluster_commodities['Volatilitas'].mean():.2f}")
                        st.write(f"- Tren Pertumbuhan: {cluster_commodities['Tren Pertumbuhan'].mean():.2%}")
                    
                    with col2:
                        st.write("**Komoditas:**")
                        for commodity in cluster_commodities['Komoditas']:
                            st.write(f"- {commodity}")
                    
                    st.write("---")
            
            st.write("---")
        
        # Perbandingan Silhouette Score
        silhouette_scores = []
        for k in range(2, 6):
            _, _, score, _ = perform_clustering(features, k)
            silhouette_scores.append(score)
        
        fig = px.line(
            x=list(range(2, 6)),
            y=silhouette_scores,
            title='Perbandingan Silhouette Score',
            labels={'x': 'Jumlah Cluster', 'y': 'Silhouette Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("📊 Perbandingan Karakteristik")
        
        # Visualisasi karakteristik per cluster
        selected_feature = st.selectbox(
            "Pilih Karakteristik:",
            ['Rata-rata Ekspor', 'Volatilitas', 'Tren Pertumbuhan', 'Nilai Maksimum', 'Nilai Minimum']
        )
        
        fig = px.box(
            cluster_df,
            x='Cluster',
            y=selected_feature,
            points="all",
            hover_name='Komoditas',
            title=f'Distribusi {selected_feature} per Cluster'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis karakteristik
        st.subheader("Analisis Karakteristik Cluster")
        for i in range(n_clusters):
            cluster_data = cluster_df[cluster_df['Cluster'] == i]
            st.write(f"**Cluster {i}:**")
            st.write(f"- Rata-rata ekspor: {cluster_data['Rata-rata Ekspor'].mean():.2f}")
            st.write(f"- Volatilitas rata-rata: {cluster_data['Volatilitas'].mean():.2f}")
            st.write(f"- Tren pertumbuhan rata-rata: {cluster_data['Tren Pertumbuhan'].mean():.2%}")
            st.write("-" * 50)
    
    with tab4:
        st.header("📊 Evaluasi Model")
        
        # Evaluasi performa clustering
        st.subheader("1. Evaluasi Performa Clustering")
        
        # Tampilkan perbandingan Silhouette Score
        silhouette_scores = []
        for k in range(2, 6):
            _, _, score, _ = perform_clustering(features, k)
            silhouette_scores.append(score)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**A. Silhouette Score per Jumlah Cluster**")
            score_df = pd.DataFrame({
                'Jumlah Cluster': range(2, 6),
                'Silhouette Score': silhouette_scores
            })
            st.dataframe(score_df)
            
            # Identifikasi cluster optimal
            optimal_k = score_df.loc[score_df['Silhouette Score'].idxmax(), 'Jumlah Cluster']
            optimal_score = score_df['Silhouette Score'].max()
            
            st.write(f"""
            **Cluster Optimal:**
            - Jumlah Cluster: {optimal_k}
            - Silhouette Score: {optimal_score:.3f}
            """)
        
        with col2:
            fig = px.line(
                score_df,
                x='Jumlah Cluster',
                y='Silhouette Score',
                title='Perbandingan Silhouette Score',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Evaluasi karakteristik cluster
        st.subheader("2. Analisis Cluster Optimal")
        
        # Gunakan cluster optimal
        clusters, centers, silhouette_avg, scaled_features = perform_clustering(features, int(optimal_k))
        
        # Buat DataFrame dengan informasi cluster
        cluster_df = pd.DataFrame({
            'Komoditas': commodities,
            'Cluster': clusters,
            'Rata-rata Ekspor': features[:, 0],
            'Volatilitas': features[:, 1],
            'Tren Pertumbuhan': features[:, 2],
            'Nilai Maksimum': features[:, 3],
            'Nilai Minimum': features[:, 4]
        })
        
        # Tampilkan statistik per cluster
        for i in range(int(optimal_k)):
            cluster_data = cluster_df[cluster_df['Cluster'] == i]
            
            st.write(f"**Cluster {i}**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Statistik Cluster:")
                stats_df = pd.DataFrame({
                    'Metrik': ['Rata-rata Ekspor', 'Volatilitas', 'Tren Pertumbuhan'],
                    'Nilai': [
                        f"{cluster_data['Rata-rata Ekspor'].mean():,.2f}",
                        f"{cluster_data['Volatilitas'].mean():.2f}",
                        f"{cluster_data['Tren Pertumbuhan'].mean():.2%}"
                    ]
                })
                st.dataframe(stats_df)
            
            with col2:
                st.write("Anggota Cluster:")
                st.write(", ".join(cluster_data['Komoditas'].tolist()))
            
            # Visualisasi distribusi karakteristik
            fig = px.box(
                cluster_data,
                y=['Rata-rata Ekspor', 'Volatilitas', 'Tren Pertumbuhan'],
                title=f'Distribusi Karakteristik Cluster {i}'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("-" * 100)
        
        # Evaluasi keseluruhan
        st.subheader("3. Kesimpulan Evaluasi")
        
        # Hitung metrik evaluasi tambahan
        inertia = KMeans(n_clusters=int(optimal_k), random_state=42).fit(scaled_features).inertia_
        
        st.write(f"""
        **Metrik Evaluasi Model:**
        1. Silhouette Score: {optimal_score:.3f}
           - Mengukur seberapa mirip objek dengan cluster-nya sendiri dibandingkan cluster lain
           - Range: -1 hingga 1 (semakin tinggi semakin baik)
           
        2. Inertia (Within-cluster Sum of Squares): {inertia:.2f}
           - Mengukur seberapa dekat points dengan centroid cluster-nya
           - Semakin rendah nilai inertia, semakin kohesif cluster yang terbentuk
        
        **Interpretasi Hasil:**
        1. Kualitas Clustering: {
            "Sangat Baik" if optimal_score > 0.7
            else "Baik" if optimal_score > 0.5
            else "Cukup" if optimal_score > 0.3
            else "Perlu Perbaikan"
        }
        
        2. Karakteristik Cluster:
           - Teridentifikasi {optimal_k} kelompok komoditas dengan karakteristik berbeda
           - Setiap cluster memiliki pola ekspor yang distinktif
           - Pembagian cluster cukup seimbang dari segi jumlah anggota
        """)

if __name__ == "__main__":
    main()  

