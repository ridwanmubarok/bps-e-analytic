import requests  
import pandas as pd  
import numpy as np  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score  
from tabulate import tabulate  
import streamlit as st  
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
import folium
from streamlit_folium import folium_static
  
def get_data_from_api():
    """
    Mengambil data dari API BPS untuk tahun 2018-2024
    """
    base_url = "https://webapi.bps.go.id/v1/api/interoperabilitas/datasource/simdasi/id/25/tahun/{}/id_tabel/MFZ0emxUcHBxOTB2R2F4Sk5oT2hqQT09/wilayah/0000000/key/b3dc419ec75b4bcd83a5ff680035a99e"
    
    all_data = []
    years = range(2018, 2025)  # 2018-2024
    
    for year in years:
        try:
            url = base_url.format(year)
            response = requests.get(url)
            data = response.json()
            
            # Pastikan data memiliki struktur yang diharapkan
            if 'data-availability' in data and data['data-availability'] == 'available':
                # Ekstrak data provinsi (kecuali total Indonesia)
                provinces_data = data['data'][1]['data'][:-1]  # Exclude Indonesia total
                
                # Tambahkan tahun ke setiap record
                for province in provinces_data:
                    province['tahun'] = year
                
                all_data.extend(provinces_data)
            else:
                print(f"Warning: Data not available for year {year}")
                
        except Exception as e:
            print(f"Error fetching data for year {year}: {str(e)}")
    
    print(f"Total records collected: {len(all_data)}")
    return all_data

def prepare_data(data):
    """
    Mempersiapkan data untuk analisis
    """
    provinces_data = []
    
    for item in data:
        try:
            # Periksa struktur data
            if not isinstance(item, dict):
                print(f"Skipping invalid data item: {item}")
                continue
                
            province_dict = {
                'Provinsi': item['label'],
                'Tahun': item['tahun']
            }
            
            # Fungsi untuk membersihkan dan mengkonversi nilai
            def clean_and_convert(value):
                if isinstance(value, dict) and 'value' in value:
                    value = value['value']
                if value == '–' or value == '...' or value is None:
                    return 0
                # Hapus tag HTML dan karakter khusus
                value = str(value).replace('<sup>*</sup>', '').strip()
                try:
                    return float(value)
                except ValueError:
                    print(f"Warning: Could not convert '{value}' to float, using 0")
                    return 0
            
            # Mapping variabel yang tersedia
            variable_mapping = {
                'Gempa Bumi': 'j8qw1r1yab',
                'Tsunami': 'j7n3axnnjx',
                'Gempa Bumi dan Tsunami': 'wrac74bhd0',
                'Letusan Gunung Api': 'rau67lshyc',
                'Tanah Longsor': 'jny6ukynae',
                'Banjir': '2ywwg0i5ag',
                'Kekeringan': 'edibod2oks',
                'Kebakaran Hutan dan Lahan': '2ze602jnr5',
                'Cuaca Ekstrem': 'ezyprnsq7m',
                'Gelombang Pasang/Abrasi': 'xryqgogeqw'
            }
            
            # Tambahkan data bencana
            for bencana, var_id in variable_mapping.items():
                try:
                    # Handle both dict and list type variables
                    if isinstance(item.get('variables', {}), dict):
                        value = item['variables'].get(var_id, {'value': '0'})
                    else:
                        # Jika variables adalah list, cari berdasarkan var_id
                        value = {'value': '0'}
                        for var in item.get('variables', []):
                            if var.get('var_id') == var_id:
                                value = var
                                break
                    
                    province_dict[bencana] = clean_and_convert(value)
                except Exception as e:
                    print(f"Error processing {bencana} for {item.get('label', 'Unknown')}: {str(e)}")
                    province_dict[bencana] = 0
            
            provinces_data.append(province_dict)
            
        except Exception as e:
            print(f"Error processing data item: {str(e)}")
            continue
    
    # Buat DataFrame
    df = pd.DataFrame(provinces_data)
    
    # Debug info
    print(f"Processed {len(provinces_data)} provinces")
    if len(provinces_data) > 0:
        print("Sample data structure:", provinces_data[0])
    print("Columns in DataFrame:", df.columns.tolist())
    
    return df

def perform_clustering(df):
    """
    Melakukan analisis clustering menggunakan K-Means
    """
    # Persiapkan data untuk clustering dengan nama kolom yang sesuai
    features = [
        'Gempa Bumi',
        'Tsunami',
        'Gempa Bumi dan Tsunami',
        'Letusan Gunung Api',
        'Tanah Longsor',
        'Banjir',
        'Kekeringan',
        'Kebakaran Hutan dan Lahan',
        'Cuaca Ekstrem',
        'Gelombang Pasang/Abrasi'
    ]
    
    # Standardisasi fitur
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Tentukan jumlah cluster optimal
    silhouette_scores = []
    K = range(3, 7)  # Ubah range menjadi 3-6 cluster
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    
    optimal_k = K[np.argmax(silhouette_scores)]
    
    # Lakukan clustering dengan jumlah cluster optimal
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    return df, optimal_k, features

def evaluate_clustering(df, X_scaled, kmeans, features):
    """
    Mengevaluasi performa model clustering dengan analisis detail
    """
    # Silhouette score analysis
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    
    # Calculate silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_scaled, kmeans.labels_)
    
    # Detailed silhouette analysis per cluster
    cluster_silhouette_scores = []
    for i in range(len(np.unique(kmeans.labels_))):
        cluster_values = sample_silhouette_values[kmeans.labels_ == i]
        cluster_stats = {
            'cluster': i,
            'mean_score': np.mean(cluster_values),
            'min_score': np.min(cluster_values),
            'max_score': np.max(cluster_values),
            'size': len(cluster_values),
            'std_score': np.std(cluster_values)
        }
        cluster_silhouette_scores.append(cluster_stats)
    
    # Inertia (Within-cluster sum of squares)
    inertia = kmeans.inertia_
    
    # Calculate total bencana untuk setiap provinsi
    df['Total Bencana'] = df[features].sum(axis=1)
    
    # Hitung statistik per cluster
    cluster_stats = []
    for i in range(len(np.unique(kmeans.labels_))):
        cluster_data = df[df['cluster'] == i]
        stats = {
            'Cluster': i,
            'Jumlah Provinsi': len(cluster_data),
            'Rata-rata Total Bencana': cluster_data['Total Bencana'].mean(),
            'Provinsi dengan Bencana Terbanyak': cluster_data.loc[cluster_data['Total Bencana'].idxmax(), 'Provinsi'],
            'Max Total Bencana': cluster_data['Total Bencana'].max(),
            'Silhouette Score': cluster_silhouette_scores[i]['mean_score'],
            'Std Silhouette': cluster_silhouette_scores[i]['std_score']
        }
        
        # Add feature means for each cluster
        for feature in features:
            stats[f'Mean_{feature}'] = cluster_data[feature].mean()
        
        cluster_stats.append(stats)
    
    # Create visualization for silhouette analysis
    def plot_silhouette_analysis():
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Silhouette Plot', 'Cluster Sizes'))
        
        # Silhouette plot
        y_lower = 10
        for i in range(len(np.unique(kmeans.labels_))):
            cluster_silhouette_vals = sample_silhouette_values[kmeans.labels_ == i]
            cluster_silhouette_vals.sort()
            cluster_size = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + cluster_size
            
            fig.add_trace(
                go.Scatter(x=cluster_silhouette_vals,
                          y=np.arange(y_lower, y_upper),
                          name=f'Cluster {i}',
                          mode='lines',
                          showlegend=True),
                row=1, col=1
            )
            y_lower = y_upper + 10
        
        # Add vertical line for average silhouette score
        fig.add_vline(x=silhouette_avg, line_dash="dash", 
                     line_color="red", row=1, col=1)
        
        # Cluster sizes bar plot
        cluster_sizes = [stats['size'] for stats in cluster_silhouette_scores]
        fig.add_trace(
            go.Bar(x=[f'Cluster {i}' for i in range(len(cluster_sizes))],
                  y=cluster_sizes,
                  name='Cluster Size'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Silhouette Analysis",
            height=500,
            showlegend=True
        )
        
        return fig
    
    # Create visualization for feature importance in clusters
    def plot_feature_importance():
        feature_importance = pd.DataFrame(columns=['Feature', 'Cluster', 'Mean Value'])
        
        for i in range(len(np.unique(kmeans.labels_))):
            cluster_data = df[df['cluster'] == i]
            for feature in features:
                feature_importance = pd.concat([feature_importance, pd.DataFrame({
                    'Feature': [feature],
                    'Cluster': [i],
                    'Mean Value': [cluster_data[feature].mean()]
                })])
        
        fig = px.bar(feature_importance,
                    x='Feature',
                    y='Mean Value',
                    color='Cluster',
                    barmode='group',
                    title='Feature Importance per Cluster')
        
        return fig
    
    evaluation_results = {
        'silhouette_avg': silhouette_avg,
        'inertia': inertia,
        'cluster_stats': pd.DataFrame(cluster_stats),
        'silhouette_analysis': plot_silhouette_analysis(),
        'feature_importance': plot_feature_importance(),
        'cluster_silhouette_scores': pd.DataFrame(cluster_silhouette_scores)
    }
    
    return evaluation_results

# Tambahkan fungsi untuk membuat peta
def create_indonesia_map(df, selected_disaster, features):
    """
    Membuat peta choropleth Indonesia
    """
    # Koordinat center Indonesia
    center_lat = -2.5489
    center_long = 118.0149
    
    # Buat peta dasar
    m = folium.Map(location=[center_lat, center_long], 
                   zoom_start=4,
                   tiles='CartoDB positron',
                   width='100%',
                   height='600px')
    
    # Tambahkan marker untuk setiap provinsi
    for idx, row in df.iterrows():
        province_coords = PROVINCE_COORDINATES.get(row['Provinsi'])  # Ubah 'provinsi' menjadi 'Provinsi'
        if province_coords:
            # Hitung ukuran marker berdasarkan jumlah kejadian
            value = row[selected_disaster]
            radius = 10 + (value * 2) if value > 0 else 10
            
            # Buat popup info
            popup_text = f"""
            <div style='width: 200px'>
                <h4>{row['Provinsi']}</h4>
                <b>Jumlah {selected_disaster}:</b> {value}<br>
                <hr>
                <b>Rincian Bencana Lain:</b><br>
                {'<br>'.join([f"{feat}: {row[feat]}" 
                             for feat in features if feat != selected_disaster and row[feat] > 0])}
            </div>
            """
            
            # Tentukan warna berdasarkan nilai relatif terhadap rata-rata
            color = 'red' if value > df[selected_disaster].mean() else 'blue'
            
            # Tambahkan marker
            folium.CircleMarker(
                location=province_coords,
                radius=radius,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
            
            # Tambahkan marker animasi jika ada kejadian
            if value > 0:
                folium.CircleMarker(
                    location=province_coords,
                    radius=radius + 5,
                    popup=folium.Popup(popup_text, max_width=300),
                    color=color,
                    fill=True,
                    fill_opacity=0.3,
                    className='marker-pulse'
                ).add_to(m)
    
    # Tambahkan custom CSS untuk animasi
    custom_css = """
    <style>
    .marker-pulse {
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.5;
            transform: scale(1.5);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    </style>
    """
    m.get_root().header.add_child(folium.Element(custom_css))
    
    return m

# Dictionary koordinat provinsi Indonesia
PROVINCE_COORDINATES = {
    'Aceh': [4.6951, 96.7494],
    'Sumatera Utara': [2.1154, 99.5451],
    'Sumatera Barat': [-0.7399, 100.8000],
    'Riau': [0.2933, 101.7068],
    'Jambi': [-1.6101, 103.6131],
    'Sumatera Selatan': [-3.3194, 104.9144],
    'Bengkulu': [-3.7928, 102.2608],
    'Lampung': [-4.5585, 105.4068],
    'DKI Jakarta': [-6.2088, 106.8456],
    'Jawa Barat': [-6.9175, 107.6191],
    'Jawa Tengah': [-7.1510, 110.1403],
    'DI Yogyakarta': [-7.7956, 110.3695],
    'Jawa Timur': [-7.5360, 112.2384],
    'Banten': [-6.4058, 106.0640],
    'Bali': [-8.3405, 115.0920],
    'Nusa Tenggara Barat': [-8.6529, 116.3506],
    'Nusa Tenggara Timur': [-8.6574, 120.5721],
    'Kalimantan Barat': [0.2787, 111.4753],
    'Kalimantan Tengah': [-1.6813, 113.3823],
    'Kalimantan Selatan': [-3.0926, 115.2838],
    'Kalimantan Timur': [0.5387, 116.4194],
    'Kalimantan Utara': [3.0731, 116.0413],
    'Sulawesi Utara': [0.6246, 123.9750],
    'Sulawesi Tengah': [-1.4300, 121.4456],
    'Sulawesi Selatan': [-3.6687, 119.9740],
    'Sulawesi Tenggara': [-4.1449, 122.1746],
    'Gorontalo': [0.6999, 122.4467],
    'Sulawesi Barat': [-2.8441, 119.2321],
    'Maluku': [-3.2385, 130.1453],
    'Maluku Utara': [1.5709, 127.8087],
    'Papua Barat': [-1.3361, 133.1747],
    'Papua': [-4.2699, 138.0804]
}

def main():
    """
    Fungsi utama yang menjalankan aplikasi Streamlit untuk analisis clustering bencana alam di Indonesia.
    Mengimplementasikan metodologi CRISP-DM dengan tahapan:
    1. Business Understanding
    2. Data Understanding
    3. Data Preparation
    4. Modeling
    5. Evaluation
    6. Deployment
    """
    # Inisialisasi session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'business_understanding'
    
    # Set page layout to wide mode
    st.set_page_config(
        page_title="Analisis Bencana Alam Indonesia",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Add title with name and student ID
    st.title("Analisis dan Pengelompokan Pola Bencana Alam di Indonesia Menggunakan Metode K-Means Clustering")
    
    # Add author info and social media links with custom styling
    st.markdown("""
    <div style='text-align: left; padding: 20px; margin: 20px 0; 
         border: 2px solid #f0f2f6; border-radius: 10px; 
         background-color: #ffffff; max-width: 500px;'>
        <h3 style='font-size: 20px; margin: 0; font-weight: normal; color: #0e1117;'>Ridwan Mubarok</h3>
        <p style='font-size: 16px; color: #666; margin: 5px 0 15px 0;'>230401010053</p>
        <div style='display: flex; gap: 20px;'>
            <a href='https://www.linkedin.com/in/ridwan-mubarok/' target='_blank' 
               style='color: #0077B5; text-decoration: none; display: flex; align-items: center; gap: 5px;'>
                <i class='fab fa-linkedin'></i> LinkedIn
            </a>
            <a href='https://www.instagram.com/amubhya/' target='_blank' 
               style='color: #E4405F; text-decoration: none; display: flex; align-items: center; gap: 5px;'>
                <i class='fab fa-instagram'></i> Instagram
            </a>
            <a href='https://github.com/ridwanmubarok' target='_blank' 
               style='color: #333; text-decoration: none; display: flex; align-items: center; gap: 5px;'>
                <i class='fab fa-github'></i> GitHub
            </a>
            <a href='https://amubhya.com' target='_blank' 
               style='color: #2196F3; text-decoration: none; display: flex; align-items: center; gap: 5px;'>
                <i class='fas fa-globe'></i> Website
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add Font Awesome for social media icons
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    """, unsafe_allow_html=True)

    # Custom CSS untuk menu navigasi
    st.markdown("""
    <style>
    .header-style {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    /* Mengatur container menu */
    .stButton > button {
        width: 100%;
        padding: 10px 5px;
        margin: 0 5px;
    }
    
    /* Mengatur grid columns */
    .menu-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 20px;
    }
    
    div.row-widget.stHorizontalBlock {
        gap: 10px;
        justify-content: center;
        max-width: 1000px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # Horizontal menu using columns with adjusted width
    col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1])
    with col1:
        if st.button("📋 Business\nUnderstanding"):
            st.session_state.current_page = 'business_understanding'
    with col2:
        if st.button("📊 Data\nUnderstanding"):
            st.session_state.current_page = 'data_understanding'
    with col3:
        if st.button("🔄 Data\nPreparation"):
            st.session_state.current_page = 'data_preparation'
    with col4:
        if st.button("🔍 Modeling"):
            st.session_state.current_page = 'modeling'
    with col5:
        if st.button("📈 Evaluation"):
            st.session_state.current_page = 'evaluation'
    with col6:
        if st.button("🚀 Deployment"):
            st.session_state.current_page = 'deployment'
    
    st.write("---")

    # Get and prepare data
    data = get_data_from_api()
    df = prepare_data(data)
    df_clustered, optimal_k, features = perform_clustering(df)
    
    # Perform clustering and get evaluation metrics
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    evaluation_results = evaluate_clustering(df, X_scaled, kmeans, features)

    # Display content based on selected page
    if st.session_state.current_page == 'business_understanding':
        st.header("Business Understanding")
        st.write("""
        ### 🌏 Latar Belakang
        Indonesia, sebagai negara kepulauan terbesar di dunia yang terletak di "Ring of Fire" Pasifik, 
        menghadapi tantangan besar dalam hal kebencanaan. Posisi geografis yang unik ini membuat Indonesia 
        rentan terhadap berbagai jenis bencana alam, mulai dari gempa bumi, tsunami, hingga bencana 
        hidrometeorologi seperti banjir dan tanah longsor.

        Beberapa fakta penting:
        - Indonesia memiliki 127 gunung api aktif
        - Terletak di pertemuan tiga lempeng tektonik utama
        - 60% wilayah daratan rawan banjir
        - 40% wilayah berpotensi longsor
        
        ### 💡 Tujuan Bisnis
        1. **Peningkatan Kesiapsiagaan**
           - Mengidentifikasi daerah-daerah rawan bencana
           - Mempersiapkan sistem peringatan dini yang efektif
           - Merencanakan jalur evakuasi dan titik pengungsian
        
        2. **Optimalisasi Sumber Daya**
           - Mengalokasikan anggaran penanggulangan bencana secara efisien
           - Mendistribusikan peralatan dan logistik sesuai kebutuhan
           - Menempatkan tim tanggap darurat di lokasi strategis
        
        3. **Mitigasi Bencana**
           - Mengembangkan strategi pencegahan berbasis data
           - Membangun infrastruktur yang sesuai dengan karakteristik bencana
           - Mengedukasi masyarakat tentang risiko bencana di wilayahnya
        
        ### 🎯 Tujuan Data Mining
        1. **Pengelompokan Wilayah**
           - Mengidentifikasi provinsi dengan pola bencana serupa
           - Membuat zonasi kerawanan bencana
           - Menentukan prioritas penanganan
        
        2. **Analisis Pola**
           - Menemukan hubungan antar jenis bencana
           - Mengidentifikasi tren kejadian bencana
           - Memprediksi potensi bencana
        
        3. **Pengambilan Keputusan**
           - Mendukung kebijakan berbasis data
           - Mengoptimalkan alokasi sumber daya
           - Meningkatkan efektivitas mitigasi bencana
        """)

    elif st.session_state.current_page == 'data_understanding':
        st.header("Data Understanding")
        
        # Tampilkan data mentah dari API
        st.write("""
        ### 📊 Data Mentah dari API BPS
        Data ini merupakan data asli yang diperoleh dari API BPS tanpa preprocessing.
        """)
        
        # Pilih tahun untuk ditampilkan
        selected_year = st.selectbox(
            "Pilih Tahun:",
            sorted(df['Tahun'].unique()),
            key='year_selector'
        )
        
        # Filter data berdasarkan tahun
        df_year = df[df['Tahun'] == selected_year]
        
        # Tampilkan data untuk tahun yang dipilih
        st.write(f"#### Data Tahun {selected_year}")
        st.dataframe(df_year, use_container_width=True)
        
        # Tampilkan ringkasan statistik per tahun
        st.write("### 📈 Ringkasan Statistik per Tahun")
        yearly_summary = df.groupby('Tahun').agg({
            'Gempa Bumi': 'sum',
            'Tsunami': 'sum',
            'Gempa Bumi dan Tsunami': 'sum',
            'Letusan Gunung Api': 'sum',
            'Tanah Longsor': 'sum',
            'Banjir': 'sum',
            'Kekeringan': 'sum',
            'Kebakaran Hutan dan Lahan': 'sum',
            'Cuaca Ekstrem': 'sum',
            'Gelombang Pasang/Abrasi': 'sum'
        }).round(2)
        
        st.dataframe(yearly_summary, use_container_width=True)
        
        # Visualisasi tren tahunan
        st.write("### 📊 Tren Bencana per Tahun")
        
        # Melting data untuk visualisasi
        yearly_data_melted = yearly_summary.reset_index().melt(
            id_vars=['Tahun'],
            var_name='Jenis Bencana',
            value_name='Jumlah Kejadian'
        )
        
        # Plot tren
        fig = px.line(
            yearly_data_melted,
            x='Tahun',
            y='Jumlah Kejadian',
            color='Jenis Bencana',
            title='Tren Kejadian Bencana per Tahun',
            markers=True
        )
        fig.update_layout(
            xaxis_title="Tahun",
            yaxis_title="Jumlah Kejadian",
            legend_title="Jenis Bencana"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan informasi dataset
        st.write("""
        ### 📋 Informasi Dataset
        - **Sumber Data**: Badan Pusat Statistik (BPS)
        - **Endpoint API**: https://webapi.bps.go.id/v1/api/interoperabilitas/
        - **Rentang Tahun**: 2018-2024
        - **Jumlah Provinsi**: {} provinsi
        - **Jenis Bencana**: {} jenis
        
        ### 🔍 Struktur Data
        - Data berbentuk JSON dengan struktur bersarang
        - Setiap provinsi memiliki data kejadian untuk 10 jenis bencana
        - Nilai '–' menunjukkan tidak ada kejadian
        """.format(len(df['Provinsi'].unique()), len(features)))

    elif st.session_state.current_page == 'data_preparation':
        st.header("Data Preparation")
        
        # Tambahkan filter tahun
        selected_year = st.selectbox(
            "Pilih Tahun:",
            sorted(df['Tahun'].unique()),
            key='prep_year_selector'
        )
        
        # Filter data berdasarkan tahun
        df_year = df[df['Tahun'] == selected_year]
        
        st.write(f"### 🔄 Proses Persiapan Data Tahun {selected_year}")
        st.write("""
        1. **Pembersihan Data**
           - Menghapus data total Indonesia
           - Mengkonversi nilai '–' menjadi 0
           - Mengubah tipe data string menjadi float untuk nilai numerik
        
        2. **Transformasi Data**
           - Standardisasi nama kolom
           - Normalisasi data menggunakan StandardScaler
           - Persiapan fitur untuk clustering
        
        3. **Hasil Transformasi**
           - Data siap untuk proses clustering
           - Semua nilai dalam format numerik
           - Skala data sudah dinormalisasi
        """)
        
        # Tampilkan data setelah preprocessing untuk tahun terpilih
        st.write(f"### 📊 Data Setelah Preprocessing - Tahun {selected_year}")
        st.dataframe(df_year, use_container_width=True)
        
        # Tampilkan statistik deskriptif untuk tahun terpilih
        st.write(f"### 📈 Statistik Deskriptif - Tahun {selected_year}")
        st.dataframe(df_year[features].describe(), use_container_width=True)
        
        # Visualisasi distribusi data untuk tahun terpilih
        st.write(f"### 📊 Distribusi Data per Jenis Bencana - Tahun {selected_year}")
        fig = px.box(
            df_year.melt(id_vars=['Provinsi'], value_vars=features),
            x='variable',
            y='value',
            title=f"Box Plot Distribusi Kejadian Bencana - Tahun {selected_year}"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.current_page == 'modeling':
        st.header("Modeling - K-Means Clustering")
        
        # Tambahkan filter tahun
        selected_year = st.selectbox(
            "Pilih Tahun:",
            sorted(df['Tahun'].unique()),
            key='model_year_selector'
        )
        
        # Filter data berdasarkan tahun
        df_year = df[df['Tahun'] == selected_year]
        
        # Lakukan clustering untuk data tahun terpilih
        df_clustered_year, optimal_k_year, _ = perform_clustering(df_year)
        X_year = df_year[features].values
        X_scaled_year = scaler.fit_transform(X_year)
        kmeans_year = KMeans(n_clusters=optimal_k_year, random_state=42)
        kmeans_year.fit(X_scaled_year)
        
        st.write(f"### 🎯 Hasil Clustering untuk Tahun {selected_year}")
        st.write(f"""
        - **Jumlah Cluster Optimal**: {optimal_k_year}
        - **Silhouette Score**: {silhouette_score(X_scaled_year, kmeans_year.labels_):.3f}
        - **Inertia**: {kmeans_year.inertia_:.2f}
        """)
        
        # Tampilkan karakteristik setiap cluster
        for i in range(optimal_k_year):
            cluster_data = df_year[df_year['cluster'] == i]
            st.write(f"""
            #### Cluster {i}:
            - Jumlah Provinsi: {len(cluster_data)}
            - Provinsi: {', '.join(cluster_data['Provinsi'].tolist())}
            - Karakteristik Utama: {', '.join([f"{feat}" for feat in features if cluster_data[feat].mean() > df[feat].mean()])}
            """)

    elif st.session_state.current_page == 'evaluation':
        st.header("Evaluation")
        
        # Tambahkan filter tahun
        selected_year = st.selectbox(
            "Pilih Tahun:",
            sorted(df['Tahun'].unique()),
            key='eval_year_selector'
        )
        
        # Filter data berdasarkan tahun
        df_year = df[df['Tahun'] == selected_year]
        
        # Lakukan clustering untuk data tahun terpilih
        df_clustered_year, optimal_k_year, _ = perform_clustering(df_year)
        X_year = df_year[features].values
        X_scaled_year = scaler.fit_transform(X_year)
        kmeans_year = KMeans(n_clusters=optimal_k_year, random_state=42)
        kmeans_year.fit(X_scaled_year)
        
        # Get evaluation results
        evaluation_results = evaluate_clustering(df_year, X_scaled_year, kmeans_year, features)
        
        # Display overall metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Silhouette Score", 
                     f"{evaluation_results['silhouette_avg']:.3f}")
        with col2:
            st.metric("Inertia", 
                     f"{evaluation_results['inertia']:.2f}")
        
        # Display silhouette analysis plot
        st.plotly_chart(evaluation_results['silhouette_analysis'],
                       use_container_width=True)
        
        # Display cluster statistics
        st.subheader("Cluster Statistics")
        st.dataframe(evaluation_results['cluster_stats'])
        
        # Display feature importance plot
        st.plotly_chart(evaluation_results['feature_importance'],
                       use_container_width=True)
        
        # Display detailed silhouette scores
        st.subheader("Detailed Silhouette Scores per Cluster")
        st.dataframe(evaluation_results['cluster_silhouette_scores'])
        
        # Add interpretation
        st.write("""
        ### Interpretasi Hasil:
        1. **Silhouette Score** mengukur seberapa mirip objek dengan clusternya sendiri 
           dibandingkan dengan cluster lain:
           - Score > 0.5: Struktur cluster kuat
           - Score 0.25-0.5: Struktur cluster sedang
           - Score < 0.25: Struktur cluster lemah
           
        2. **Inertia** mengukur seberapa compact cluster yang terbentuk (semakin kecil semakin baik)
        
        3. **Analisis per Cluster**:
           - Ukuran cluster yang seimbang menunjukkan clustering yang baik
           - Silhouette score yang konsisten antar cluster menunjukkan stabilitas
        """)

    elif st.session_state.current_page == 'deployment':
        st.header("Deployment - Dashboard Analisis Bencana")
        
        # Tambahkan filter tahun
        selected_year = st.selectbox(
            "Pilih Tahun:",
            sorted(df['Tahun'].unique()),
            key='deploy_year_selector'
        )
        
        # Filter data berdasarkan tahun
        df_year = df[df['Tahun'] == selected_year]
        
        # Lakukan clustering untuk data tahun terpilih
        df_clustered_year, optimal_k_year, _ = perform_clustering(df_year)
        X_year = df_year[features].values
        X_scaled_year = scaler.fit_transform(X_year)
        kmeans_year = KMeans(n_clusters=optimal_k_year, random_state=42)
        kmeans_year.fit(X_scaled_year)
        
        # Hitung evaluasi metrics untuk tahun terpilih
        evaluation_results = evaluate_clustering(df_year, X_scaled_year, kmeans_year, features)
        
        viz_type = st.radio(
            "Pilih Jenis Visualisasi:",
            ["📊 Analisis Cluster", "🌍 Peta Persebaran", "📈 Analisis Tren", "🔍 Insight Detail", "📑 Kesimpulan & Hasil Model"]
        )
        
        if viz_type == "📊 Analisis Cluster":
            st.subheader(f"Analisis Pengelompokan Provinsi - Tahun {selected_year}")
            
            # Display overall metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Silhouette Score", 
                         f"{evaluation_results['silhouette_avg']:.3f}")
            with col2:
                st.metric("Inertia", 
                         f"{evaluation_results['inertia']:.2f}")
            
            # Display silhouette analysis plot
            st.plotly_chart(evaluation_results['silhouette_analysis'],
                           use_container_width=True)
            
            # Display cluster statistics
            st.subheader("Cluster Statistics")
            st.dataframe(evaluation_results['cluster_stats'])
            
            # Display feature importance plot
            st.plotly_chart(evaluation_results['feature_importance'],
                           use_container_width=True)
            
        elif viz_type == "🌍 Peta Persebaran":
            st.subheader(f"Peta Persebaran Bencana - Tahun {selected_year}")
            
            # Pilih jenis bencana
            selected_disaster = st.selectbox(
                "Pilih Jenis Bencana:",
                features,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            # Buat dan tampilkan peta
            m = create_indonesia_map(df_year, selected_disaster, features)
            folium_static(m, width=1300, height=600)
            
            # Tambahkan legenda
            st.markdown("""
            ### Legenda:
            - 🔴 **Merah**: Kejadian di atas rata-rata
            - 🔵 **Biru**: Kejadian di bawah rata-rata
            - ⭕ **Ukuran lingkaran**: Jumlah kejadian
            - ✨ **Animasi**: Menunjukkan kejadian aktif
            """)
            
        elif viz_type == "📈 Analisis Tren":
            st.subheader(f"Analisis Tren Bencana - Tahun {selected_year}")
            
            # Visualisasi distribusi bencana
            fig = px.bar(
                df_year.melt(id_vars=['Provinsi'], value_vars=features),
                x='variable',
                y='value',
                title="Distribusi Kejadian per Jenis Bencana",
                labels={'variable': 'Jenis Bencana', 'value': 'Jumlah Kejadian'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap korelasi
            corr_matrix = df_year[features].corr()
            fig = px.imshow(
                corr_matrix,
                title="Heatmap Korelasi Antar Jenis Bencana",
                labels=dict(color="Korelasi"),
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "🔍 Insight Detail":
            st.subheader(f"Detail Insight per Cluster - Tahun {selected_year}")
            
            # Analisis per cluster
            for cluster_num in range(optimal_k_year):
                st.write(f"### Cluster {cluster_num}")
                cluster_data = df_year[df_year['cluster'] == cluster_num]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Jumlah Provinsi:** {len(cluster_data)}")
                    st.write("**Provinsi dalam cluster ini:**")
                    st.write(", ".join(cluster_data['Provinsi'].tolist()))
                
                with col2:
                    # Radar chart karakteristik cluster
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=cluster_data[features].mean(),
                        theta=features,
                        fill='toself',
                        name=f'Cluster {cluster_num}'
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=False,
                        title=f"Karakteristik Cluster {cluster_num}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.write("---")
        
        elif viz_type == "📑 Kesimpulan & Hasil Model":
            st.subheader(f"Kesimpulan Analisis Clustering Tahun {selected_year}")
            
            # Tampilkan metrik model
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jumlah Cluster Optimal", 
                         f"{optimal_k_year}")
            with col2:
                st.metric("Silhouette Score", 
                         f"{evaluation_results['silhouette_avg']:.3f}")
            with col3:
                st.metric("Inertia", 
                         f"{evaluation_results['inertia']:.2f}")
            
            # Analisis kualitas cluster
            st.write("### 📊 Kualitas Clustering")
            quality_score = evaluation_results['silhouette_avg']
            if quality_score > 0.5:
                quality_text = "SANGAT BAIK"
                explanation = "Model menghasilkan cluster yang sangat terpisah dengan baik"
            elif quality_score > 0.25:
                quality_text = "BAIK"
                explanation = "Model menghasilkan cluster yang cukup terpisah"
            else:
                quality_text = "CUKUP"
                explanation = "Cluster yang dihasilkan memiliki beberapa overlap"
            
            st.info(f"Kualitas Model: **{quality_text}**\n\n{explanation}")
            
            # Karakteristik tiap cluster
            st.write("### 🎯 Karakteristik Cluster")
            cluster_stats = evaluation_results['cluster_stats']
            
            for i in range(optimal_k_year):
                cluster_data = cluster_stats[cluster_stats['Cluster'] == i]
                
                # Hitung bencana dominan
                feature_means = {feat: cluster_data[f'Mean_{feat}'].values[0] 
                               for feat in features}
                dominant_disasters = sorted(feature_means.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:3]
                
                with st.expander(f"Cluster {i}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"""
                        #### Statistik Umum:
                        - Jumlah Provinsi: {cluster_data['Jumlah Provinsi'].values[0]}
                        - Rata-rata Total Bencana: {cluster_data['Rata-rata Total Bencana'].values[0]:.2f}
                        - Silhouette Score: {cluster_data['Silhouette Score'].values[0]:.3f}
                        """)
                    with col2:
                        st.write("#### Bencana Dominan:")
                        for disaster, value in dominant_disasters:
                            st.write(f"- {disaster}: {value:.2f}")
                
                # Tampilkan provinsi dalam cluster
                provinces = df_year[df_year['cluster'] == i]['Provinsi'].tolist()
                st.write("#### Provinsi dalam Cluster:")
                st.write(", ".join(provinces))
            
            # Rekomendasi berdasarkan hasil clustering
            st.write("### 🎯 Rekomendasi Tindakan")
            
            # Generate rekomendasi berdasarkan karakteristik cluster
            for i in range(optimal_k_year):
                cluster_data = cluster_stats[cluster_stats['Cluster'] == i]
                feature_means = {feat: cluster_data[f'Mean_{feat}'].values[0] 
                               for feat in features}
                dominant_disaster = max(feature_means.items(), key=lambda x: x[1])[0]
                
                st.write(f"#### Cluster {i}:")
                
                # Rekomendasi berdasarkan jenis bencana dominan
                if "Gempa" in dominant_disaster:
                    st.write("""
                    - Perkuat infrastruktur tahan gempa
                    - Siapkan sistem peringatan dini
                    - Lakukan simulasi evakuasi rutin
                    """)
                elif "Banjir" in dominant_disaster:
                    st.write("""
                    - Tingkatkan sistem drainase
                    - Buat peta zonasi banjir
                    - Siapkan pompa air dan peralatan evakuasi
                    """)
                elif "Longsor" in dominant_disaster:
                    st.write("""
                    - Lakukan penghijauan di area rawan
                    - Bangun tanggul penahan
                    - Pasang sensor pergerakan tanah
                    """)
                else:
                    st.write("""
                    - Siapkan rencana mitigasi khusus
                    - Tingkatkan awareness masyarakat
                    - Koordinasi dengan BMKG setempat
                    """)
            
            # Kesimpulan akhir
            st.write("### 📝 Kesimpulan Akhir")
            st.write(f"""
            Berdasarkan analisis clustering untuk tahun {selected_year}, dapat disimpulkan:
            
            1. **Kualitas Model**:
               - Model menghasilkan {optimal_k_year} cluster optimal
               - Kualitas clustering {quality_text.lower()} (Silhouette Score: {quality_score:.3f})
               - Pemisahan antar cluster {explanation.lower()}
            
            2. **Pola Bencana**:
               - Terdapat pola spasial yang jelas dalam distribusi bencana
               - Beberapa provinsi menunjukkan karakteristik bencana yang mirip
               - Ada korelasi antara lokasi geografis dan jenis bencana
            
            3. **Implikasi**:
               - Perlu pendekatan mitigasi yang berbeda untuk setiap cluster
               - Alokasi sumber daya dapat dioptimalkan berdasarkan karakteristik cluster
               - Sistem peringatan dini perlu disesuaikan dengan jenis bencana dominan
            """)

if __name__ == "__main__":
    main()

