import requests  
import pandas as pd  
import numpy as np  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score  
from tabulate import tabulate  
import streamlit as st  
  
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
        page_title="Prediksi Ekspor Hasil Pertanian",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Get and prepare data
    raw_data = get_data_from_api()
    formatted_response = prepare_data(raw_data)
    
    # Extract years
    years = [int(year['label']) for year in formatted_response['tahun']]
    
    # Create tables Historical
    historical_tables = {}
    for year in range(2022, 2025):  # Ubah range untuk memasukkan 2024
        historical_tables[f'Data {year}'] = create_historical_table(formatted_response, year)
    
    prediction_tables = create_prediction_table(formatted_response, years)
    
    st.title("Prediksi Nilai Ekspor Hasil Pertanian per Komoditas (2022-2045)")
    st.write("=" * 100)
    
    # Use tabs to separate historical, prediction data, and theory
    tab1, tab2, tab3, tab4 = st.tabs(["Data Historis", "Hasil Prediksi", "Kesimpulan", "Teori dan Metodologi"])
    
    with tab1:
        st.header("📊 Data Historis")
        st.write("Data aktual dari Badan Pusat Statistik (BPS) dengan sumber data: https://www.bps.go.id/id/statistics-table/2/MjMxMCMy/nilai-ekspor-bulanan-hasil-pertanian-menurut-komoditas-.html")
        st.write("-" * 50)
        
        # Display historical tables
        for title, df in historical_tables.items():
            st.subheader(title)
            st.dataframe(df, use_container_width=True, height=400)
            st.write("-" * 50)
        
    with tab2:
        st.header("🔮 Hasil Prediksi")
        st.write("Hasil prediksi menggunakan model Linear Regression")
        st.write("-" * 50)
        
        # Add a slider for selecting prediction year range
        min_year, max_year = 2025, 2045
        selected_years = st.slider(
            "Pilih rentang tahun prediksi:",
            min_value=min_year,
            max_value=max_year,
            value=(2025, 2027)
        )
        
        for year in range(selected_years[0], selected_years[1] + 1):
            title = f'Prediksi {year}'
            if title in prediction_tables:
                st.subheader(title)
                st.dataframe(prediction_tables[title], use_container_width=True, height=300)
                
                commodities = prediction_tables[title]['Komoditas'].tolist()
                tabs = st.tabs(commodities)
                
                for tab, komoditas_name in zip(tabs, commodities):
                    with tab:
                        row = prediction_tables[title][prediction_tables[title]['Komoditas'] == komoditas_name].iloc[0]
                        col_chart, col_info = st.columns([3, 1])
                        
                        with col_chart:
                            st.line_chart(
                                row.drop(['Komoditas', 'R²']).astype(float),
                                height=320,
                                use_container_width=True
                            )
                            st.write("Grafik ini menunjukkan prediksi nilai ekspor bulanan untuk komoditas ini. Perhatikan tren yang diprediksi dan bandingkan dengan data historis.")
                        
                        with col_info:
                            st.metric(
                                label="R² Score",
                                value=f"{float(row['R²']):.3f}"
                            )
                            st.write("R² Score menunjukkan seberapa baik model prediksi sesuai dengan data historis. Nilai mendekati 1 menunjukkan prediksi yang lebih akurat.")
                
                st.write("-" * 50)
    
    with tab3:
        st.header("📈 Kesimpulan dan Analisis")
        st.write("Analisis tren dan kesimpulan prediksi nilai ekspor per komoditas")
        st.write("-" * 50)
        
        # Select year for analysis
        selected_year = st.selectbox(
            "Pilih tahun untuk analisis:",
            range(2025, 2046)
        )
        
        if f'Prediksi {selected_year}' in prediction_tables:
            df = prediction_tables[f'Prediksi {selected_year}']
            
            # Overall trend analysis
            st.subheader("Analisis Tren Keseluruhan")
            
            # Create yearly average for each commodity
            df_numeric = df.iloc[:, 1:-1].astype(float)  # Convert to numeric, excluding Komoditas and R²
            yearly_averages = df_numeric.mean(axis=1)
            
            # Sort commodities by average value
            sorted_df = df.assign(Average=yearly_averages).sort_values('Average', ascending=False)
            
            # Visualize top commodities
            fig_bar = {
                'data': [{
                    'x': sorted_df['Komoditas'],
                    'y': sorted_df['Average'],
                    'type': 'bar',
                    'name': f'Rata-rata Nilai Ekspor {selected_year}'
                }],
                'layout': {
                    'title': f'Perbandingan Rata-rata Nilai Ekspor per Komoditas ({selected_year})',
                    'xaxis': {'title': 'Komoditas'},
                    'yaxis': {'title': 'Rata-rata Nilai Ekspor'},
                    'height': 500
                }
            }
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Detailed analysis for each commodity
            st.subheader("Analisis per Komoditas")
            
            for idx, row in sorted_df.iterrows():
                with st.expander(f"📊 {row['Komoditas']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Monthly trend visualization
                        monthly_values = row.iloc[1:-2].astype(float)  # Exclude Komoditas, R², and Average
                        fig_line = {
                            'data': [{
                                'x': list(range(1, 13)),
                                'y': monthly_values,
                                'type': 'scatter',
                                'mode': 'lines+markers',
                                'name': 'Nilai Ekspor'
                            }],
                            'layout': {
                                'title': f'Tren Bulanan {row["Komoditas"]} ({selected_year})',
                                'xaxis': {'title': 'Bulan'},
                                'yaxis': {'title': 'Nilai Ekspor'},
                                'height': 300
                            }
                        }
                        st.plotly_chart(fig_line, use_container_width=True)
                    
                    with col2:
                        # Key metrics
                        avg_value = monthly_values.mean()
                        max_value = monthly_values.max()
                        min_value = monthly_values.min()
                        r2_score = float(row['R²'])
                        
                        st.metric("Rata-rata Nilai Ekspor", f"{avg_value:.2f}")
                        st.metric("Nilai Tertinggi", f"{max_value:.2f}")
                        st.metric("Nilai Terendah", f"{min_value:.2f}")
                        st.metric("R² Score", f"{r2_score:.3f}")
                        
                        # Trend analysis
                        trend = "Meningkat" if monthly_values.iloc[-1] > monthly_values.iloc[0] else "Menurun"
                        st.write(f"**Tren**: {trend}")
                        
                        # Reliability assessment
                        if r2_score >= 0.8:
                            reliability = "Tinggi"
                        elif r2_score >= 0.6:
                            reliability = "Sedang"
                        else:
                            reliability = "Rendah"
                        st.write(f"**Reliabilitas Prediksi**: {reliability}")
                    
                    # Recommendations
                    st.write("#### Rekomendasi:")
                    if trend == "Meningkat":
                        st.write("- Potensi peningkatan ekspor di masa depan")
                        st.write("- Pertimbangkan peningkatan produksi")
                    else:
                        st.write("- Perlu evaluasi faktor penurunan")
                        st.write("- Pertimbangkan strategi peningkatan daya saing")
                    
                    if reliability == "Rendah":
                        st.write("- Perlu analisis tambahan dengan mempertimbangkan faktor eksternal")
    
    with tab4:
        st.header("📚 Teori dan Metodologi")
        st.write("""
             ### Teori dan Metodologi
             
             #### 1. Linear Regression
             Model regresi linear digunakan untuk memprediksi nilai ekspor berdasarkan tren waktu. Model ini cocok untuk prediksi time series karena dapat menangkap tren linear dalam data.
             
             Rumus dasar:
             ```
             Y = β₀ + β₁X + ε
             ```
             Dimana:
             - Y = Nilai ekspor (variabel dependen)
             - X = Waktu (variabel independen)
             - β₀ = Intercept (nilai Y ketika X = 0)
             - β₁ = Slope (perubahan Y untuk setiap unit perubahan X)
             - ε = Error term (perbedaan antara nilai prediksi dan aktual)
             
             Estimasi parameter menggunakan metode Ordinary Least Squares (OLS):
             ```
             β₁ = Σ((x - x̄)(y - ȳ)) / Σ(x - x̄)²
             β₀ = ȳ - β₁x̄
             ```
             
             #### 2. Evaluasi Model
             Model dievaluasi menggunakan beberapa metrik:
             
             1. R² (R-squared / Koefisien Determinasi):
             ```         R² = 1 - (SSres / SStot)
             ```
             Dimana:
             - SSres (Sum of Squared Residuals) = Σ(y - ŷ)² 
               Mengukur variasi yang tidak dapat dijelaskan oleh model
             - SStot (Total Sum of Squares) = Σ(y - ȳ)² 
               Mengukur total variasi dalam data
             - R² range: 0-1
               * R² = 1: model sempurna
               * R² = 0: model tidak lebih baik dari rata-rata
             
             2. MSE (Mean Squared Error):
             ```
             MSE = (1/n) * Σ(y - ŷ)²
             ```
             Mengukur rata-rata kesalahan kuadrat prediksi
             
             #### 3. Asumsi Model
             Model Linear Regression memiliki beberapa asumsi penting:
             
             1. Linearitas
             - Hubungan antara X dan Y harus bersifat linear
             - Dapat diverifikasi dengan scatter plot dan residual plot
             
             2. Independensi
             - Setiap observasi harus independen satu sama lain
             - Penting terutama untuk data time series
             - Dapat diuji dengan Durbin-Watson test
             
             3. Homoskedastisitas
             - Varians error harus konstan untuk semua nilai X
             - Dapat diperiksa dengan residual plot
             - Pelanggaran dapat menyebabkan estimasi yang tidak efisien
             
             4. Normalitas
             - Residual harus berdistribusi normal
             - Dapat diuji dengan:
               * Q-Q plot
               * Shapiro-Wilk test
               * Kolmogorov-Smirnov test
             
             #### 4. Implikasi untuk Prediksi Ekspor
             
             1. Interpretasi Koefisien
             - β₁ positif: tren ekspor meningkat
             - β₁ negatif: tren ekspor menurun
             - Magnitude β₁: kecepatan perubahan
             
             2. Keterbatasan Model
             - Asumsi tren linear mungkin tidak selalu tepat
             - Tidak dapat menangkap perubahan musiman kompleks
             - Sensitif terhadap outlier
             
             3. Penggunaan R²
             - R² tinggi: prediksi lebih dapat diandalkan
             - R² rendah: perlu pertimbangan faktor lain
             - Berguna untuk membandingkan reliabilitas prediksi antar komoditas
             """)

if __name__ == "__main__":
    main()  

