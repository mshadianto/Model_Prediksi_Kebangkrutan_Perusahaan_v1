import streamlit as st
import pandas as pd
import joblib
import os
import yfinance as yf
import warnings
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
from datetime import datetime

# Suppress warnings from statsmodels and yfinance
# Setup config dan logging awal
st.set_page_config(page_title="Futuristic Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")
st.write("✅ App berhasil dijalankan! Memulai load model...")

# --- Konfigurasi Aplikasi ---
APP_VERSION = "1.0.7" # Versi Aplikasi (diperbarui)
AUTHOR_NAME = "MS Hadianto" # Nama Pembuat

# --- Definisi Penjelasan Fitur (untuk XAI) ---
FEATURE_EXPLANATIONS = {
    "Net income to Stockholder's Equity": "Mengukur berapa banyak laba bersih yang dihasilkan untuk setiap dolar ekuitas pemegang saham. Rasio tinggi menunjukkan efisiensi dalam menghasilkan keuntungan bagi pemilik.",
    "ROA(A) before interest and % after tax": "Return on Assets (ROA) sebelum bunga dan pajak. Menunjukkan seberapa efisien perusahaan menggunakan asetnya untuk menghasilkan keuntungan. Rasio yang lebih tinggi umumnya lebih baik.",
    "Attr2": "Rasio Total Kewajiban terhadap Total Aset. Mengukur tingkat utang perusahaan relatif terhadap asetnya. Rasio tinggi menunjukkan ketergantungan yang lebih besar pada utang.",
    "Attr4": "Rasio Lancar (Current Assets / Current Liabilities). Mengukur kemampuan perusahaan untuk memenuhi kewajiban jangka pendek. Rasio yang sehat biasanya di atas 1, menunjukkan perusahaan memiliki aset lancar yang cukup untuk membayar kewajiban lancarnya.",
    "Attr6": "Retained Earnings (Laba Ditahan) terhadap Total Aset. Menunjukkan proporsi laba yang diinvestasikan kembali dalam perusahaan dibandingkan dengan yang didistribusikan sebagai dividen. Ini mencerminkan kebijakan investasi perusahaan.",
    "Attr7": "EBIT (Earnings Before Interest and Taxes) terhadap Total Aset. Menunjukkan efisiensi operasional perusahaan dalam menghasilkan laba dari asetnya sebelum memperhitungkan biaya bunga dan pajak.",
    "Attr9": "Total Revenue terhadap Total Aset. Mengukur seberapa efisien perusahaan menggunakan asetnya untuk menghasilkan penjualan. Rasio yang lebih tinggi berarti penggunaan aset yang lebih baik.",
    "Attr15": "Total Kewajiban terhadap Total Ekuitas. Ini adalah rasio utang-ekuitas. Mengukur tingkat leverage keuangan perusahaan. Rasio tinggi menunjukkan risiko finansial yang lebih tinggi.",
    "Operating Profit Rate": "Mengukur persentase pendapatan yang tersisa setelah dikurangi biaya operasional. Menunjukkan efisiensi inti operasi bisnis, semakin tinggi semakin baik.",
    "Cash flow rate": "Arus Kas Operasi terhadap Total Aset. Menunjukkan kemampuan perusahaan menghasilkan kas dari operasi inti relatif terhadap asetnya.",
    "Attr45": "(Current Assets - Inventory) terhadap Current Liabilities. Ini adalah Quick Ratio (Rasio Cepat), yang lebih konservatif dari Rasio Lancar karena tidak memasukkan persediaan (inventory).",
    "Persistent EPS in the Last Four Seasons": "Earning Per Share (EPS) yang konsisten dalam empat periode terakhir. Menunjukkan stabilitas kinerja laba per saham.",
    "Net profit before tax/Paid-in capital": "Laba bersih sebelum pajak terhadap modal disetor. Mengukur efisiensi laba sebelum pajak dibandingkan dengan investasi modal awal.",
    "Net Value Per Share (A)": "Nilai bersih per saham (Total Ekuitas / Jumlah Saham Beredar). Menunjukkan nilai buku per saham yang dipegang investor."
}


# =====================================================================
# Fungsi-fungsi Backend
# =====================================================================
@st.cache_resource
def load_artifacts():
    """Memuat artefak model 'Highest Power'."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    try:
        model = joblib.load(os.path.join(models_dir, 'highest_power_model.joblib'))
        scaler = joblib.load(os.path.join(models_dir, 'highest_power_scaler.joblib'))
        feature_names = joblib.load(os.path.join(models_dir, 'highest_power_feature_names.joblib'))
        feature_importances = joblib.load(os.path.join(models_dir, 'highest_power_feature_importances.joblib'))
        return model, scaler, feature_names, feature_importances
    except FileNotFoundError:
        st.error("Kesalahan: File model tidak ditemukan. Pastikan Anda telah menjalankan `train_ultimate_model.py` (jika ada) dan file model ada di direktori `models`.")
        return None, None, None, None

def get_forecast(series: pd.Series, series_name: str = "Data", fill_method='interpolate'):
    """
    Membuat peramalan untuk 2 periode ke depan.
    Memastikan indeks adalah DatetimeIndex dan nilai numerik.
    Mengembalikan Series dengan indeks tahunan (misal: 2020, 2021 (F)).
    Menambahkan penanganan NaN dengan interpolate atau dropna.
    """
    series_original_index = series.index
    series = pd.to_numeric(series, errors='coerce')

    if series.isnull().all():
        st.warning(f"Peramalan untuk '{series_name}' gagal: Semua nilai adalah NaN setelah konversi numerik. Tidak ada data untuk peramalan.")
        return pd.Series(dtype='float64')

    series_cleaned = series.dropna()
    if len(series_cleaned) < 2:
        st.warning(f"Peramalan untuk '{series_name}' gagal: Hanya ada {len(series_cleaned)} titik data valid (minimal 2 dibutuhkan). Tidak cukup data untuk peramalan.")
        if not series.empty:
            series.index = series_original_index.map(lambda x: str(x.year) if isinstance(x, (datetime, pd.Timestamp)) else str(x))
        return pd.Series(dtype='float64') if series.empty else series.dropna()


    if fill_method == 'interpolate':
        series = series.interpolate(method='linear', limit_direction='both', limit_area=None)
    elif fill_method == 'dropna':
        series = series.dropna()
    else:
        series = series.fillna(series.mean() if not series.empty else 0)

    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index, format='%Y').to_period('A').to_timestamp('M')
        except Exception as e:
            st.warning(f"Gagal mengkonversi indeks '{series_name}' ke format tanggal. Error: {e}. Menggunakan data historis saja.")
            series.index = series_original_index.map(lambda x: str(x.year) if isinstance(x, (datetime, pd.Timestamp)) else str(x))
            return series.dropna()

    series = series.sort_index()

    try:
        model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method='estimated').fit()
        forecast = model.forecast(2)
        
        combined_series = pd.concat([series, forecast])
        formatted_index = combined_series.index.map(lambda x: f"{x.year} (F)" if x in forecast.index else str(x.year))
        combined_series.index = formatted_index
        
        return combined_series
    except Exception as e:
        st.warning(f"Peramalan Exponential Smoothing untuk '{series_name}' gagal: {e}. Menggunakan data historis saja.")
        series.index = series_original_index.map(lambda x: str(x.year) if isinstance(x, (datetime, pd.Timestamp)) else str(x))
        return series.dropna()

def safe_divide(n,d):
    n_safe = pd.to_numeric(n, errors='coerce')
    d_safe = pd.to_numeric(d, errors='coerce')

    n_safe = np.nan_to_num(n_safe, nan=0.0)
    d_safe = np.nan_to_num(d_safe, nan=0.0)

    if d_safe == 0:
        return 0.0
    return n_safe / d_safe

# --- PEMBARUAN DI SINI: calculate_features_historical agar selalu menghitung semua fitur ---
def calculate_features_historical(fin, bs, cf, feature_names_for_model):
    """
    Menghitung fitur untuk semua tahun yang tersedia.
    Mengembalikan DataFrame dengan tahun sebagai indeks dan fitur sebagai kolom.
    Fungsi ini akan menghitung SEMUA fitur yang didefinisikan secara internal,
    kemudian memastikan hanya fitur yang diharapkan model yang ada di output utama.
    """
    all_dates = sorted(list(set(fin.columns) | set(bs.columns) | set(cf.columns)), reverse=False)

    features_data = []
    
    for date_ts in all_dates:
        fin_y = fin.get(date_ts, pd.Series(dtype='float64'))
        bs_y = bs.get(date_ts, pd.Series(dtype='float64'))
        cf_y = cf.get(date_ts, pd.Series(dtype='float64'))
        
        if fin_y.empty and bs_y.empty and cf_y.empty:
            continue
            
        current_year_features_full = {} # Fitur lengkap yang dihitung

        ni = pd.to_numeric(fin_y.get('Net Income', np.nan), errors='coerce')
        ta = pd.to_numeric(bs_y.get('Total Assets', np.nan), errors='coerce')
        tl = pd.to_numeric(bs_y.get('Total Liabilities Net Minority Interest', np.nan), errors='coerce')
        ca = pd.to_numeric(bs_y.get('Current Assets', np.nan), errors='coerce')
        cl = pd.to_numeric(bs_y.get('Current Liabilities', np.nan), errors='coerce')
        te = pd.to_numeric(bs_y.get('Total Equity Gross Minority Interest', np.nan), errors='coerce')
        re = pd.to_numeric(bs_y.get('Retained Earnings', np.nan), errors='coerce')
        ebit = pd.to_numeric(fin_y.get('EBIT', np.nan), errors='coerce')
        tr = pd.to_numeric(fin_y.get('Total Revenue', np.nan), errors='coerce')
        inv = pd.to_numeric(bs_y.get('Inventory', np.nan), errors='coerce')
        cse = pd.to_numeric(bs_y.get('Common Stock Equity', te if pd.notna(te) else np.nan), errors='coerce')
        pi = pd.to_numeric(fin_y.get('Pretax Income', np.nan), errors='coerce')
        eps = pd.to_numeric(fin_y.get('Basic EPS', np.nan), errors='coerce')
        
        ocf_keys = ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities', 'Total Cash Flow From Operating Activities']
        ocf_val = next((fin_y.get(key, np.nan) for key in ocf_keys if key in fin_y), np.nan)
        ocf = pd.to_numeric(ocf_val, errors='coerce')

        # Hitung semua fitur yang mungkin dan simpan dalam current_year_features_full
        current_year_features_full["Net income to Stockholder's Equity"] = safe_divide(ni, te)
        current_year_features_full['ROA(A) before interest and % after tax'] = safe_divide(ni, ta)
        current_year_features_full['Attr2'] = safe_divide(tl, ta)
        current_year_features_full['Attr4'] = safe_divide(ca, cl) # Ini akan selalu dihitung
        current_year_features_full['Attr6'] = safe_divide(re, ta)
        current_year_features_full['Attr7'] = safe_divide(ebit, ta)
        current_year_features_full['Attr9'] = safe_divide(tr, ta)
        current_year_features_full['Attr15'] = safe_divide(tl, te)
        current_year_features_full['Operating Profit Rate'] = safe_divide(pd.to_numeric(fin_y.get('Operating Income', np.nan), errors='coerce'), tr)
        current_year_features_full['Cash flow rate'] = safe_divide(ocf, ta)
        current_year_features_full['Attr45'] = safe_divide((ca - inv), cl)
        current_year_features_full['Persistent EPS in the Last Four Seasons'] = eps
        current_year_features_full['Net profit before tax/Paid-in capital'] = safe_divide(pi, cse)
        current_year_features_full['Net Value Per Share (A)'] = safe_divide(te, pd.to_numeric(fin_y.get('Share Issued', 1), errors='coerce'))

        # Buat row_data yang HANYA berisi fitur yang diharapkan oleh model
        # Ini adalah input yang akan diberikan ke model untuk prediksi
        row_data_for_model = {feature: np.nan_to_num(current_year_features_full.get(feature, np.nan)) for feature in feature_names_for_model}
        
        # Simpan semua fitur yang dihitung (current_year_features_full) untuk historical_calculated_features_df
        # Ini akan memastikan 'Attr4' selalu ada untuk plotting
        features_data.append({'Tahun': date_ts, **{k: np.nan_to_num(v) for k,v in current_year_features_full.items()}})


    # DataFrame ini akan memiliki SEMUA fitur yang dihitung (termasuk Attr4)
    features_df = pd.DataFrame(features_data).set_index('Tahun')
    features_df = features_df.sort_index()

    # Namun, untuk PREDIKSI model, kita akan menggunakan sub-set fitur yang sesuai
    # dengan feature_names_for_model (yang dimuat dari joblib)
    # Ini akan dilakukan di get_prediction_and_trends
    return features_df

def generate_professional_narrative(result):
    info=result['info'];prediction=result['prediction'];probability=result['probability'];top_factors=result['top_factors'];historical_data=result['historical_data'];prob_percent=probability*100;top_factor_1=top_factors.iloc[0]['Fitur']
    
    top_factor_1_explanation = FEATURE_EXPLANATIONS.get(top_factor_1, "faktor kunci yang berpengaruh")

    if prediction==1:
        p1=f"Berdasarkan analisis model *ensemble* kami, **{info.get('longName','')}** menunjukkan **profil risiko kebangkrutan yang tinggi** dengan probabilitas terkuantifikasi sebesar **{prob_percent:.1f}%**. Skor ini mengindikasikan adanya sinyal-sinyal tekanan finansial yang signifikan, di mana faktor seperti **'{top_factor_1}'** ({top_factor_1_explanation}) teridentifikasi sebagai variabel dengan pengaruh paling dominan terhadap penilaian risiko ini."
    else:
        p1=f"Berdasarkan analisis model *ensemble* kami, **{info.get('longName','')}** dari sektor *{info.get('sector','N/A')}*, menunjukkan profil risiko kebangkrutan yang **rendah (STABIL)**. Model memberikan skor kuantitatif probabilitas risiko sebesar **{prob_percent:.1f}%**, yang mencerminkan kesehatan fundamental yang solid. Penilaian ini, meskipun positif, tetap mempertimbangkan secara cermat faktor-faktor kunci seperti **'{top_factor_1}'** ({top_factor_1_explanation}) yang teridentifikasi memiliki pengaruh signifikan dalam kalkulasi risiko model kami."
    
    revenue_trend_series=historical_data.get("Pendapatan (Revenue)",pd.Series(dtype='float64'))
    net_income_trend_series=historical_data.get("Laba Bersih (Net Income)",pd.Series(dtype='float64'))
    
    def safe_extract_and_convert(series_data: pd.Series, index_pos: int):
        if series_data is None or series_data.empty:
            return np.nan
        try:
            val = pd.to_numeric(series_data, errors='coerce').iloc[index_pos]
            if pd.notna(val):
                return val
        except (IndexError, TypeError, ValueError):
            pass
        return np.nan

    last_revenue = safe_extract_and_convert(revenue_trend_series, -1)
    first_revenue = safe_extract_and_convert(revenue_trend_series, 0)
    
    revenue_growth="konsisten bertumbuh" if pd.notna(last_revenue) and pd.notna(first_revenue) and last_revenue > first_revenue else "mengalami stagnasi atau penurunan" if pd.notna(last_revenue) and pd.notna(first_revenue) and last_revenue < first_revenue else "cenderung stabil"

    last_net_income = safe_extract_and_convert(net_income_trend_series, -1)
    first_net_income = safe_extract_and_convert(net_income_trend_series, 0)
    net_income_growth="turut menunjukkan pertumbuhan" if pd.notna(last_net_income) and pd.notna(first_net_income) and last_net_income > first_net_income else "menunjukkan adanya tekanan" if pd.notna(last_net_income) and pd.notna(first_net_income) and last_net_income < first_net_income else "cenderung stabil"
    
    p2=f"Penilaian stabilitas ini didukung oleh data historis yang menunjukkan **pendapatan perusahaan {revenue_growth}**. Di sisi lain, fakta bahwa laba bersihnya **{net_income_growth}** menjelaskan mengapa skor risiko tidak mencapai nol. Secara keseluruhan, fundamental perusahaan dinilai kuat, dengan kemampuan menjaga pendapatan sebagai bantalan utama terhadap potensi risiko.";return f"{p1}\n\n{p2}"

def get_prediction_and_trends(ticker_str, model, scaler, feature_names, feature_importances):
    try:
        ticker=yf.Ticker(ticker_str)
        info=ticker.info
        
        financials=ticker.financials 
        balance_sheet=ticker.balance_sheet
        cash_flow=ticker.cashflow
        
        if financials.empty or balance_sheet.empty or cash_flow.empty: 
            return {'error': f"Data keuangan (L/R, Neraca, atau Arus Kas) tidak lengkap untuk {ticker_str}. Pastikan data tersedia di Yahoo Finance."}
        
        raw_yfinance_data = {
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow
        }

        # --- PEMBARUAN DI SINI: calculate_features_historical mengembalikan semua fitur yang mungkin ---
        # DataFrame ini akan memiliki semua fitur yang dihitung, termasuk Attr4
        historical_calculated_features_df = calculate_features_historical(financials, balance_sheet, cash_flow, feature_names)
        
        if historical_calculated_features_df.empty:
            return {'error': f"Gagal menghitung fitur historis untuk {ticker_str}. Data mungkin tidak cukup atau tidak valid. Cek apakah ada data yang tersedia di Yahoo Finance."}

        calculated_features_latest = historical_calculated_features_df.iloc[-1] 
        
        # --- PEMBARUAN DI SINI: Filter fitur hanya yang ada di feature_names model untuk prediksi ---
        prediction_df_for_model = pd.DataFrame(0, index=[0], columns=feature_names) # Gunakan feature_names langsung
        for feature in feature_names: # Hanya iterasi fitur yang diharapkan model
            value = calculated_features_latest.get(feature, np.nan) # Ambil nilai dari fitur yang dihitung
            prediction_df_for_model[feature] = np.nan_to_num(pd.to_numeric(value, errors='coerce')) # Pastikan numerik & isi NaN/Inf dengan 0

        input_scaled=scaler.transform(prediction_df_for_model); # Gunakan prediction_df_for_model
        prediction=model.predict(input_scaled)[0];
        probability=model.predict_proba(input_scaled)[0][1]
        
        importance_df=pd.DataFrame({'Fitur':feature_names,'Pentingnya':feature_importances}).sort_values(by='Pentingnya',ascending=False).head(5)
        
        historical_data={}
        if 'Total Revenue' in financials.index: historical_data["Pendapatan (Revenue)"]=get_forecast(financials.loc['Total Revenue'], "Pendapatan (Revenue)")
        if 'Net Income' in financials.index: historical_data["Laba Bersih (Net Income)"]=get_forecast(financials.loc['Net Income'], "Laba Bersih (Net Income)")
        
        # Rasio-rasio ini sekarang selalu ada di historical_calculated_features_df (jika datanya ada)
        if 'ROA(A) before interest and % after tax' in historical_calculated_features_df.columns:
            historical_data["ROA"] = get_forecast(historical_calculated_features_df['ROA(A) before interest and % after tax'], "ROA")
        else: # Hanya untuk jaga-jaga, seharusnya ada jika data input tersedia
            historical_data["ROA"] = pd.Series(dtype='float64') 
        
        if 'Attr4' in historical_calculated_features_df.columns: 
             historical_data["Rasio Lancar"] = get_forecast(historical_calculated_features_df['Attr4'], "Rasio Lancar")
        else: # Hanya untuk jaga-jaga
            historical_data["Rasio Lancar"] = pd.Series(dtype='float64')
        
        if 'Operating Profit Rate' in historical_calculated_features_df.columns:
            historical_data["Operating Profit Rate"] = get_forecast(historical_calculated_features_df['Operating Profit Rate'], "Operating Profit Rate")
        else: # Hanya untuk jaga-jaga
            historical_data["Operating Profit Rate"] = pd.Series(dtype='float64')


        result={
            'error':None,
            'prediction':prediction,
            'probability':probability,
            'top_factors':importance_df,
            'info':info,
            'historical_data':historical_data,
            'historical_calculated_features': historical_calculated_features_df, # Ini punya Attr4 sekarang
            'raw_yfinance_data': raw_yfinance_data
        }
        result['narrative']=generate_professional_narrative(result)
        return result
    except Exception as e:
        return {'error': f"Terjadi error saat analisis: {e}. Pastikan ticker benar dan data keuangan tersedia."}

# =====================================================================
# Antarmuka Pengguna (UI)
# =====================================================================
st.title("🌟 Futuristic Analytics Dashboard")
st.markdown("Menganalisis Risiko Kebangkrutan dengan *Optimized Ensemble Model*, *Explainable AI*, dan *Time Series Forecasting*.")

model, scaler, feature_names, feature_importances = load_artifacts()
if model is None:
    st.error("File model 'Highest Power' tidak ditemukan! Silakan cek kembali file artefak model.")
    st.stop()

# --- PEMBARUAN DI SINI: Info Model di sidebar yang diperbarui ---
st.sidebar.markdown("---")
st.sidebar.subheader("Info Model (Debugging)")
if feature_names is not None:
    st.sidebar.write("**Fitur yang diharapkan oleh model:**")
    st.sidebar.code(str(feature_names))
    if 'Attr4' in feature_names:
        st.sidebar.success("✅ 'Attr4' ditemukan di fitur model. Model Anda dilatih dengan fitur ini.")
    else:
        st.sidebar.error("❌ 'Attr4' TIDAK ditemukan di fitur model. **Model Anda tidak menggunakan Rasio Lancar untuk prediksi.** Meskipun grafik akan ditampilkan, prediksi model tidak mempertimbangkan fitur ini.")
else:
    st.sidebar.info("Fitur model belum dimuat.")
st.sidebar.markdown("---")


st.sidebar.header("⚙️ Panel Analisis")
ticker_input = st.sidebar.text_input("Masukkan Ticker Saham (.JK)", "ASII.JK").upper()
analyze_button = st.sidebar.button("Analisis Sekarang!", type="primary", use_container_width=True)

if analyze_button:
    if not ticker_input:
        st.warning("Mohon masukkan ticker saham.", icon="⚠️")
    else:
        with st.spinner(f"Menganalisis **{ticker_input}**..."):
            result = get_prediction_and_trends(ticker_input, model, scaler, feature_names, feature_importances)

        st.header(f"Hasil Analisis untuk {ticker_input}", divider='rainbow')
        
        if result.get('error'):
            st.error(result['error'], icon="💔")
        else:
            info = result['info']
            st.subheader(f"{info.get('longName', ticker_input)}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Prediksi Risiko")
                if result['prediction'] == 1:
                    st.error("🔴 **Berisiko Tinggi**")
                else:
                    st.success("🟢 **Berisiko Rendah**")
            
            with col2:
                st.metric("Probabilitas Bangkrut", f"{result['probability']:.2%}")

            with col3:
                st.metric("Sektor Industri", info.get('sector', 'N/A'))

            st.subheader("📝 Ringkasan Analitik Profesional")
            st.info(result['narrative'])

            st.subheader("🔍 Detail Analisis")
            
            historical_data = result.get('historical_data', {}) 
            historical_calculated_features = result.get('historical_calculated_features', pd.DataFrame())
            raw_yfinance_data = result.get('raw_yfinance_data', {})

            if not historical_data and historical_calculated_features.empty:
                st.warning("Tidak ada data historis yang cukup untuk menampilkan grafik tren dan perhitungan fitur.")
            else:
                # --- Bagian Grafik ---
                st.markdown("---")
                st.markdown("### Proyeksi Tren Kinerja Keuangan (Historis + Peramalan 2 Tahun)")
                hist_cols_main = st.columns(2)
                
                with hist_cols_main[0]:
                    st.markdown("###### Pendapatan (Revenue)")
                    if "Pendapatan (Revenue)" in historical_data and not historical_data["Pendapatan (Revenue)"].empty:
                        revenue_df = historical_data['Pendapatan (Revenue)'].reset_index()
                        revenue_df.columns = ['Tahun', 'Nilai']
                        revenue_df['Jenis'] = np.where(revenue_df['Tahun'].astype(str).str.contains("\(F\)"), 'Peramalan', 'Historis')
                        revenue_df['Tahun_Numerik'] = revenue_df['Tahun'].astype(str).str.replace(' \(F\)', '', regex=True).astype(int)
                        
                        fig_revenue = px.bar(revenue_df, x='Tahun_Numerik', y='Nilai', color='Jenis',
                                            color_discrete_map={'Historis': '#20C20E', 'Peramalan': '#98FB98'},
                                            title='Tren Pendapatan Tahunan')
                        fig_revenue.update_layout(xaxis_title="Tahun", yaxis_title="Nilai Pendapatan", showlegend=True)
                        st.plotly_chart(fig_revenue, use_container_width=True)
                    else:
                        st.caption("Data pendapatan tidak tersedia.")
                
                with hist_cols_main[1]:
                    st.markdown("###### Laba Bersih (Net Income)")
                    if "Laba Bersih (Net Income)" in historical_data and not historical_data["Laba Bersih (Net Income)"].empty:
                        net_income_df = historical_data['Laba Bersih (Net Income)'].reset_index()
                        net_income_df.columns = ['Tahun', 'Nilai']
                        net_income_df['Jenis'] = np.where(net_income_df['Tahun'].astype(str).str.contains("\(F\)"), 'Peramalan', 'Historis')
                        net_income_df['Tahun_Numerik'] = net_income_df['Tahun'].astype(str).str.replace(' \(F\)', '', regex=True).astype(int)

                        fig_net_income = px.bar(net_income_df, x='Tahun_Numerik', y='Nilai', color='Jenis',
                                                color_discrete_map={'Historis': '#0E20C2', 'Peramalan': '#98C2FB'},
                                                title='Tren Laba Bersih Tahunan')
                        fig_net_income.update_layout(xaxis_title="Tahun", yaxis_title="Nilai Laba Bersih", showlegend=True)
                        st.plotly_chart(fig_net_income, use_container_width=True)
                    else:
                        st.caption("Data laba bersih tidak tersedia.")
                
                # --- Grafik 3: Faktor Kunci (Feature Importances) ---
                st.markdown("---")
                st.markdown("### Faktor-faktor yang Paling Berpengaruh (Explainable AI)")
                if not result['top_factors'].empty and not historical_calculated_features.empty:
                    top_factors_with_values = result['top_factors'].copy()
                    
                    latest_features_series = historical_calculated_features.iloc[-1]
                    top_factors_with_values['Nilai Aktual (Tahun Terakhir)'] = top_factors_with_values['Fitur'].map(lambda f: f"{latest_features_series.get(f, np.nan):.4f}" if pd.notna(latest_features_series.get(f, np.nan)) else "N/A")
                    top_factors_with_values['Penjelasan'] = top_factors_with_values['Fitur'].map(FEATURE_EXPLANATIONS)
                    
                    fig_top_factors = px.bar(top_factors_with_values, x='Pentingnya', y='Fitur', orientation='h',
                                            title='5 Faktor Teratas yang Mempengaruhi Prediksi',
                                            labels={'Pentingnya': 'Tingkat Kepentingan Relatif', 'Fitur': 'Faktor Analisis'},
                                            color='Pentingnya', color_continuous_scale=px.colors.sequential.Greens,
                                            hover_data=['Nilai Aktual (Tahun Terakhir)', 'Penjelasan'])
                    fig_top_factors.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_top_factors, use_container_width=True)
                else:
                    st.caption("Data faktor kunci tidak tersedia atau tidak ada fitur historis yang dihitung.")

                # --- Grafik 4 & 5 (Contoh): Proyeksi Tren Rasio Keuangan Penting ---
                st.markdown("---")
                st.markdown("### Proyeksi Tren Rasio Keuangan")
                
                financial_ratio_cols_bottom = st.columns(2)
                
                with financial_ratio_cols_bottom[0]:
                    st.markdown("###### ROA (Return on Assets)")
                    if "ROA" in historical_data and not historical_data["ROA"].empty:
                        roa_df = historical_data['ROA'].reset_index()
                        roa_df.columns = ['Tahun', 'Nilai']
                        roa_df['Jenis'] = np.where(roa_df['Tahun'].astype(str).str.contains("\(F\)"), 'Peramalan', 'Historis')
                        roa_df['Tahun_Numerik'] = roa_df['Tahun'].astype(str).str.replace(' \(F\)', '', regex=True).astype(int)

                        fig_roa = px.line(roa_df, x='Tahun_Numerik', y='Nilai', color='Jenis', markers=True,
                                        color_discrete_map={'Historis': 'blue', 'Peramalan': 'red'},
                                        title='Tren Rasio ROA Tahunan')
                        fig_roa.update_layout(xaxis_title="Tahun", yaxis_title="ROA (%)", showlegend=True)
                        st.plotly_chart(fig_roa, use_container_width=True)
                    else:
                        st.caption("Data ROA tidak tersedia atau tidak lengkap untuk analisis tren. Pastikan data keuangan lengkap.")
                
                with financial_ratio_cols_bottom[1]:
                    st.markdown("###### Rasio Lancar (Current Ratio)")
                    # --- PEMBARUAN DI SINI: Menampilkan grafik jika historical_data["Rasio Lancar"] tidak kosong,
                    # meskipun Attr4 tidak diakui model untuk prediksi
                    if "Rasio Lancar" in historical_data and not historical_data["Rasio Lancar"].empty:
                        cr_df = historical_data['Rasio Lancar'].reset_index()
                        cr_df.columns = ['Tahun', 'Nilai']
                        cr_df['Jenis'] = np.where(cr_df['Tahun'].astype(str).str.contains("\(F\)"), 'Peramalan', 'Historis')
                        cr_df['Tahun_Numerik'] = cr_df['Tahun'].astype(str).str.replace(' \(F\)', '', regex=True).astype(int)

                        fig_cr = px.line(cr_df, x='Tahun_Numerik', y='Nilai', color='Jenis', markers=True,
                                         color_discrete_map={'Historis': 'purple', 'Peramalan': 'orange'},
                                         title='Tren Rasio Lancar Tahunan')
                        fig_cr.update_layout(xaxis_title="Tahun", yaxis_title="Rasio Lancar", showlegend=True)
                        st.plotly_chart(fig_cr, use_container_width=True)
                    else:
                        st.caption("Data Rasio Lancar tidak tersedia atau tidak lengkap untuk analisis tren. Pastikan data keuangan lengkap.")
                
                # --- Grafik 6 (tambahan): Operating Profit Rate ---
                st.markdown("---")
                st.markdown("### Tren Rasio Profitabilitas")
                profit_ratio_cols = st.columns(1)

                with profit_ratio_cols[0]:
                    st.markdown("###### Operating Profit Rate")
                    if "Operating Profit Rate" in historical_data and not historical_data["Operating Profit Rate"].empty:
                        opr_df = historical_data['Operating Profit Rate'].reset_index()
                        opr_df.columns = ['Tahun', 'Nilai']
                        opr_df['Jenis'] = np.where(opr_df['Tahun'].astype(str).str.contains("\(F\)"), 'Peramalan', 'Historis')
                        opr_df['Tahun_Numerik'] = opr_df['Tahun'].astype(str).str.replace(' \(F\)', '', regex=True).astype(int)

                        fig_opr = px.line(opr_df, x='Tahun_Numerik', y='Nilai', color='Jenis', markers=True,
                                         color_discrete_map={'Historis': 'darkgreen', 'Peramalan': 'lightgreen'},
                                         title='Tren Operating Profit Rate Tahunan')
                        fig_opr.update_layout(xaxis_title="Tahun", yaxis_title="Operating Profit Rate (%)", showlegend=True)
                        st.plotly_chart(fig_opr, use_container_width=True)
                    else:
                        st.caption("Data Operating Profit Rate tidak tersedia atau tidak lengkap untuk analisis tren. Pastikan data keuangan lengkap.")

                # --- Bagian Debugging & Inspeksi Data Mentah ---
                st.markdown("---")
                with st.expander("🔬 **Debugging & Inspeksi Data Mentah**"):
                    st.markdown("###### Data Keuangan Mentah dari Yahoo Finance (4 Periode Terakhir)")
                    if raw_yfinance_data:
                        st.write("**Financials (Laporan Laba Rugi):**")
                        st.dataframe(raw_yfinance_data['financials'])
                        st.write("**Balance Sheet (Neraca):**")
                        st.dataframe(raw_yfinance_data['balance_sheet'])
                        st.write("**Cash Flow (Arus Kas):**")
                        st.dataframe(raw_yfinance_data['cash_flow'])
                    else:
                        st.info("Data mentah Yahoo Finance tidak tersedia.")

                    st.markdown("###### Fitur Historis yang Dihitung untuk Model (per tahun)")
                    if not historical_calculated_features.empty:
                        # --- PEMBARUAN DI SINI: Tampilkan semua fitur yang dihitung oleh calculate_features_historical
                        # Termasuk yang mungkin tidak dipakai model
                        st.dataframe(historical_calculated_features) 
                        
                        st.markdown("---")
                        st.markdown("###### Analisis Debugging Rasio Lancar (Attr4)")
                        bs_df = raw_yfinance_data.get('balance_sheet', pd.DataFrame())
                        
                        if 'Current Assets' in bs_df.index:
                            st.write("**Current Assets dari Neraca:**")
                            st.dataframe(bs_df.loc[['Current Assets']].T)
                        else:
                            st.error("❌ `Current Assets` **TIDAK DITEMUKAN** di Neraca dari Yahoo Finance.")
                        
                        if 'Current Liabilities' in bs_df.index:
                            st.write("**Current Liabilities dari Neraca:**")
                            st.dataframe(bs_df.loc[['Current Liabilities']].T)
                        else:
                            st.error("❌ `Current Liabilities` **TIDAK DITEMUKAN** di Neraca dari Yahoo Finance.")

                        # --- PEMBARUAN DI SINI: Tampilkan Attr4 dari historical_calculated_features
                        # Karena sekarang historical_calculated_features akan selalu memiliki Attr4 (jika dihitung)
                        if 'Attr4' in historical_calculated_features.columns:
                            st.write("**Nilai Attr4 (Rasio Lancar) setelah perhitungan fitur (untuk plotting):**")
                            attr4_series = historical_calculated_features['Attr4'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN").rename("Rasio Lancar (Attr4)")
                            st.dataframe(attr4_series.to_frame())
                            if attr4_series.isnull().all() or (attr4_series == 0).all():
                                st.warning("⚠️ Semua nilai Rasio Lancar (Attr4) adalah NaN atau nol. Ini mungkin mengapa grafik tidak ditampilkan.")
                        else:
                            # Pesan ini seharusnya tidak muncul lagi jika Attr4 selalu dihitung
                            st.error("❌ Kolom 'Attr4' (Rasio Lancar) **TIDAK DITEMUKAN** di fitur historis yang dihitung. Ini mengindikasikan kegagalan perhitungan atau nama fitur yang salah.")
                    else:
                        st.info("Fitur historis yang dihitung tidak tersedia.")
else:
    st.info("Masukkan ticker saham di panel sebelah kiri untuk memulai analisis.")

# --- Bagian Footer ---
st.markdown("---")
st.markdown("""
<div style="font-size: 0.8em; color: grey;">
<b>Disclaimer:</b> Informasi dan analisis yang disajikan dalam dashboard ini murni untuk tujuan edukasi dan informatif. Ini bukan merupakan saran investasi atau rekomendasi finansial. Model AI dan peramalan waktu tunduk pada ketidakpastian dan tidak menjamin kinerja masa depan. Keputusan investasi harus didasarkan pada riset independen, penilaian profesional, dan konsultasi dengan penasihat keuangan berlisensi. Pengembang tidak bertanggung jawab atas kerugian atau kerusakan yang timbul dari penggunaan informasi ini.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="font-size: 0.75em; color: #888888; text-align: center; margin-top: 1em;">
Versi Aplikasi: <b>{APP_VERSION}</b> | Dibuat oleh: <b>{AUTHOR_NAME}</b>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='font-size: 0.7em; color: #999999; text-align: center; margin-top: 0.5em;'>Dibuat dengan Streamlit, Plotly Express, dan Model AI</div>", unsafe_allow_html=True)
