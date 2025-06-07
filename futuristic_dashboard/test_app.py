import streamlit as st
import os

# Konfigurasi tampilan dasar
st.set_page_config(page_title="Test App Streamlit", layout="wide")

# Log pembuka
st.write("✅ App berhasil dijalankan!")

# Tes akses folder kerja & isi folder models
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'models')

st.write(f"📁 Path saat ini: {script_dir}")
if os.path.exists(models_dir):
    st.success("📦 Folder 'models' ditemukan.")
    st.write("🗂 Isi folder models:", os.listdir(models_dir))
else:
    st.error("❌ Folder 'models' TIDAK ditemukan.")

# Tes file model
file_list = [
    "highest_power_model.joblib",
    "highest_power_scaler.joblib",
    "highest_power_feature_names.joblib",
    "highest_power_feature_importances.joblib"
]

for fname in file_list:
    full_path = os.path.join(models_dir, fname)
    if os.path.isfile(full_path):
        st.write(f"✅ File ditemukan: {fname}")
    else:
        st.error(f"❌ File TIDAK ditemukan: {fname}")
