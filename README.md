# 📉 Model Prediksi Kebangkrutan Perusahaan

**Dashboard Streamlit berbasis AI untuk memprediksi risiko kebangkrutan perusahaan menggunakan data keuangan dari Yahoo Finance.**

![Dashboard Screenshot](https://user-images.githubusercontent.com/your-image-placeholder.png) <!-- (Opsional jika ada preview UI) -->

---

## 🚀 Fitur Utama

- ✅ Prediksi risiko kebangkrutan berbasis model ensemble machine learning
- 📊 Visualisasi tren keuangan: pendapatan, laba bersih, ROA, rasio lancar
- 🧠 Explainable AI (XAI): faktor dominan dalam penilaian risiko
- 🔮 Peramalan 2 tahun ke depan dengan Exponential Smoothing
- 💡 Narasi otomatis profesional (analytical narrative generator)

---

## 📁 Struktur Folder

futuristic_dashboard/
├── dashboard_final.py # Main Streamlit app
├── logo MSH.png # Logo organisasi
├── requirements.txt # Daftar library Python
└── models/ # Folder model joblib
├── highest_power_model.joblib
├── highest_power_scaler.joblib
├── highest_power_feature_names.joblib
└── highest_power_feature_importances.joblib

---

## ⚙️ Cara Menjalankan Lokal

1. **Clone repo ini**
```bash
git clone https://github.com/mshadianto/model_prediksi_kebangkrutan_perusahaan_v1.git
cd model_prediksi_kebangkrutan_perusahaan_v1/futuristic_dashboard

2. Install dependencies
pip install -r requirements.txt

3. Jalankan aplikasi
streamlit run dashboard_final.py

4. 🌐 Deployment Online
Aplikasi ini tersedia online via Streamlit Cloud

5. 🧠 Model Machine Learning
> Dataset: Diambil dari Yahoo Finance via yfinance
> Fitur: 14+ rasio keuangan penting
> Model: Ensemble Classifier (voting, scaling, joblib)
> XAI: Interpretasi pentingnya fitur berdasarkan magnitude SHAP-like weights
## 📚 Dataset Pelatihan Model

6. Model dilatih menggunakan dua dataset publik:

a. **Taiwan Bankruptcy Dataset**  
   - Sumber: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Taiwan+Bankruptcy+Prediction)
   - Jumlah record: 6.000
   - Fitur: 96 indikator keuangan selama 5 tahun

b. **Polish Companies Bankruptcy Data**  
   - Sumber: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data)
   - Jumlah record: ~10.000+
   - Tahun: 2007–2013

👨‍💻 Author
MS Hadianto
AI Governance Enthusiast
LinkedIn | Email: sopian.hadianto@gmail.com

📜 Lisensi
MIT License – bebas digunakan, mohon beri atribusi.
