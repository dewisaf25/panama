# ⚡ Panama Load Forecast Dashboard

Dashboard Streamlit untuk hasil eksperimen peramalan **beban listrik nasional Panama per jam, 24 jam ke depan**:
perbandingan model klasik, Machine Learning, Deep Learning, transfer learning, dan analisis variabel berpengaruh.

## Halaman
| Halaman | Isi |
|---|---|
| 🏠 Ringkasan | KPI, rentang split 70:10:20, temuan utama |
| 📊 Eksplorasi Data | tren, pola jam × hari, suhu, libur & sekolah, korelasi |
| 🏆 Perbandingan Model | peringkat per periode (total / pra-COVID / COVID), transfer learning, kurva overfitting GRU |
| 🔍 Variabel Berpengaruh | 4 metode importance + peringkat konsensus |
| 🔮 Forecast Explorer | ramalan vs aktual per tanggal, analisis galat, unduh CSV |
| 📘 Metodologi & Catatan | desain eksperimen, asumsi, keterbatasan |

## Struktur
```
app.py                 # router + sidebar
utils.py               # loader data, tema, helper plot
views/                 # satu file per halaman
data/                  # CSV hasil notebook (+ data per jam)
assets/                # gambar dari notebook
.streamlit/config.toml # tema
requirements.txt
```

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy dari GitHub (Streamlit Community Cloud)
1. Buat repository baru di GitHub (mis. `panama-load-forecast`), lalu **Add file → Upload files** dan unggah **seluruh isi folder ini**
   (termasuk folder `data/`, `assets/`, `views/`, dan `.streamlit/`). Pastikan `app.py` berada di root repo.
2. Buka <https://share.streamlit.io>, login dengan GitHub, klik **Create app → Deploy a public app from GitHub**.
3. Isi: **Repository** = repo Anda, **Branch** = `main`, **Main file path** = `app.py`. Di *Advanced settings* pilih Python 3.11 atau 3.12.
4. Klik **Deploy**. Setiap `git push` / upload baru ke GitHub akan memperbarui dashboard otomatis.

## Memperbarui data dari notebook Kaggle
Jalankan ulang notebook (versi terbaru berisi sel **“Export untuk dashboard”** di bagian akhir), lalu unduh dari panel *Output* dan timpa file di folder `data/`:

`dashboard_predictions.csv.gz`, `hourly_data.csv.gz`, `hasil_test.csv`, `hasil_test_pra_covid.csv`, `hasil_test_covid.csv`,
`importance_variabel.csv`, `training_log.csv`

> `dashboard_predictions.csv.gz` mengaktifkan semua model di halaman **Forecast Explorer**. Tanpa file itu, halaman tersebut hanya menampilkan baseline.
> Model baru (mis. Chronos) otomatis muncul di semua halaman karena nama dan kategorinya dibaca dari file hasil.

## Catatan
Cuaca jam target memakai data aktual sebagai pengganti ramalan cuaca, jadi akurasi riil akan sedikit lebih rendah. Detail ada di halaman Metodologi.
