import streamlit as st

from utils import ASSETS, note, page_header

page_header("📘 Metodologi & Catatan", "Bagaimana eksperimen dirancang, asumsi yang dipakai, dan keterbatasannya.")

st.markdown("### Skema peramalan (tanpa kebocoran data)")
st.markdown("""
- **Titik asal (origin)** adalah pukul **23:00** hari sebelumnya. Dari situ diramalkan 24 jam hari berikutnya (00:00 – 23:00).
- Fitur historis hanya memakai `nat_demand` sampai origin: lag ≥ 24 jam, nilai di origin, rata-rata 24 dan 168 jam terakhir.
- Kalender, libur, dan sekolah diketahui di muka. **Cuaca jam target memakai data aktual sebagai pengganti ramalan cuaca**, sehingga
  akurasi di dunia nyata (dengan ramalan NWP) akan sedikit lebih buruk.
- **Split kronologis 70 : 10 : 20** (tanpa diacak): Train ≤ 3 Nov 2018, Validasi 4 Nov 2018 – 22 Mei 2019 (early stopping),
  Test 23 Mei 2019 – 26 Jun 2020. Test memuat masa normal dan guncangan COVID-19 (sejak ± Maret 2020).
- **Normalisasi level:** target dan fitur lag dibagi rata-rata 168 jam terakhir di origin, sehingga model pohon tidak terganggu tren kenaikan beban.
- **Anomali:** 8 jam bernilai < 600 MWh (mis. pemadaman 20 Jan 2019) diinterpolasi untuk training dan fitur; evaluasi memakai data mentah.
""")

st.markdown("### Model dan alasan pemilihan")
c1, c2 = st.columns(2)
with c1:
    with st.expander("Baseline (Naive-24, Naive-168, rata-rata 7 hari)", expanded=True):
        st.write("Patokan minimum. Model yang lebih rumit wajib mengalahkannya agar kompleksitasnya layak.")
    with st.expander("Klasik: Dynamic Regression + ARMA(2,1)", expanded=True):
        st.write("Beban punya musiman ganda (24 dan 168 jam), sementara SARIMA hanya mendukung satu periode musiman. Solusinya regresi "
                 "dengan 168 dummy jam-dalam-minggu + suhu, libur, sekolah, dengan galat ARMA(2,1). Di-refit tiap 7 hari secara rolling.")
    with st.expander("Machine Learning: Ridge dan LightGBM", expanded=True):
        st.write("Ridge sebagai pembanding linear. LightGBM menangkap hubungan non-linear suhu–beban dan interaksi jam × hari × libur, "
                 "cepat, dan menyediakan importance. Satu model global untuk horizon 1–24 jam (direct multi-horizon).")
with c2:
    with st.expander("Deep Learning: GRU encoder + decoder kovariat", expanded=True):
        st.write("GRU membaca 168 jam terakhir, lalu decoder menggabungkannya dengan kovariat 24 jam ke depan dan memprediksi 24 nilai "
                 "sekaligus (residual terhadap beban minggu lalu).")
    with st.expander("Transfer learning", expanded=True):
        st.write("(a) Transfer antar-waktu: GRU di-pre-train pada data lama, lalu di-fine-tune (freeze encoder atau full) pada 365 hari terakhir train. "
                 "(b) Foundation model Chronos-Bolt, zero-shot dan sebagai fitur LightGBM, **disiapkan di notebook tetapi belum dijalankan "
                 "pada hasil yang ditampilkan di sini**.")
    with st.expander("Analisis variabel", expanded=True):
        st.write("Importance per kelompok variabel dengan 4 sudut pandang: gain, permutasi (model penuh), ablation (retrain), dan model "
                 "eksogen-saja tanpa lag, plus SHAP.")

st.markdown("### Catatan jujur tentang hasil")
note("""<ul style="margin:0;padding-left:18px">
<li><b>LightGBM unggul</b>, tetapi saat COVID keunggulannya menyempit dan model sederhana (rata-rata 7 hari, naive) bisa lebih baik pada RMSE.</li>
<li><b>Dynamic Regression + ARMA</b> bagus pada periode normal, namun memburuk drastis saat COVID. Regresi 365 hari dan pemodelan galat ARMA
sulit mengikuti perubahan level mendadak, dan terdapat peringatan konvergensi saat estimasi.</li>
<li><b>GRU overfitting</b>: error validasi terendah muncul di epoch 1 pada semua varian. Hasil DL ini belum mewakili potensi terbaiknya.</li>
<li><b>Transfer learning</b> pada GRU memberi perbaikan ±5–9%, tetapi tidak cukup mengejar LightGBM.</li>
<li><b>Chronos</b> belum menghasilkan angka karena pemanggilan API pada versi pustaka yang dipakai berbeda; sudah diperbaiki di notebook terbaru.</li>
<li>Satu seed untuk DL, periode COVID di Test hanya ± 3,5 bulan, dan cuaca memakai data aktual. Anggap kesimpulan bersifat <b>indikatif</b>.</li>
</ul>""", "warn")

st.markdown("### Gambar dari notebook")
imgs = [("eda.png", "Eksplorasi data"), ("importance.png", "Importance variabel"),
        ("perbandingan_model.png", "Perbandingan model"), ("forecast_week.png", "Ramalan vs aktual (satu minggu)"),
        ("mae_per_horizon.png", "MAE per horizon")]
for f, cap in imgs:
    p = ASSETS / f
    if p.exists():
        with st.expander(cap):
            st.image(str(p))
