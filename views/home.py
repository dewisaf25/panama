import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (SPLITS, COVID_START, TL_MODELS, card, cat_color, hero, kpi, load_hourly, load_importance,
                   load_results, plot, shade, style_fig, vline, note)

res = load_results()
te, pre, cov = res["Test total"], res["Pra-COVID"], res["Saat COVID"]
imp = load_importance()
h = load_hourly()

cand = te[~te["kategori"].str.contains("ablation")]
best_name = cand["MAE"].idxmin()
best = te.loc[best_name]
naive = te.loc["Naive-168 (minggu lalu)"]
gain_vs_naive = 100 * (1 - best["MAE"] / naive["MAE"])

hero("⚡ Peramalan Beban Listrik Nasional Panama",
     "Dashboard hasil eksperimen peramalan beban per jam <b>24 jam ke depan</b> (day-ahead): membandingkan model "
     "<b>time series klasik, Machine Learning, dan Deep Learning</b>, termasuk <b>transfer learning</b>, "
     "serta mengidentifikasi variabel yang paling memengaruhi beban.",
     tags=["48.048 jam data", "Jan 2015 – Jun 2020", "Split kronologis 70 : 10 : 20", "12 model dibandingkan"])

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Model terbaik (MAE)", best_name.replace(" (ML)", ""), f"kategori: {best['kategori']}", cat_color(best["kategori"]))
with c2:
    kpi("MAE Test", f"{best['MAE']:.1f} MWh", f"RMSE {best['RMSE']:.1f} · MAPE {best['MAPE%']:.2f}%", "#4F46E5")
with c3:
    kpi("Lebih baik dari Naive-168", f"{gain_vs_naive:.0f}%", f"MAE {naive['MAE']:.1f} → {best['MAE']:.1f} MWh", "#10B981")
with c4:
    kpi("R² (Test)", f"{best['R2']:.3f}", f"pra-COVID {pre.loc[best_name,'R2']:.3f} · COVID {cov.loc[best_name,'R2']:.3f}", "#EC4899")

st.markdown("&nbsp;")
st.markdown("### Rentang data & pembagian Train / Validasi / Test")
daily = h.set_index("datetime")["nat_clean"].resample("D").mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily.index, y=daily.values, mode="lines", line=dict(color="#1E1B4B", width=1.1),
                         name="Rata-rata harian", hovertemplate="%{x|%d %b %Y}<br>%{y:.0f} MWh<extra></extra>"))
for name, (a, b, col) in SPLITS.items():
    shade(fig, a, pd.Timestamp(b) + pd.Timedelta(days=1), col, name)
vline(fig, COVID_START, "#EF4444", "Awal COVID-19")
fig.update_yaxes(title="MWh")
plot(style_fig(fig, height=360, legend_bottom=False))

st.markdown("### Temuan utama")
tl_scratch, tl_full = te.loc[TL_MODELS["scratch_1y"], "MAE"], te.loc[TL_MODELS["tf_full"], "MAE"]
tl_gain = 100 * (1 - tl_full / tl_scratch)
cov_best_name = cov[~cov["kategori"].str.contains("ablation")]["MAE"].idxmin()
dyn_pre, dyn_cov = pre.loc["Dynamic Regression + ARMA", "MAE"], cov.loc["Dynamic Regression + ARMA", "MAE"]
exo = imp["ΔMAE permutasi (eksogen-saja)"].dropna().sort_values(ascending=False)
top1, top2, top3 = exo.index[0], exo.index[1], exo.index[2]
best_dl = te[te["kategori"].str.startswith("DL")]["MAE"].idxmin()

a, b, c = st.columns(3)
with a:
    card("🥇 Machine Learning paling akurat",
         f"<b>{best_name}</b> mencapai MAE {best['MAE']:.1f} MWh ({best['MAPE%']:.2f}%), {gain_vs_naive:.0f}% lebih baik dari "
         f"patokan 'minggu lalu jam yang sama'. DL terbaik ({best_dl}) hanya MAE {te.loc[best_dl,'MAE']:.1f} MWh.", "#10B981")
with b:
    card("🔁 Transfer learning membantu, tapi terbatas",
         f"Pre-train + fine-tune menurunkan MAE GRU dari {tl_scratch:.1f} → {tl_full:.1f} MWh (<b>{tl_gain:.1f}%</b> lebih baik) "
         f"dibanding melatih dari nol pada data yang sama. Namun semua GRU masih jauh di bawah LightGBM.", "#8B5CF6")
with c:
    card("🔍 Rutinitas lebih dominan dari cuaca",
         f"Tanpa lag historis, urutan pengaruh: <b>{top1}</b>, <b>{top2}</b>, lalu <b>{top3}</b>. "
         f"Suhu, kelembapan, dan sekolah berpengaruh kecil setelah rutinitas jam/hari diperhitungkan.", "#4F46E5")

st.markdown("&nbsp;")
d, e = st.columns(2)
with d:
    card("🦠 COVID-19 mengubah peringkat",
         f"Saat COVID, MAE terendah dicapai <b>{cov_best_name}</b> ({cov.loc[cov_best_name,'MAE']:.1f} MWh), tetapi LightGBM turun dari "
         f"{pre.loc[best_name,'MAE']:.1f} ke {cov.loc[best_name,'MAE']:.1f} MWh. Dynamic Regression + ARMA memburuk drastis: "
         f"{dyn_pre:.1f} → {dyn_cov:.1f} MWh.", "#EF4444")
with e:
    card("⚠️ Deep Learning belum optimal",
         "Semua varian GRU mencapai error validasi terendah di <b>epoch 1</b> lalu overfitting. Arsitektur perlu diregularisasi "
         "(dropout, model lebih kecil, weight decay) sebelum bisa menyaingi LightGBM. Lihat halaman Perbandingan Model.", "#F59E0B")

st.markdown("&nbsp;")
note("💡 <b>Cara membaca:</b> pilih halaman di sidebar. <b>Eksplorasi Data</b> untuk pola beban, <b>Perbandingan Model</b> untuk "
     "peringkat dan efek transfer learning, <b>Variabel Berpengaruh</b> untuk suhu/libur/sekolah/jam/hari, dan "
     "<b>Forecast Explorer</b> untuk melihat ramalan vs aktual pada tanggal tertentu.")
