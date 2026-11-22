import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (COVID_START, DOW_ID, TEST_END, TEST_START, category_map, load_hourly, load_predictions, metrics,
                   model_color, model_columns, note, page_header, plot, style_fig, table, vline)

page_header("🔮 Forecast Explorer", "Bandingkan ramalan day-ahead dengan beban aktual pada tanggal pilihan Anda (periode Test).")
pred, is_real = load_predictions()
h = load_hourly()
models = model_columns(pred)
cats = category_map()
res_te = None

if not is_real:
    note("ℹ️ <b>Mode terbatas:</b> file <code>data/dashboard_predictions.csv.gz</code> belum ada, jadi hanya model <b>baseline</b> yang bisa "
         "ditampilkan. Jalankan sel <i>Export untuk dashboard</i> di notebook Kaggle, unduh filenya, lalu taruh di folder <code>data/</code> "
         "repo untuk menampilkan LightGBM, Ridge, GRU, dan model lainnya.", "warn")
    st.markdown("")

# ------------------------------------------------------------------ pilih tanggal & model
hol_days = sorted(h.loc[(h["holiday"] == 1) & (h["dow"] < 5) & (h["date"] >= TEST_START) & (h["date"] <= TEST_END), "date"].unique())
presets = {"Pilih sendiri": None, "Hari kerja biasa (13 Nov 2019)": pd.Timestamp("2019-11-13"),
           "Akhir pekan (16 Nov 2019)": pd.Timestamp("2019-11-16"),
           "Awal pembatasan COVID (± 25 Mar 2020)": pd.Timestamp("2020-03-25")}
for d in hol_days[:6]:
    presets[f"Libur nasional ({pd.Timestamp(d).strftime('%d %b %Y')})"] = pd.Timestamp(d)

c1, c2, c3 = st.columns([1.6, 1.2, 1.2])
with c1:
    pre_sel = st.selectbox("Lompat cepat ke", list(presets.keys()))
with c2:
    default_day = presets[pre_sel] or pd.Timestamp("2019-11-13")
    day = st.date_input("Tanggal", value=default_day.date(), min_value=TEST_START.date(), max_value=TEST_END.date())
with c3:
    span = st.radio("Rentang", ["1 hari (24 jam)", "7 hari"], horizontal=True)

order = sorted(models, key=lambda m: (cats.get(m, "").startswith("ML (abl"), m))
default = [m for m in ["LightGBM (ML)", "Ridge (ML linear)", "Naive-168 (minggu lalu)"] if m in models] or models[:2]
chosen = st.multiselect("Model yang ditampilkan", order, default=default)

t0 = pd.Timestamp(day)
t1 = t0 + pd.Timedelta(days=1 if span.startswith("1") else 7)
w = pred[(pred["datetime"] >= t0) & (pred["datetime"] < t1)]
if w.empty:
    st.warning("Tidak ada data pada rentang itu."); st.stop()

# ------------------------------------------------------------------ grafik ramalan vs aktual
fig = go.Figure()
fig.add_trace(go.Scatter(x=w["datetime"], y=w["actual"], mode="lines", name="Aktual", line=dict(color="#0F172A", width=3.2),
                         hovertemplate="%{x|%d %b %H:%M}<br>Aktual: %{y:.0f} MWh<extra></extra>"))
for m in chosen:
    fig.add_trace(go.Scatter(x=w["datetime"], y=w[m], mode="lines", name=m,
                             line=dict(color=model_color(m, models), width=2, dash="dot" if "Naive" in m else "solid"),
                             hovertemplate="%{x|%d %b %H:%M}<br>" + m + ": %{y:.0f} MWh<extra></extra>"))
fig.update_yaxes(title="MWh")
plot(style_fig(fig, 430, f"Ramalan vs aktual: {t0.strftime('%d %b %Y')} ({DOW_ID[t0.dayofweek]})"))

# ------------------------------------------------------------------ metrik pada rentang terpilih
if chosen:
    rows = []
    for m in chosen:
        ok = w[m].notna() & w["actual"].notna()
        if ok.any():
            rows.append({"Model": m, "Kategori": cats.get(m, "-"), **metrics(w.loc[ok, "actual"], w.loc[ok, m])})
    if rows:
        st.markdown("#### Akurasi pada rentang terpilih")
        table(pd.DataFrame(rows).set_index("Model").sort_values("MAE").round(2))

# ------------------------------------------------------------------ analisis galat seluruh test
if chosen:
    st.markdown("### Analisis galat pada seluruh periode Test")
    a, b = st.columns(2)
    full = pred.copy()
    full["hour"] = full["datetime"].dt.hour
    with a:
        fig = go.Figure()
        for m in chosen:
            e = (full[m] - full["actual"]).abs().groupby(full["hour"]).mean()
            fig.add_trace(go.Scatter(x=e.index, y=e.values, mode="lines+markers", name=m,
                                     line=dict(color=model_color(m, models), width=2)))
        fig.update_layout(xaxis_title="Jam target", yaxis_title="MAE (MWh)")
        plot(style_fig(fig, 360, "MAE per jam dalam sehari"))
    with b:
        fig = go.Figure()
        for m in chosen:
            e = (full[m] - full["actual"]).abs().groupby(full["datetime"].dt.normalize()).mean().rolling(7, min_periods=3).mean()
            fig.add_trace(go.Scatter(x=e.index, y=e.values, mode="lines", name=m, line=dict(color=model_color(m, models), width=2)))
        vline(fig, COVID_START, "#EF4444", "COVID")
        fig.update_layout(xaxis_title="", yaxis_title="MAE (MWh)")
        plot(style_fig(fig, 360, "MAE harian (rata-rata bergulir 7 hari)"))

    best = chosen[0]
    sc = go.Figure(go.Scattergl(x=full["actual"], y=full[best], mode="markers", marker=dict(size=3, color=model_color(best, models), opacity=.35)))
    lo, hi = float(full["actual"].min()), float(full["actual"].max())
    sc.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#EF4444", dash="dash"))
    sc.update_layout(xaxis_title="Aktual (MWh)", yaxis_title="Prediksi (MWh)")
    with st.expander(f"Scatter aktual vs prediksi: {best}"):
        plot(style_fig(sc, 420, None, legend_bottom=False))

st.download_button("⬇️ Unduh data rentang ini (CSV)", data=w[["datetime", "actual"] + chosen].to_csv(index=False).encode("utf-8"),
                   file_name=f"prediksi_{t0.date()}.csv", mime="text/csv")
