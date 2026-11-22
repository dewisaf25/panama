import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (CAT_COLORS, LOG_TAGS, TL_MODELS, cat_color, load_results, load_training_log, note, page_header,
                   plot, style_fig, table, chip)

page_header("🏆 Perbandingan Model", "Klasik vs Machine Learning vs Deep Learning, termasuk dampak transfer learning dan guncangan COVID-19.")
res = load_results()

# ------------------------------------------------------------------ kontrol
c1, c2, c3 = st.columns([2, 1.2, 1.4])
with c1:
    periode = st.radio("Periode evaluasi", list(res.keys()), horizontal=True)
with c2:
    metric = st.selectbox("Metrik", ["MAE", "RMSE", "MAPE%", "R2"])
with c3:
    show_abl = st.checkbox("Sertakan model ablation (eksogen-saja)", value=False,
                           help="Model LightGBM tanpa fitur lag historis. Dipakai untuk analisis variabel, bukan kandidat model.")

df = res[periode].copy()
if not show_abl:
    df = df[~df["kategori"].str.contains("ablation")]
asc = metric != "R2"
df = df.sort_values(metric, ascending=asc)
naive_mae = res[periode].loc["Naive-168 (minggu lalu)", "MAE"]
df["Skill vs Naive-168 (%)"] = 100 * (1 - df["MAE"] / naive_mae)

# ------------------------------------------------------------------ bar chart
plot_df = df.reset_index()
fig = go.Figure()
for cat, g in plot_df.groupby("kategori"):
    fig.add_trace(go.Bar(x=g[metric], y=g["model"], orientation="h", name=cat, marker_color=cat_color(cat),
                         text=[f"{v:.2f}" if metric in ("R2", "MAPE%") else f"{v:.1f}" for v in g[metric]],
                         textposition="outside", hovertemplate="%{y}<br>" + metric + ": %{x:.3f}<extra></extra>"))
fig.update_yaxes(categoryorder="array", categoryarray=list(plot_df["model"])[::-1], title="")
fig.update_xaxes(title=metric + (" (MWh)" if metric in ("MAE", "RMSE") else ""))
if metric == "MAE":
    fig.add_shape(type="line", xref="x", yref="paper", x0=naive_mae, x1=naive_mae, y0=0, y1=1,
                  line=dict(color="#64748B", dash="dash", width=1.5))
    fig.add_annotation(x=naive_mae, y=1, yref="paper", text="Naive-168", showarrow=False, xanchor="left",
                       yanchor="bottom", font=dict(size=11, color="#64748B"))
fig.update_layout(barmode="overlay")
plot(style_fig(fig, max(380, 38 * len(plot_df)), f"{metric} per model: {periode}"))

best = df.index[0]
st.markdown(f"**Terbaik pada {metric} ({periode}):** {best} · "
            + chip(df.loc[best, 'kategori'], cat_color(df.loc[best, 'kategori'])), unsafe_allow_html=True)
if periode == "Saat COVID":
    note("Saat COVID, urutan berubah bergantung metrik: <b>MAE</b> dan <b>RMSE</b> bisa menunjuk model berbeda. RMSE menghukum "
         "error besar (mis. salah memperkirakan jam puncak), sedangkan MAE lebih mewakili galat rata-rata. Bandingkan keduanya.")

with st.expander("📋 Tabel lengkap", expanded=False):
    table(df.round(3), column_config={
        "Skill vs Naive-168 (%)": st.column_config.ProgressColumn("Skill vs Naive-168 (%)", min_value=-100, max_value=100, format="%.1f"),
    })

# ------------------------------------------------------------------ kategori terbaik
st.markdown("### Model terbaik per kategori")
base_df = res[periode][~res[periode]["kategori"].str.contains("ablation")]
cats = base_df.groupby("kategori")["MAE"].agg(["min", "idxmin"]).sort_values("min")
cols = st.columns(len(cats))
for col, (cat, row) in zip(cols, cats.iterrows()):
    with col:
        st.markdown(f'<div class="kpi" style="--accent:{cat_color(cat)}"><div class="kpi-l">{cat}</div>'
                    f'<div class="kpi-v">{row["min"]:.1f}</div><div class="kpi-s">{row["idxmin"]}</div></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ COVID dumbbell
st.markdown("### Ketahanan terhadap COVID-19 (MAE pra-COVID → saat COVID)")
pre, cov = res["Pra-COVID"], res["Saat COVID"]
dd = pd.DataFrame({"pre": pre["MAE"], "cov": cov["MAE"], "kat": pre["kategori"]})
dd = dd[~dd["kat"].str.contains("ablation")].sort_values("pre", ascending=False)
fig = go.Figure()
for name, r in dd.iterrows():
    fig.add_trace(go.Scatter(x=[r["pre"], r["cov"]], y=[name, name], mode="lines", line=dict(color="#CBD5E1", width=3),
                             showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=dd["pre"], y=dd.index, mode="markers", name="Pra-COVID", marker=dict(color="#10B981", size=12)))
fig.add_trace(go.Scatter(x=dd["cov"], y=dd.index, mode="markers", name="Saat COVID", marker=dict(color="#EF4444", size=12)))
fig.update_layout(xaxis_title="MAE (MWh)", yaxis_title="")
plot(style_fig(fig, max(380, 34 * len(dd)), None))
worst = (dd["cov"] / dd["pre"]).idxmax()
st.caption(f"Pemburuk terbesar: **{worst}** ({dd.loc[worst,'pre']:.1f} → {dd.loc[worst,'cov']:.1f} MWh). "
           "Model yang mengandalkan pola historis jangka panjang paling terpukul oleh perubahan perilaku mendadak.")

# ------------------------------------------------------------------ transfer learning
st.markdown("### 🔁 Transfer learning vs non-transfer (GRU)")
st.markdown("Skenario: **pre-train** pada data lama (train sebelum 365 hari terakhir), lalu **fine-tune** pada 365 hari terakhir train. "
            "Pembandingnya: GRU yang dilatih **dari nol** pada 365 hari terakhir saja, dan GRU dengan **seluruh** data train.")
rows = []
for key, name in TL_MODELS.items():
    r = {"Model": name, "Jenis": res["Test total"].loc[name, "kategori"]}
    for p in res:
        r[f"MAE {p}"] = res[p].loc[name, "MAE"]
    rows.append(r)
tl = pd.DataFrame(rows)
ref = tl.loc[tl["Model"] == TL_MODELS["scratch_1y"]].iloc[0]
for p in res:
    tl[f"Δ vs scratch-1th ({p})"] = 100 * (1 - tl[f"MAE {p}"] / ref[f"MAE {p}"])
table(tl.set_index("Model").round(2), column_config={
    f"Δ vs scratch-1th ({p})": st.column_config.NumberColumn(f"Δ % ({p})", format="%.1f") for p in res})

figtl = go.Figure()
for p, col in zip(res, ["#4F46E5", "#10B981", "#EF4444"]):
    figtl.add_trace(go.Bar(x=tl["Model"], y=tl[f"MAE {p}"], name=p, marker_color=col,
                           hovertemplate="%{x}<br>MAE %{y:.1f}<extra>" + p + "</extra>"))
figtl.update_layout(barmode="group", yaxis_title="MAE (MWh)", xaxis_title="")
plot(style_fig(figtl, 380, "MAE model GRU per periode"))

g_full = 100 * (1 - res["Test total"].loc[TL_MODELS["tf_full"], "MAE"] / res["Test total"].loc[TL_MODELS["scratch_1y"], "MAE"])
g_all = res["Test total"].loc[TL_MODELS["scratch_all"], "MAE"]
note(f"<b>Hasil:</b> transfer learning memperbaiki GRU data-terbatas sebesar ±{g_full:.1f}% (Test), paling terasa saat COVID. "
     f"GRU yang dilatih dengan <i>seluruh</i> data train (MAE {g_all:.1f}) {'sudah setara atau lebih baik' if g_all <= res['Test total'].loc[TL_MODELS['tf_full'], 'MAE'] else 'masih lebih buruk'} "
     "dibanding varian transfer, sehingga manfaat transfer paling relevan ketika data target terbatas.")

log = load_training_log()
if log is not None:
    st.markdown("#### Kurva validasi GRU: tanda overfitting")
    fig = go.Figure()
    for i, (m, g) in enumerate(log.groupby("model")):
        fig.add_trace(go.Scatter(x=g["epoch"], y=g["val_mae"], mode="lines+markers", name=m,
                                 hovertemplate="epoch %{x}<br>val MAE %{y:.1f}<extra>" + m + "</extra>"))
        b = g.loc[g["val_mae"].idxmin()]
        fig.add_trace(go.Scatter(x=[b["epoch"]], y=[b["val_mae"]], mode="markers", showlegend=False,
                                 marker=dict(symbol="star", size=14, color="#EF4444"), hoverinfo="skip"))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="MAE validasi (MWh)")
    plot(style_fig(fig, 380, "MAE validasi per epoch (⭐ = epoch terbaik)"))
    note("Semua varian GRU mencapai MAE validasi terbaik di <b>epoch 1</b>, sementara error training terus turun. Model menghafal data "
         "latih. Perbaikan yang disarankan: dropout lebih besar, hidden size lebih kecil, weight decay lebih kuat, learning rate lebih rendah, "
         "atau input jendela yang lebih pendek.", "warn")
