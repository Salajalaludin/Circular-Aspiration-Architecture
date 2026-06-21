"""EduVoice dashboard for the Circular Aspiration Architecture prototype."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "organized_csv"
WEIGHT_DIR = DATA_DIR / "07_weighting"
LDA_DIR = DATA_DIR / "06_lda"

COLORS = {
    "background": "#101417",
    "surface": "#171D21",
    "surface_alt": "#20272C",
    "border": "#303A40",
    "text": "#F1F4F5",
    "muted": "#9AA7AD",
    "cyan": "#3CB7C8",
    "green": "#4DAA72",
    "amber": "#D99B42",
    "red": "#D76055",
    "blue": "#4F83CC",
}

SENTIMENT_COLORS = {
    "Negatif": COLORS["red"],
    "Netral": COLORS["muted"],
    "Positif": COLORS["green"],
}
REGION_COLORS = {"Jakarta": COLORS["blue"], "Banten": COLORS["green"]}

st.set_page_config(
    page_title="EduVoice | Aspirasi Pendidikan",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    :root {{ color-scheme: dark; }}
    .stApp {{ background: {COLORS['background']}; color: {COLORS['text']}; }}
    [data-testid="stSidebar"] {{
        background: {COLORS['surface']};
        border-right: 1px solid {COLORS['border']};
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ max-width: 1500px; padding-top: 1.4rem; padding-bottom: 2rem; }}
    h1, h2, h3 {{ letter-spacing: 0; }}
    h1 {{ font-size: 1.65rem !important; margin-bottom: .15rem !important; }}
    h2 {{ font-size: 1.15rem !important; margin-top: .7rem !important; }}
    .app-kicker {{ color: {COLORS['cyan']}; font-size: .72rem; font-weight: 700; text-transform: uppercase; }}
    .app-subtitle {{ color: {COLORS['muted']}; font-size: .84rem; margin-bottom: 1rem; }}
    .scope-note {{
        color: {COLORS['muted']}; background: {COLORS['surface']};
        border-left: 3px solid {COLORS['amber']}; padding: .7rem .9rem;
        font-size: .78rem; margin: .4rem 0 1rem;
    }}
    [data-testid="stMetric"] {{
        background: {COLORS['surface']}; border: 1px solid {COLORS['border']};
        border-radius: 6px; padding: .75rem .9rem;
    }}
    [data-testid="stMetricLabel"] {{ color: {COLORS['muted']}; }}
    [data-testid="stMetricValue"] {{ font-size: 1.55rem; }}
    [data-testid="stMetricDelta"] {{ font-size: .75rem; }}
    .status-row {{
        display: flex; justify-content: space-between; gap: 1rem; padding: .5rem 0;
        border-bottom: 1px solid {COLORS['border']}; font-size: .78rem;
    }}
    .status-label {{ color: {COLORS['muted']}; }}
    .status-value {{ color: {COLORS['text']}; font-weight: 600; text-align: right; }}
    .thin-warning {{ color: {COLORS['amber']}; font-weight: 600; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {COLORS['border']}; border-radius: 6px; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid {COLORS['border']}; }}
    .stTabs [data-baseweb="tab"] {{ padding: .6rem .9rem; }}
    .stTabs [aria-selected="true"] {{ color: {COLORS['cyan']}; border-bottom-color: {COLORS['cyan']}; }}
    hr {{ border-color: {COLORS['border']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def find_file(*paths: Path, required: bool = True) -> Path | None:
    match = next((path for path in paths if path.exists()), None)
    if required and match is None:
        st.error(f"Data wajib tidak ditemukan: {paths[0].name}")
        st.stop()
    return match


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


weighted_path = find_file(
    WEIGHT_DIR / "weighted_analysis_results.csv",
    ROOT / "sentiment_results.csv",
)
coverage_keyword_path = find_file(WEIGHT_DIR / "coverage_by_keyword.csv")
coverage_region_path = find_file(WEIGHT_DIR / "coverage_by_region.csv")
diagnostics_path = find_file(WEIGHT_DIR / "weight_diagnostics.csv")
bootstrap_path = find_file(WEIGHT_DIR / "cluster_bootstrap_intervals.csv")
account_path = find_file(WEIGHT_DIR / "account_contribution.csv")
lda_path = find_file(LDA_DIR / "lda_results.csv", ROOT / "lda_results.csv", required=False)
topic_region_path = find_file(LDA_DIR / "weighted_topics_by_region.csv", required=False)

tweets = load_csv(weighted_path)
coverage_keyword = load_csv(coverage_keyword_path)
coverage_region = load_csv(coverage_region_path)
diagnostics = load_csv(diagnostics_path)
bootstrap = load_csv(bootstrap_path)
accounts = load_csv(account_path)
lda = load_csv(lda_path) if lda_path else pd.DataFrame()
topic_region = load_csv(topic_region_path) if topic_region_path else pd.DataFrame()
metadata = load_json(WEIGHT_DIR / "run_metadata.json")

for column in ["weight_operational", "source_weight", "sentiment_value"]:
    if column not in tweets:
        tweets[column] = 1.0 if column != "sentiment_value" else np.nan
    tweets[column] = pd.to_numeric(tweets[column], errors="coerce")
tweets["id"] = tweets["id"].astype(str)
tweets["createdAt"] = pd.to_datetime(tweets.get("createdAt"), errors="coerce", utc=True)
if "analysis_account_id" not in tweets:
    if "author_username" in tweets:
        username = tweets["author_username"].fillna("").astype(str).str.strip().str.lower()
    else:
        username = pd.Series("", index=tweets.index)
    tweets["analysis_account_id"] = np.where(
        username.ne(""), username, "missing_user__" + tweets["id"]
    )


with st.sidebar:
    st.markdown('<div class="app-kicker">EduVoice</div>', unsafe_allow_html=True)
    st.markdown("### Filter analisis")

    region_options = ["Semua", *sorted(tweets["_region"].dropna().unique())]
    selected_region = st.selectbox("Wilayah kueri", region_options)

    region_filtered = tweets if selected_region == "Semua" else tweets[tweets["_region"].eq(selected_region)]
    keyword_options = sorted(region_filtered["_search_keyword"].dropna().unique())
    selected_keywords = st.multiselect("Kueri", keyword_options, placeholder="Semua kueri")
    selected_sentiments = st.multiselect(
        "Sentimen",
        ["Negatif", "Netral", "Positif"],
        default=["Negatif", "Netral", "Positif"],
    )
    weight_mode = st.radio(
        "Mode estimasi",
        ["Bobot operasional", "Tanpa bobot"],
        horizontal=True,
    )

    st.divider()
    st.markdown("### Status data")
    status_items = {
        "Tweet analisis": f"{len(tweets):,}",
        "Akun unik": f"{tweets['analysis_account_id'].nunique():,}",
        "Model topik": "Tersedia" if not lda.empty else "Tidak tersedia",
        "Kalibrasi populasi": "Tidak diterapkan" if metadata.get("population_calibration_status", "").startswith("skipped") else "Diterapkan",
    }
    for label, value in status_items.items():
        st.markdown(
            f'<div class="status-row"><span class="status-label">{label}</span>'
            f'<span class="status-value">{value}</span></div>',
            unsafe_allow_html=True,
        )


filtered = tweets.copy()
if selected_region != "Semua":
    filtered = filtered[filtered["_region"].eq(selected_region)]
if selected_keywords:
    filtered = filtered[filtered["_search_keyword"].isin(selected_keywords)]
filtered = filtered[filtered["sentiment_label"].isin(selected_sentiments)]

weight_column = "weight_operational" if weight_mode == "Bobot operasional" else None
weights = filtered[weight_column] if weight_column else pd.Series(1.0, index=filtered.index)


def weighted_share(mask: pd.Series, values: pd.Series) -> float:
    return float(values[mask].sum() / values.sum()) if len(values) and values.sum() else 0.0


def weighted_average(values: pd.Series, values_weight: pd.Series) -> float:
    valid = values.notna() & values_weight.notna()
    return float(np.average(values[valid], weights=values_weight[valid])) if valid.any() else np.nan


def plot_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=20, r=20, t=50, b=25),
        legend=dict(title_text="", bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08),
        xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    )
    return fig


st.markdown('<div class="app-kicker">Circular Aspiration Architecture</div>', unsafe_allow_html=True)
st.title("Aspirasi Pendidikan Jakarta dan Banten")
st.markdown(
    '<div class="app-subtitle">Pemantauan sentimen, isu dominan, dan kualitas cakupan percakapan publik di X.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="scope-note"><b>Batas interpretasi:</b> hasil menggambarkan percakapan X yang terkumpul. '
    'Wilayah adalah strata kueri, bukan domisili pengguna, dan pembobotan tidak membuat sampel mewakili populasi.</div>',
    unsafe_allow_html=True,
)

negative_share = weighted_share(filtered["sentiment_label"].eq("Negatif"), weights)
mean_sentiment = weighted_average(filtered["sentiment_value"], weights)
unique_accounts = filtered["analysis_account_id"].nunique()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tweet dianalisis", f"{len(filtered):,}")
m2.metric("Akun unik", f"{unique_accounts:,}")
m3.metric("Proporsi negatif", f"{negative_share:.1%}")
m4.metric("Skor sentimen", f"{mean_sentiment:+.3f}" if pd.notna(mean_sentiment) else "n/a", help="Rentang -1 sampai +1")

tab_overview, tab_sentiment, tab_topics, tab_coverage, tab_explorer = st.tabs(
    ["Ringkasan", "Sentimen", "Topik", "Cakupan", "Eksplorasi"]
)


with tab_overview:
    left, right = st.columns([1.05, 1])
    with left:
        sentiment_mass = pd.DataFrame(
            {
                "sentiment": ["Negatif", "Netral", "Positif"],
                "mass": [weights[filtered["sentiment_label"].eq(label)].sum() for label in ["Negatif", "Netral", "Positif"]],
            }
        )
        fig = px.bar(
            sentiment_mass,
            x="mass",
            y="sentiment",
            orientation="h",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            title=f"Komposisi sentimen | {weight_mode}",
            labels={"mass": "Kontribusi", "sentiment": ""},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(plot_layout(fig, 310), use_container_width=True)

    with right:
        region_summary = (
            filtered.assign(_weight=weights)
            .groupby("_region", as_index=False)
            .agg(kontribusi=("_weight", "sum"), tweet=("id", "size"))
        )
        fig = px.bar(
            region_summary,
            x="_region",
            y="kontribusi",
            color="_region",
            color_discrete_map=REGION_COLORS,
            text="tweet",
            title="Kontribusi menurut wilayah kueri",
            labels={"_region": "", "kontribusi": "Kontribusi", "tweet": "Tweet"},
        )
        fig.update_layout(showlegend=False)
        fig.update_traces(texttemplate="%{text:,} tweet", textposition="outside")
        st.plotly_chart(plot_layout(fig, 310), use_container_width=True)

    st.subheader("Ketidakpastian estimasi utama")
    bootstrap_view = bootstrap.copy()
    if selected_region != "Semua":
        bootstrap_view = bootstrap_view[bootstrap_view["query_region"].eq(selected_region)]
    else:
        bootstrap_view = bootstrap_view[bootstrap_view["query_region"].eq("all")]
    if selected_keywords or set(selected_sentiments) != {"Negatif", "Netral", "Positif"}:
        st.caption("Interval bootstrap tersedia untuk sampel utama per wilayah, sebelum filter kueri atau kelas sentimen.")
    display_bootstrap = bootstrap_view.rename(
        columns={
            "query_region": "Wilayah",
            "n_unique_accounts": "Akun",
            "mean_sentiment": "Skor",
            "mean_sentiment_ci_low": "CI 2,5%",
            "mean_sentiment_ci_high": "CI 97,5%",
            "negative_share": "Negatif",
        }
    )[["Wilayah", "Akun", "Skor", "CI 2,5%", "CI 97,5%", "Negatif"]]
    st.dataframe(
        display_bootstrap.style.format({"Skor": "{:+.3f}", "CI 2,5%": "{:+.3f}", "CI 97,5%": "{:+.3f}", "Negatif": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )


with tab_sentiment:
    monthly = filtered.dropna(subset=["createdAt"]).copy()
    monthly["bulan"] = monthly["createdAt"].dt.tz_convert(None).dt.to_period("M").astype(str)
    monthly["_weight"] = weights.loc[monthly.index]
    trend = monthly.groupby(["bulan", "sentiment_label"], as_index=False)["_weight"].sum()
    if not trend.empty:
        fig = px.line(
            trend,
            x="bulan",
            y="_weight",
            color="sentiment_label",
            markers=True,
            color_discrete_map=SENTIMENT_COLORS,
            title="Tren kontribusi sentimen per bulan",
            labels={"bulan": "Bulan", "_weight": "Kontribusi", "sentiment_label": "Sentimen"},
        )
        st.plotly_chart(plot_layout(fig, 360), use_container_width=True)

    by_keyword = filtered.assign(_weight=weights).pivot_table(
        index="_search_keyword",
        columns="sentiment_label",
        values="_weight",
        aggfunc="sum",
        fill_value=0,
    )
    by_keyword = by_keyword.div(by_keyword.sum(axis=1), axis=0).fillna(0) * 100
    by_keyword = by_keyword.sort_values("Negatif", ascending=True) if "Negatif" in by_keyword else by_keyword
    fig = go.Figure()
    for label in ["Negatif", "Netral", "Positif"]:
        if label in by_keyword:
            fig.add_bar(
                y=by_keyword.index,
                x=by_keyword[label],
                name=label,
                orientation="h",
                marker_color=SENTIMENT_COLORS[label],
            )
    fig.update_layout(barmode="stack", title="Komposisi sentimen per kueri", xaxis_title="Persentase", yaxis_title="")
    st.plotly_chart(plot_layout(fig, max(420, 34 * len(by_keyword))), use_container_width=True)


with tab_topics:
    if lda.empty:
        st.info("Keluaran LDA belum tersedia.")
    else:
        topic_base = filtered[["id"]].copy()
        topic_base["_analysis_weight"] = weights.to_numpy()
        topic_ids = topic_base.merge(
            lda[["id", "topic_name", "topic_probability", "topic_review_status", "text", "_region", "_search_keyword"]]
            .assign(id=lambda x: x["id"].astype(str)),
            on="id",
            how="inner",
        )
        topic_ids["topic_mass"] = topic_ids["_analysis_weight"] * topic_ids["topic_probability"]
        topic_summary = topic_ids.groupby("topic_name", as_index=False).agg(
            kontribusi=("topic_mass", "sum"),
            tweet=("id", "size"),
            probabilitas_rata_rata=("topic_probability", "mean"),
        ).sort_values("kontribusi")
        fig = px.bar(
            topic_summary,
            x="kontribusi",
            y="topic_name",
            orientation="h",
            color="probabilitas_rata_rata",
            color_continuous_scale=[[0, COLORS["surface_alt"]], [1, COLORS["cyan"]]],
            title="Prevalensi topik berbasis probabilitas",
            labels={"kontribusi": "Massa topik berbobot", "topic_name": "", "probabilitas_rata_rata": "Keyakinan"},
        )
        st.plotly_chart(plot_layout(fig, 340), use_container_width=True)

        representative = topic_ids.sort_values("topic_probability", ascending=False).drop_duplicates("topic_name")
        representative = representative[["topic_name", "topic_probability", "text", "_region", "_search_keyword"]]
        representative.columns = ["Topik", "Probabilitas", "Tweet representatif", "Wilayah", "Kueri"]
        st.subheader("Tweet representatif")
        st.dataframe(
            representative.style.format({"Probabilitas": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
            column_config={"Tweet representatif": st.column_config.TextColumn(width="large")},
        )


with tab_coverage:
    operational_diag = diagnostics[diagnostics["weight_scheme"].eq("operational")].iloc[0]
    raw_top_share = accounts["unweighted_share"].max()
    weighted_top_share = accounts["operational_share"].max()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Effective sample size", f"{operational_diag['effective_sample_size']:,.0f}")
    c2.metric("Rasio ESS", f"{operational_diag['ess_ratio']:.1%}")
    c3.metric("Dominasi akun terbesar", f"{raw_top_share:.1%}")
    c4.metric("Setelah pembobotan", f"{weighted_top_share:.2%}")

    coverage_view = coverage_keyword.copy()
    if selected_region != "Semua":
        coverage_view = coverage_view[coverage_view["_region"].eq(selected_region)]
    coverage_view["status"] = np.select(
        [coverage_view["n_unique_accounts"].lt(10), coverage_view["n_unique_accounts"].lt(30)],
        ["Sangat tipis", "Tipis"],
        default="Memadai untuk eksplorasi",
    )
    coverage_view = coverage_view.sort_values(["n_unique_accounts", "n_tweets"])

    fig = px.scatter(
        coverage_view,
        x="n_unique_accounts",
        y="n_tweets",
        color="_region",
        color_discrete_map=REGION_COLORS,
        hover_name="_search_keyword",
        size="tweets_per_unique_account",
        title="Cakupan kueri dan konsentrasi akun",
        labels={"n_unique_accounts": "Akun unik", "n_tweets": "Tweet", "_region": "Wilayah"},
    )
    st.plotly_chart(plot_layout(fig, 390), use_container_width=True)

    st.subheader("Audit strata kueri")
    audit_columns = {
        "_region": "Wilayah",
        "_search_keyword": "Kueri",
        "n_tweets": "Tweet",
        "n_unique_accounts": "Akun unik",
        "tweets_per_unique_account": "Tweet per akun",
        "mean_source_weight": "Bobot sumber rata-rata",
        "status": "Status",
    }
    st.dataframe(
        coverage_view[list(audit_columns)].rename(columns=audit_columns).style.format(
            {"Tweet per akun": "{:.2f}", "Bobot sumber rata-rata": "{:.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )


with tab_explorer:
    search_text = st.text_input("Cari teks", placeholder="Contoh: KJP, zonasi, biaya sekolah")
    explorer = filtered.copy()
    if search_text.strip():
        explorer = explorer[explorer["text"].fillna("").str.contains(search_text.strip(), case=False, regex=False)]
    sort_option = st.selectbox("Urutkan", ["Terbaru", "Engagement tertinggi", "Bobot tertinggi"])
    if sort_option == "Terbaru":
        explorer = explorer.sort_values("createdAt", ascending=False)
    elif sort_option == "Engagement tertinggi":
        engagement_columns = [c for c in ["likeCount", "replyCount", "retweetCount", "quoteCount"] if c in explorer]
        explorer["engagement"] = explorer[engagement_columns].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        explorer = explorer.sort_values("engagement", ascending=False)
    else:
        explorer = explorer.sort_values("weight_operational", ascending=False)

    columns = [
        "createdAt", "text", "_region", "_search_keyword", "sentiment_label",
        "sentiment_score", "author_username", "account_class", "weight_operational",
    ]
    columns = [column for column in columns if column in explorer]
    labels = {
        "createdAt": "Waktu",
        "text": "Tweet",
        "_region": "Wilayah",
        "_search_keyword": "Kueri",
        "sentiment_label": "Sentimen",
        "sentiment_score": "Keyakinan",
        "author_username": "Akun",
        "account_class": "Kelas akun",
        "weight_operational": "Bobot",
    }
    st.dataframe(
        explorer[columns].head(200).rename(columns=labels),
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={"Tweet": st.column_config.TextColumn(width="large")},
    )
    st.caption(f"Menampilkan {min(len(explorer), 200):,} dari {len(explorer):,} tweet sesuai filter.")


st.divider()
st.caption(
    "EduVoice | Prototipe lomba esai SMATIC 7.0 | Data X dikumpulkan melalui Apify | "
    "Analisis sentimen IndoBERT, topic modeling LDA, dan diagnostik pembobotan representasi."
)
