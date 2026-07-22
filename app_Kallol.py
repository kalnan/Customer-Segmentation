"""
app_Kallol.py
=============
Universal Customer Segmentation Web App

Upload any customer Excel/CSV file and get the same depth of analysis as the
reference segmentation notebooks: data understanding, cleaning, feature
engineering, EDA, optimal-k selection, K-Means clustering, PCA visualization,
and per-segment profiling — all interactive, in the browser.

Run with:
    streamlit run app_Kallol.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(page_title="Customer Segmentation", layout="wide", page_icon="📊")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def list_sheets(file_bytes, name):
    if name.lower().endswith(".csv"):
        return None
    import io
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def load_file(file_bytes, name, sheet_name=None):
    import io
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


def unique_colname(preferred: str, existing_cols) -> str:
    """Return `preferred` if it's not already a column name; otherwise a de-duplicated variant.
    This is what prevents any internally-generated column (like our cluster label) from ever
    silently colliding with a column that already exists in the user's uploaded file."""
    existing = set(existing_cols)
    if preferred not in existing:
        return preferred
    i = 2
    while f"{preferred}_{i}" in existing:
        i += 1
    return f"{preferred}_{i}"


def is_id_like(series: pd.Series, n_rows: int) -> bool:
    """Heuristic: near-unique numeric/text column -> likely an ID, exclude from clustering."""
    nunique = series.nunique(dropna=True)
    return nunique >= 0.95 * n_rows


def build_feature_matrix(df, numeric_cols, ordinal_map_cols, nominal_cols):
    """Encode a mix of numeric / ordinal / nominal columns into one numeric matrix."""
    parts = []

    if numeric_cols:
        num_part = df[numeric_cols].copy()
        for c in numeric_cols:
            num_part[c] = num_part[c].fillna(num_part[c].median())
        parts.append(num_part)

    for col, mapping in ordinal_map_cols.items():
        s = df[col].map(mapping)
        s = s.fillna(s.median())
        parts.append(s.rename(f"{col}_ord"))

    if nominal_cols:
        nom_part = df[nominal_cols].copy()
        for c in nominal_cols:
            mode_series = nom_part[c].mode(dropna=True)
            fill_val = mode_series.iloc[0] if len(mode_series) else "Unknown"
            nom_part[c] = nom_part[c].fillna(fill_val)
        dummies = pd.get_dummies(nom_part, prefix=nominal_cols, dtype=int)
        parts.append(dummies)

    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def grouped_long_table(df, group_col, value_cols):
    """Mean of `value_cols` per `group_col`, returned in long format for a snake plot.
    Built without reset_index()/insert() so it can never collide with an existing column
    name — the group key is (re)assigned directly instead."""
    value_cols = [c for c in value_cols if c != group_col]  # never aggregate the group key itself
    if not value_cols:
        return pd.DataFrame(columns=[group_col, "Feature", "Value"])
    grouped = df.groupby(group_col)[value_cols].mean()
    grouped[group_col] = grouped.index
    long_df = grouped.melt(id_vars=group_col, var_name="Feature", value_name="Value")
    return long_df


# ----------------------------------------------------------------------
# Sidebar — Upload
# ----------------------------------------------------------------------

st.title("📊 Customer Segmentation Studio")
st.caption("Upload any customer dataset (CSV or Excel) and get a full clustering analysis with interactive visuals — data understanding, cleaning, feature engineering, optimal-k selection, K-Means, PCA, and segment profiles.")

with st.sidebar:
    st.header("1. Upload data")
    uploaded_file = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
    st.markdown("---")
    st.caption("Built to match a consistent segmentation methodology: understand → clean → engineer features → find optimal k → cluster → profile.")

if uploaded_file is None:
    st.info("👈 Upload a CSV or Excel file to begin.")
    st.stop()

file_bytes = uploaded_file.getvalue()
sheet_names = list_sheets(file_bytes, uploaded_file.name)
sheet_choice = None
if sheet_names and len(sheet_names) > 1:
    with st.sidebar:
        sheet_choice = st.selectbox("Sheet", sheet_names)
elif sheet_names:
    sheet_choice = sheet_names[0]

df_raw = load_file(file_bytes, uploaded_file.name, sheet_choice)

if df_raw.empty or df_raw.shape[1] == 0:
    st.error("This file has no usable rows/columns.")
    st.stop()

n_rows, n_cols = df_raw.shape

# Pick a collision-proof name for the cluster label column up front, and keep it out of any
# feature-selection widget so it can never be re-selected as a clustering input.
CLUSTER_COL = unique_colname("Cluster", df_raw.columns)
selectable_cols = [c for c in df_raw.columns if c != CLUSTER_COL]

# ----------------------------------------------------------------------
# Section 1 — Data Understanding
# ----------------------------------------------------------------------

st.header("1. Data Understanding")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{n_rows:,}")
c2.metric("Columns", n_cols)
c3.metric("Duplicate rows", int(df_raw.duplicated().sum()))

with st.expander("Sample rows", expanded=True):
    st.dataframe(df_raw.head(10), use_container_width=True)

with st.expander("Column types & missing values"):
    info_tbl = pd.DataFrame({
        "dtype": df_raw.dtypes.astype(str),
        "missing": df_raw.isnull().sum(),
        "missing_%": (df_raw.isnull().mean() * 100).round(2),
        "unique_values": df_raw.nunique(),
    })
    st.dataframe(info_tbl, use_container_width=True)
    miss = info_tbl[info_tbl["missing"] > 0].sort_values("missing", ascending=False)
    if len(miss):
        fig = px.bar(miss, x=miss.index, y="missing_%", title="Missing Values by Column (%)")
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Summary statistics"):
    st.dataframe(df_raw.describe(include="all").T, use_container_width=True)

# ----------------------------------------------------------------------
# Section 2 — Feature Selection
# ----------------------------------------------------------------------

st.header("2. Choose Features for Segmentation")

suggested_id_cols = [c for c in selectable_cols if is_id_like(df_raw[c], n_rows)]
suggested_features = [c for c in selectable_cols if c not in suggested_id_cols]

numeric_cols_all = [c for c in suggested_features if pd.api.types.is_numeric_dtype(df_raw[c])]
categorical_cols_all = [c for c in suggested_features if not pd.api.types.is_numeric_dtype(df_raw[c])]

col_a, col_b = st.columns(2)
with col_a:
    numeric_cols = st.multiselect(
        "Numeric features (used as-is, median-imputed)",
        options=[c for c in selectable_cols if pd.api.types.is_numeric_dtype(df_raw[c])],
        default=numeric_cols_all,
    )
with col_b:
    nominal_cols = st.multiselect(
        "Categorical features (one-hot encoded, mode-imputed)",
        options=[c for c in selectable_cols if not pd.api.types.is_numeric_dtype(df_raw[c])],
        default=[c for c in categorical_cols_all if df_raw[c].nunique(dropna=True) <= 15],
    )

st.caption("Columns that look ID-like (near-unique values) are excluded from the defaults automatically — you can still add them above if needed.")

ordinal_cols = st.multiselect(
    "Any ordinal categorical column? (has a natural order, e.g. Low/Average/High)",
    options=list(nominal_cols),
)
ordinal_map_cols = {}
if ordinal_cols:
    st.caption("Set the order (lowest → highest) for each ordinal column:")
    for c in ordinal_cols:
        nominal_cols.remove(c)
        levels = sorted(df_raw[c].dropna().unique().tolist(), key=str)
        ordered = st.multiselect(f"Order for '{c}'", options=levels, default=levels, key=f"ord_{c}")
        ordinal_map_cols[c] = {v: i for i, v in enumerate(ordered)}

if not numeric_cols and not nominal_cols and not ordinal_map_cols:
    st.warning("Select at least one feature to proceed.")
    st.stop()

# ----------------------------------------------------------------------
# Section 3 — Cleaning + Feature Engineering + Scaling
# ----------------------------------------------------------------------

st.header("3. Cleaning, Feature Engineering & Scaling")

df = df_raw.copy()
features = build_feature_matrix(df, numeric_cols, ordinal_map_cols, nominal_cols)

if features.shape[1] == 0:
    st.error("No usable features after encoding. Pick different columns above.")
    st.stop()

st.write(f"Engineered feature matrix: **{features.shape[0]} rows × {features.shape[1]} columns** "
         f"(numeric passthrough + ordinal encoding + one-hot dummies, missing values imputed).")
with st.expander("Preview engineered features"):
    st.dataframe(features.head(10), use_container_width=True)

# Drop any zero-variance columns — they add nothing to distance-based clustering and can
# distort StandardScaler (division by a std of 0).
zero_var_cols = [c for c in features.columns if features[c].std(ddof=0) == 0]
if zero_var_cols:
    st.caption(f"Dropping {len(zero_var_cols)} constant (zero-variance) column(s) from the feature set: {', '.join(zero_var_cols[:8])}{'...' if len(zero_var_cols) > 8 else ''}")
    features = features.drop(columns=zero_var_cols)

if features.shape[1] == 0:
    st.error("All selected features are constant — nothing left to cluster on. Pick different columns above.")
    st.stop()

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)

if len(numeric_cols) >= 1:
    with st.expander("EDA — numeric distributions & correlation"):
        sel = st.multiselect("Numeric columns to plot", numeric_cols, default=numeric_cols[:3])
        for c in sel:
            fig = px.histogram(df, x=c, nbins=30, marginal="box", title=f"Distribution of {c}")
            st.plotly_chart(fig, use_container_width=True)
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             title="Correlation Heatmap — Numeric Features")
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Section 4 — Optimal k
# ----------------------------------------------------------------------

st.header("4. Finding the Optimal Number of Segments (k)")

if n_rows < 4:
    st.error("Need at least 4 rows to run clustering.")
    st.stop()

max_k = max(2, min(10, n_rows // 5, n_rows - 1))
k_range = list(range(2, max_k + 1)) if max_k >= 2 else [2]

with st.spinner(f"Evaluating k = 2 to {max_k} ..."):
    inertias, sil_scores, dbi_scores = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        labels_k = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels_k))
        dbi_scores.append(davies_bouldin_score(X_scaled, labels_k))

metrics_df = pd.DataFrame({"k": k_range, "Inertia": inertias, "Silhouette": sil_scores, "Davies-Bouldin": dbi_scores})
suggested_k = int(metrics_df.loc[metrics_df["Silhouette"].idxmax(), "k"])

c1, c2 = st.columns(2)
with c1:
    fig = px.line(metrics_df, x="k", y="Inertia", markers=True, title="Elbow Method (Inertia)")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.line(metrics_df, x="k", y="Silhouette", markers=True, title="Silhouette Score (higher = better)")
    st.plotly_chart(fig, use_container_width=True)

st.info(f"Suggested k based on best Silhouette Score: **{suggested_k}**")

# ----------------------------------------------------------------------
# Section 5 — Clustering
# ----------------------------------------------------------------------

st.header("5. Run Segmentation")
k = st.slider("Number of segments (k)", min_value=2, max_value=max(2, max_k), value=min(suggested_k, max_k))

kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=500, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
df[CLUSTER_COL] = cluster_labels
features[CLUSTER_COL] = cluster_labels

sil = silhouette_score(X_scaled, cluster_labels)
dbi = davies_bouldin_score(X_scaled, cluster_labels)
chi = calinski_harabasz_score(X_scaled, cluster_labels)

m1, m2, m3 = st.columns(3)
m1.metric("Silhouette Score", f"{sil:.3f}")
m2.metric("Davies-Bouldin", f"{dbi:.3f}", help="Lower is better")
m3.metric("Calinski-Harabasz", f"{chi:,.0f}")

# PCA (guard against fewer than 2 usable feature dimensions)
n_components = min(2, features.shape[1] - 1) if features.shape[1] > 1 else 1
if n_components >= 2:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df["PCA1"], df["PCA2"] = coords[:, 0], coords[:, 1]
    fig = px.scatter(df, x="PCA1", y="PCA2", color=df[CLUSTER_COL].astype(str),
                      title=f"Customer Segments — PCA 2D Projection (k={k})",
                      labels={"color": "Cluster"}, opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Not enough feature dimensions for a 2D PCA plot — add more features above to see the scatter view.")

# ----------------------------------------------------------------------
# Section 6 — Segment Profiles
# ----------------------------------------------------------------------

st.header("6. Segment Profiles")

seg_counts = df[CLUSTER_COL].value_counts().sort_index()
fig = px.bar(x=seg_counts.index.astype(str), y=seg_counts.values,
             labels={"x": "Cluster", "y": "Customers"}, title="Segment Size Distribution")
st.plotly_chart(fig, use_container_width=True)

profile_cols = [c for c in (numeric_cols + [f"{c}_ord" for c in ordinal_map_cols]) if c in features.columns and c != CLUSTER_COL]
if profile_cols:
    profile = features.groupby(CLUSTER_COL)[profile_cols].mean().round(2)
    st.subheader("Average feature values per cluster")
    st.dataframe(profile, use_container_width=True)

    norm = features[profile_cols + [CLUSTER_COL]].copy()
    for c in profile_cols:
        rng = norm[c].max() - norm[c].min()
        norm[c] = (norm[c] - norm[c].min()) / rng if rng else 0.0

    snake = grouped_long_table(norm, CLUSTER_COL, profile_cols)
    fig = px.line(snake, x="Feature", y="Value", color=snake[CLUSTER_COL].astype(str), markers=True,
                  title="Snake Plot — Normalized Feature Profile per Cluster", labels={"color": "Cluster"})
    st.plotly_chart(fig, use_container_width=True)

if nominal_cols:
    pick = st.selectbox("Categorical breakdown by cluster", nominal_cols)
    mix = pd.crosstab(df[CLUSTER_COL], df[pick], normalize="index").round(3) * 100
    fig = px.bar(mix, barmode="stack", title=f"{pick} Mix by Cluster (%)")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Section 7 — Download
# ----------------------------------------------------------------------

st.header("7. Download Results")
drop_cols = [c for c in ["PCA1", "PCA2"] if c in df.columns]
out_csv = df.drop(columns=drop_cols).to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download segmented customer data (CSV)", out_csv, file_name="segmented_customers.csv", mime="text/csv")

st.caption("Tip: adjust the feature selections or k above and the whole analysis re-runs automatically.")
