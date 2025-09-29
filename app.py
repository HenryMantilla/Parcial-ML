import streamlit as st
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder

import streamlit as st
st.set_page_config(page_title="ML Playground", page_icon="📊", layout="wide")

st.markdown("""
<style>
/* Newer Streamlit: sidebar multipage list */
[data-testid="stSidebarNav"] { display: none !important; }

/* Older/other builds: sometimes the header pills render as a <nav> */
header nav, header [data-testid="stHeaderNav"] { display: none !important; }

/* If a leftover container renders inside the sidebar, hide its UL */
section[data-testid="stSidebar"] ul[role="listbox"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ ML Playground")
    choice = option_menu(
        None,  # no title inside the card
        ["Visualización", "Algoritmos"],
        icons=["graph-up", "code-slash"],
        default_index=0,
        key="menu_sel",
        styles={
            "container": {"padding": "8px", "background-color": "#0e1117", "border-radius": "12px"},
            "icon": {"color": "#e5e7eb", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "color": "#c9d1d9",
                "text-align": "left",
                "margin": "4px 0",
                "padding": "10px 14px",
                "border-radius": "10px",
            },
            "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
        },
    )

    PAGE_MAP = {
    "Visualización": "app.py",
    "Algoritmos": "pages/algorithms.py",  
    }

    if "last_menu_sel" not in st.session_state:
        st.session_state["last_menu_sel"] = choice

    if choice != st.session_state["last_menu_sel"]:
        st.session_state["last_menu_sel"] = choice
        target = PAGE_MAP[choice]
        st.switch_page(target)            

st.title("📈 Visualización de datos")

@st.cache_data(show_spinner=False)
def read_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


def tidy_numeric_desc(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return pd.DataFrame()
    desc = df[num_cols].describe().T.reset_index().rename(columns={"index": "column"})
    # Types/format
    if "count" in desc.columns:
        desc["count"] = pd.to_numeric(desc["count"], errors="coerce").astype("Int64")
    for c in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        if c in desc.columns:
            desc[c] = pd.to_numeric(desc[c], errors="coerce").round(3)
    return desc

def tidy_categorical_desc(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    if not cat_cols:
        return pd.DataFrame()
    # Older pandas: describe() on objects returns count/unique/top/freq
    try:
        desc = df[cat_cols].describe(include="all").T
    except Exception:
        desc = df[cat_cols].astype(str).describe().T
    desc = desc.reset_index().rename(columns={"index": "column"})
    if "count" in desc.columns:
        desc["count"] = pd.to_numeric(desc["count"], errors="coerce").astype("Int64")
    return desc

def info_as_table(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    mem_col = df.memory_usage(index=False, deep=True)
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null": df.notna().sum().values,
        "nulls": df.isna().sum().values,
        "% null": (df.isna().sum().mul(100).div(n)).round(2).values,
        "unique": [df[c].nunique(dropna=True) for c in df.columns],
        "memory_bytes": mem_col.values,
    })

def safe_describe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        desc = df.describe(include="all", datetime_is_numeric=True)
    except TypeError:
        desc = df.describe(include="all")
    desc = desc.T.reset_index().rename(columns={"index": "column"})

    num_stats = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    for c in [c for c in num_stats if c in desc.columns]:
        desc[c] = pd.to_numeric(desc[c], errors="coerce").round(3)
    if "count" in desc.columns:
        desc["count"] = pd.to_numeric(desc["count"], errors="coerce").astype("Int64")
    return desc

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to explore.")
    st.stop()

df = read_csv(uploaded)

with st.expander("⚙️ Limpieza (opcional)"):
    drop_na = st.checkbox("Drop rows with any NA", value=False)
    placeholder = st.text_input("Treat this token as missing (e.g., '?')", value="?")
    apply_placeholder = st.checkbox("Replace placeholder with NA", value=True)

if apply_placeholder and placeholder:
    df = df.replace(placeholder, np.nan)

if drop_na:
    df = df.dropna()

st.session_state["df"] = df.copy()

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)
st.caption(
    f"Rows: {len(df):,} | Columns: {df.shape[1]} | "
    f"Total memory (deep): {df.memory_usage(deep=True).sum():,} bytes"
)

st.subheader("Descriptivo")

# --- Numeric summary (full width) ---
num_desc = tidy_numeric_desc(df)
if not num_desc.empty:
    num_cfg = {"column": st.column_config.TextColumn("Column")}
    # Add only columns that exist
    if "count" in num_desc.columns: num_cfg["count"] = st.column_config.NumberColumn("Count", format="%d")
    for c,label in [("mean","Mean"),("std","Std"),("min","Min"),
                    ("25%","25%"),("50%","50%"),("75%","75%"),("max","Max")]:
        if c in num_desc.columns:
            num_cfg[c] = st.column_config.NumberColumn(label, format="%.3f")
    st.markdown("**Numeric summary**")
    st.dataframe(num_desc, use_container_width=True, column_config=num_cfg)
else:
    st.info("No numeric columns to summarize.")

# --- Categorical summary (full width, below numeric) ---
cat_desc = tidy_categorical_desc(df)
if not cat_desc.empty:
    cat_cfg = {"column": st.column_config.TextColumn("Column")}
    if "count" in cat_desc.columns:  cat_cfg["count"]  = st.column_config.NumberColumn("Count",  format="%d")
    if "unique" in cat_desc.columns: cat_cfg["unique"] = st.column_config.NumberColumn("Unique", format="%d")
    if "top" in cat_desc.columns:    cat_cfg["top"]    = st.column_config.TextColumn("Top")
    if "freq" in cat_desc.columns:   cat_cfg["freq"]   = st.column_config.NumberColumn("Freq",   format="%d")
    st.markdown("**Categorical summary**")
    st.dataframe(cat_desc, use_container_width=True, column_config=cat_cfg)

st.markdown("---")  

st.subheader("Info por columna")
st.dataframe(
    info_as_table(df),
    use_container_width=True,
    column_config={
        "column":        st.column_config.TextColumn("Column"),
        "dtype":         st.column_config.TextColumn("Dtype"),
        "non_null":      st.column_config.NumberColumn("Non-null", format="%d"),
        "nulls":         st.column_config.NumberColumn("Nulls", format="%d"),
        "% null":        st.column_config.NumberColumn("% null", format="%.2f"),
        "unique":        st.column_config.NumberColumn("Unique", format="%d"),
        "memory_bytes":  st.column_config.NumberColumn("Memory (bytes)", format="%d"),
    },
)
# Separate numeric / categorical
numeric_df = df.select_dtypes(include="number")
categoric_df = df.select_dtypes(exclude="number")

# Tidy names (optional)
df_plot_numeric = numeric_df.copy()
df_plot_numeric.columns = (df_plot_numeric.columns
                           .str.strip().str.replace(r"[^\w]+", "_", regex=True))
df_plot_categoric = categoric_df.copy()
df_plot_categoric.columns = (df_plot_categoric.columns
                             .str.strip().str.replace(r"[^\w]+", "_", regex=True))

with st.expander("Distribuciones (opcional)"):
    if not df_plot_numeric.empty:
        col = st.selectbox("Numeric column", df_plot_numeric.columns)
        st.bar_chart(df_plot_numeric[col].value_counts().sort_index())
    if not df_plot_categoric.empty:
        colc = st.selectbox("Categorical column", df_plot_categoric.columns)
        st.bar_chart(df_plot_categoric[colc].astype(str).value_counts())

st.subheader("Correlación (num + categ codificada)")

if not categoric_df.empty:
    df_cat2num = categoric_df.apply(lambda c: LabelEncoder().fit_transform(c.astype(str)))
    corr_df = pd.concat([numeric_df, df_cat2num], axis=1)
else:
    corr_df = numeric_df

if corr_df.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(corr_df.corr(numeric_only=True), cbar=False, ax=ax, annot=False)
    st.pyplot(fig)
else:
    st.info("Need ≥2 columns to compute correlations.")
