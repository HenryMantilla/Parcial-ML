import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, auc, roc_curve as sk_roc_curve,
    classification_report, confusion_matrix
)

# ------------------------- Small helper: manual CV (avoids sklearn.clone) -------------------------
def stratified_cv_scores(model_factory, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, va in skf.split(X, y):
        clf = model_factory()
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[va])
        scores.append(accuracy_score(y[va], y_pred))
    return float(np.mean(scores)), float(np.std(scores))

st.set_page_config(page_title="ML Playground — Algoritmos", page_icon="🧠", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
header nav, header [data-testid="stHeaderNav"] { display: none !important; }
section[data-testid="stSidebar"] ul[role="listbox"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

from streamlit_option_menu import option_menu
with st.sidebar:
    st.markdown("## ⚙️ ML Playground")
    choice = option_menu(
        None, ["Visualización", "Algoritmos"],
        icons=["graph-up", "code-slash"], default_index=1, key="menu_sel",
        styles={
            "container": {"padding": "8px", "background-color": "#0e1117", "border-radius": "12px"},
            "icon": {"color": "#e5e7eb", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "color": "#c9d1d9", "text-align": "left",
                         "margin": "4px 0", "padding": "10px 14px", "border-radius": "10px"},
            "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
        },
    )

PAGE_MAP = {"Visualización": "app.py", "Algoritmos": "pages/algorithms.py"}
if "last_menu_sel" not in st.session_state:
    st.session_state["last_menu_sel"] = "Algoritmos"
if choice != st.session_state["last_menu_sel"]:
    st.session_state["last_menu_sel"] = choice
    target = PAGE_MAP[choice]
    st.switch_page(target)
st.title("🧠 Construcción y evaluación de modelos")

# ------------------------- Plots & reports -------------------------
def plot_roc_streamlit(model, X_test, y_test, title="ROC Curve"):
    # Binary ROC plot; multiclass handled elsewhere with macro-OvR AUC only
    if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
        st.info("El modelo no expone puntajes continuos para ROC.")
        return
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) != 2:
        st.info("ROC binaria solo para problemas con 2 clases.")
        return

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    fpr, tpr, _ = sk_roc_curve(y_test, y_score)
    auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right")
    st.pyplot(fig)

def metrics_block(y_true, y_pred, y_score=None, classes_=None):
    acc = accuracy_score(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # AUC
    auc_val, auc_note = None, ""
    if y_score is not None:
        if y_score.ndim == 1:
            try:
                auc_val = roc_auc_score(y_true, y_score)
            except Exception:
                auc_val = None
        else:
            try:
                auc_val = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
                auc_note = " (macro-OvR)"
            except Exception:
                auc_val = None

    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rep_df = (pd.DataFrame(rep).T
              .rename(columns={"precision": "Precision", "recall": "Recall",
                               "f1-score": "F1", "support": "Support"}))

    # Coerce numerics & round metric columns
    for c in ["Precision", "Recall", "F1", "Support"]:
        if c in rep_df.columns:
            rep_df[c] = pd.to_numeric(rep_df[c], errors="coerce")

    for c in ["Precision", "Recall", "F1"]:
        if c in rep_df.columns:
            rep_df[c] = rep_df[c].round(3)

    # Safely cast Support to nullable Int64 only if integer-like
    if "Support" in rep_df.columns:
        supp = rep_df["Support"]
        # integer-like if fractional part is ~0 (allowing for float noise)
        integer_like = supp.dropna().map(lambda v: np.isclose(v % 1, 0.0)).all()
        if integer_like:
            rep_df["Support"] = supp.round().astype("Int64")  # nullable int
        else:
            rep_df["Support"] = supp  # leave as float to avoid casting error

    return acc, prec, rec, f1, auc_val, auc_note, rep_df


# ------------------------- Fast Gain Ratio Decision Tree -------------------------
class GainRatioDecisionTree:
    """
    Fast Gain Ratio (C4.5-style) decision tree.

    Key speed-ups:
      • Global quantile binning for numeric features (computed once at fit).
      • Vectorized split scoring with class-by-bin contingency matrices.
      • Feature subsampling per node (max_features), early stop via min_gain_ratio.
      • Caps high-cardinality categoricals per node (max_categories).

    Parameters
    ----------
    max_depth : int
    min_samples_split : int
    min_samples_leaf : int
    categorical_features : list[int]            # column indices that are categorical (already label-encoded ints)
    random_state : int | None
    n_bins : int                                 # numeric histogram bins (16–64 is typical)
    bin_strategy : {'quantile','uniform'}        # how to build global bin edges for numeric features
    max_features : {'sqrt','log2',int,float,None}
    min_gain_ratio : float                       # early stop if best GR < this
    max_categories : int                         # cap categories per node (group rest into 'other')
    """

    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 categorical_features=None, random_state=None,
                 n_bins=32, bin_strategy="quantile",
                 max_features="sqrt", min_gain_ratio=1e-3, max_categories=32):
        # store EXACTLY as passed (clone-safe)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.max_features = max_features
        self.min_gain_ratio = min_gain_ratio
        self.max_categories = max_categories

        # fitted attrs
        self.n_features_in_ = None
        self.classes_ = None
        self._tree_ = None
        self._cat_set_ = None
        self._rng_ = None
        self._bin_edges_ = None  # dict: j -> edges (len = n_bins-1)

    # ---------- information measures ----------
    @staticmethod
    def _entropy_from_counts(counts_2d):
        # counts_2d: (M, C)
        totals = counts_2d.sum(axis=1, keepdims=True).astype(float)
        totals = np.clip(totals, 1.0, None)
        p = counts_2d / totals
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
        return H  # (M,)

    @staticmethod
    def _split_info_from_sizes(sizes):
        total = np.sum(sizes, axis=-1, keepdims=True).astype(float)
        total = np.clip(total, 1.0, None)
        p = sizes / total
        with np.errstate(divide="ignore", invalid="ignore"):
            si = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=-1)
        return si

    # ---------- feature subsampling ----------
    def _feature_indices(self):
        p = self.n_features_in_
        mf = self.max_features
        if mf in (None, "all"):
            k = p
        elif mf == "sqrt":
            k = max(1, int(np.sqrt(p)))
        elif mf == "log2":
            k = max(1, int(np.log2(p)))
        elif isinstance(mf, float) and 0 < mf <= 1:
            k = max(1, int(np.ceil(mf * p)))
        elif isinstance(mf, int) and 1 <= mf <= p:
            k = mf
        else:
            k = p
        if k == p:
            return np.arange(p, dtype=int)
        return self._rng_.choice(p, size=k, replace=False)

    # ---------- numeric split via global histogram ----------
    def _best_split_numeric(self, j, x, y_idx, parent_H):
        # bin with global edges (computed on full column once)
        edges = self._bin_edges_[j]
        if edges.size == 0:
            return None
        bins = np.digitize(x, edges, right=True)  # 0..B
        B = edges.size + 1
        C = self.classes_.size

        # contingency over (B bins, C classes)
        M = np.zeros((B, C), dtype=np.int32)
        np.add.at(M, (bins, y_idx), 1)

        # cumulative counts across bins -> evaluate split after bin 0..B-2
        cum = np.cumsum(M, axis=0)            # (B, C)
        left_counts = cum[:-1]                # (B-1, C)
        right_counts = M.sum(axis=0, keepdims=True) - left_counts  # (B-1, C)

        left_n = left_counts.sum(axis=1)
        right_n = right_counts.sum(axis=1)
        valid = (left_n >= self.min_samples_leaf) & (right_n >= self.min_samples_leaf)
        if not np.any(valid):
            return None
        left_counts = left_counts[valid]
        right_counts = right_counts[valid]
        split_ids = np.nonzero(valid)[0]      # which boundary

        # entropies & GR
        H_left = self._entropy_from_counts(left_counts)
        H_right = self._entropy_from_counts(right_counts)
        n_tot = (left_n[valid] + right_n[valid]).astype(float)
        H_children = (left_n[valid] * H_left + right_n[valid] * H_right) / n_tot
        ig = parent_H - H_children
        si = self._split_info_from_sizes(np.stack([left_n[valid], right_n[valid]], axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            gr = np.where(si > 0, ig / si, -np.inf)

        best_idx = int(np.argmax(gr))
        best_gr = float(gr[best_idx])
        if not np.isfinite(best_gr) or best_gr < self.min_gain_ratio:
            return None

        # threshold is right edge of the left bin
        boundary = split_ids[best_idx]
        thr = edges[boundary]  # edge at boundary index
        return {"type": "numeric", "threshold": float(thr), "gain_ratio": best_gr}
    
    def score(self, X, y):
        """Sklearn-compatible: return accuracy on (X, y)."""
        y_pred = self.predict(X)
        return float(accuracy_score(y, y_pred))

    # ---------- categorical split (vectorized) ----------
    def _best_split_categorical(self, x, y_idx, parent_H):
        cats, inv = np.unique(x, return_inverse=True)
        K = cats.size
        if K <= 1:
            return None

        C = self.classes_.size

        # Cap categories to top-k and reindex to 0..(max_categories-1)
        if K > self.max_categories:
            counts = np.bincount(inv, minlength=K)
            topk = np.argpartition(counts, -self.max_categories)[-self.max_categories:]  # indices in 0..K-1

            # Build remap: kept -> 0..(max_categories-1), others -> max_categories (the "other" bucket)
            remap = np.full(K, self.max_categories, dtype=int)
            remap[topk] = np.arange(self.max_categories, dtype=int)

            inv = remap[inv]               # now inv ∈ {0..max_categories} (other = max_categories)
            cats = cats[topk]              # only kept raw values are explicitly named
            K = self.max_categories + 1    # include "other" bucket at index = max_categories

        M = np.zeros((K, C), dtype=np.int32)
        np.add.at(M, (inv, y_idx), 1)

        sizes = M.sum(axis=1)
        if np.any(sizes < self.min_samples_leaf):
            return None

        H_children = self._entropy_from_counts(M)
        w = sizes.astype(float) / np.clip(sizes.sum(), 1.0, None)
        Hw = np.sum(w * H_children)

        ig = parent_H - Hw
        si = self._split_info_from_sizes(sizes.astype(float))
        gr = ig / si if si > 0 else -np.inf
        if not np.isfinite(gr) or gr < self.min_gain_ratio:
            return None

        # cat_map encodes mapping for kept raw values; 'other' is implicit at index len(cats)
        return {
            "type": "categorical",
            "categories": np.arange(K, dtype=int),   # 0..(len(cats)) where last is 'other' if K>len(cats)
            "gain_ratio": float(gr),
            "cat_map": cats,                         # length = min(original K, max_categories)
        }


    # ---------- choose best split among a subsample of features ----------
    def _choose_best_split(self, X, idx, y_idx):
        C = self.classes_.size
        parent_counts = np.bincount(y_idx, minlength=C).reshape(1, C)
        parent_H = float(self._entropy_from_counts(parent_counts)[0])

        best = None
        for j in self._feature_indices():
            col = X[idx, j]
            if j in self._cat_set_:
                cand = self._best_split_categorical(col, y_idx, parent_H)
            else:
                cand = self._best_split_numeric(j, col.astype(float), y_idx, parent_H)
            if cand is None:
                continue
            if (best is None) or (cand["gain_ratio"] > best["gain_ratio"]):
                best = {"feature": j, **cand}
        return best

    # ---------- recursion using index views ----------
    def _leaf_stats(self, y_idx):
        C = self.classes_.size
        counts = np.bincount(y_idx, minlength=C).astype(float)
        total = counts.sum()
        if total <= 0:
            return int(0), (np.ones(C) / C).tolist()
        proba = (counts / total).tolist()
        majority = int(np.argmax(counts))
        return majority, proba

    def _build(self, X, idx, y_idx, depth):
        node = {}
        maj, proba = self._leaf_stats(y_idx)
        node["prediction"] = maj
        node["proba"] = proba
        node["is_leaf"] = True

        if (depth >= self.max_depth or
            np.unique(y_idx).size <= 1 or
            idx.size < self.min_samples_split):
            return node

        split = self._choose_best_split(X, idx, y_idx)
        if split is None:
            return node

        node["is_leaf"] = False
        node["feature"] = split["feature"]
        node["split"] = {"type": split["type"], "gain_ratio": split["gain_ratio"]}

        j = split["feature"]
        if split["type"] == "numeric":
            thr = split["threshold"]
            node["split"]["threshold"] = thr
            left_mask = X[idx, j] <= thr
            right_mask = ~left_mask
            left_idx = idx[left_mask]; right_idx = idx[right_mask]
            if left_idx.size < self.min_samples_leaf or right_idx.size < self.min_samples_leaf:
                node["is_leaf"] = True
                return node
            node["left"] = self._build(X, left_idx, y_idx[left_mask], depth + 1)
            node["right"] = self._build(X, right_idx, y_idx[right_mask], depth + 1)
        else:
            cats = split["cat_map"]
            node["split"]["categories"] = split["categories"]
            node["split"]["cat_map"] = cats
            node["children"] = {}

            xj = X[idx, j]
            val_to_local = {int(v): i for i, v in enumerate(cats)}
            other_id = len(cats)
            local_ids = np.array([val_to_local.get(int(v), other_id) for v in xj], dtype=int)

            for local in np.unique(local_ids):
                mask = (local_ids == local)
                child_idx = idx[mask]
                child_y = y_idx[mask]
                if child_idx.size < self.min_samples_leaf:
                    continue
                node["children"][int(local)] = self._build(X, child_idx, child_y, depth + 1)

            # If all children were pruned, keep this node as a leaf
            if len(node["children"]) == 0:
                node["is_leaf"] = True
                return node
        return node


    # ---------- public API ----------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._cat_set_ = set(self.categorical_features or [])
        self._rng_ = np.random.RandomState(self.random_state)

        # Build global bin edges for numeric features once
        self._bin_edges_ = {}
        numeric_idx = [j for j in range(self.n_features_in_) if j not in self._cat_set_]
        for j in numeric_idx:
            col = X[:, j].astype(float)
            if self.bin_strategy == "uniform":
                lo, hi = np.nanmin(col), np.nanmax(col)
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    edges = np.array([], dtype=float)
                else:
                    edges = np.linspace(lo, hi, num=self.n_bins + 1)[1:-1]
            else:  # quantile
                uq = np.unique(col)
                if uq.size <= 1:
                    edges = np.array([], dtype=float)
                else:
                    qs = np.linspace(0, 1, num=self.n_bins + 1, endpoint=True)[1:-1]
                    edges = np.quantile(col, qs, method="linear")
                    edges = np.unique(edges)
            self._bin_edges_[j] = edges.astype(float)

        # y -> compact 0..C-1
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.vectorize(class_to_idx.get, otypes=[int])(y)

        idx = np.arange(X.shape[0], dtype=int)
        self._tree_ = self._build(X, idx, y_idx, depth=0)
        return self

    def _predict_one(self, row):
        # If somehow predict is called before fit, return a safe default
        if self._tree_ is None or self.classes_ is None:
            C = int(self.classes_.size) if isinstance(self.classes_, np.ndarray) else 2
            return 0, np.ones(C, dtype=float) / C

        node = self._tree_

        # Initialize fallback from whatever the current node carries (or safe defaults)
        pred = node.get("prediction", 0)
        proba = np.array(node.get("proba", np.ones(self.classes_.size) / self.classes_.size), dtype=float)

        while node is not None and not node.get("is_leaf", True):
            j = node["feature"]
            split = node["split"]

            if split["type"] == "numeric":
                thr = split["threshold"]
                next_node = node["left"] if row[j] <= thr else node["right"]
            else:
                cats = split["cat_map"]
                val_to_local = {int(v): i for i, v in enumerate(cats)}
                other_id = len(cats)  # aggregated "other" bucket
                local = val_to_local.get(int(row[j]), other_id)
                next_node = node["children"].get(local)

            # If child is missing (e.g., pruned), stop and use the best known fallback
            if next_node is None:
                break

            node = next_node
            pred = node.get("prediction", pred)
            proba = np.array(node.get("proba", proba), dtype=float)

        return pred, proba


    def predict(self, X):
        X = np.asarray(X)
        preds = [self._predict_one(r)[0] for r in X]
        return self.classes_[np.array(preds, dtype=int)]

    def predict_proba(self, X):
        X = np.asarray(X)
        P = [self._predict_one(r)[1] for r in X]
        return np.vstack(P)

    # sklearn compatibility
    def get_params(self, deep=True):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "categorical_features": self.categorical_features,
            "random_state": self.random_state,
            "n_bins": self.n_bins,
            "bin_strategy": self.bin_strategy,
            "max_features": self.max_features,
            "min_gain_ratio": self.min_gain_ratio,
            "max_categories": self.max_categories,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
    
# ------------------------- Data ingestion -------------------------
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
use_session_df = st.checkbox("Use DataFrame from Visualización page", value=True)

if use_session_df and "df" in st.session_state:
    df = st.session_state["df"].copy()
elif uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV or use the DataFrame from the other page.")
    st.stop()

st.caption(f"Rows: {len(df):,}, Columns: {df.shape[1]}")

# Target selection
default_target = "income" if "income" in df.columns else None
target_col = st.selectbox("Select target column", options=df.columns,
                          index=(list(df.columns).index(default_target) if default_target else 0))
feature_cols = [c for c in df.columns if c != target_col]

# Encode features: numeric pass-through, others label-encoded
encoders, numeric_mask = {}, {}
df_enc = pd.DataFrame(index=df.index)
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]):
        df_enc[c] = df[c]
        numeric_mask[c] = True
    else:
        le = LabelEncoder()
        df_enc[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le
        numeric_mask[c] = False

# Target encoding
y_le = LabelEncoder()
y = y_le.fit_transform(df[target_col].astype(str))
X = df_enc.values
feature_names = df_enc.columns.tolist()
categorical_idx = [i for i, c in enumerate(feature_names) if not numeric_mask[c]]

# ------------------------- Controls -------------------------
colA, colB, colC = st.columns(3)
with colA:
    classifier = st.radio("Classifier", ["Decision Tree", "Random Forest", "Gradient Boosting", "Bagging"], horizontal=True)
with colB:
    gain_criterion = st.radio("Criterion", ["gini", "entropy", "gain_ratio (C4.5)"], horizontal=True)
with colC:
    depth = st.slider("Max depth", 1, 20, 5)

if classifier == "Decision Tree":
    kfold = st.slider("K-Fold (CV para Decision Tree)", 2, 10, 5)

n_estimators = st.slider("Number of estimators (RF/GB/Bagging)", 10, 100, 30, step=10)

# ------------------------- Extra / advanced hyperparameters -------------------------
with st.expander("⚙️ Advanced hyperparameters"):
    if classifier == "Decision Tree" and gain_criterion.startswith("gain_ratio"):
        # Gain Ratio (our custom tree)
        gr_min_samples_split = st.number_input("min_samples_split (GR)", min_value=2, value=20, step=1)
        gr_min_samples_leaf  = st.number_input("min_samples_leaf (GR)",  min_value=1, value=10, step=1)
        gr_n_bins            = st.slider("n_bins (numeric features, GR)", 8, 128, 32, step=4)
        gr_bin_strategy      = st.selectbox("bin_strategy (GR)", ["quantile", "uniform"], index=0)
        gr_max_features_opt  = st.selectbox("max_features (GR)", ["sqrt", "log2", "all", "None", 0.25, 0.5, 0.75], index=0,
                                            help="'all' uses all features; floats are fractions")
        gr_min_gain_ratio    = st.number_input("min_gain_ratio (GR)", min_value=0.0, value=1e-3, step=1e-3, format="%.5f")
        gr_max_categories    = st.slider("max_categories (GR)", 4, 128, 24, step=4)
        gr_random_state      = st.number_input("random_state (GR)", value=42, step=1)

    elif classifier == "Decision Tree":
        dt_splitter          = st.selectbox("splitter", ["best", "random"], index=0)
        dt_min_samples_split = st.number_input("min_samples_split", min_value=2, value=2, step=1)
        dt_min_samples_leaf  = st.number_input("min_samples_leaf",  min_value=1, value=1, step=1)
        dt_max_features_opt  = st.selectbox("max_features", ["sqrt", "log2", "None"], index=2)
        dt_ccp_alpha         = st.number_input("ccp_alpha", min_value=0.0, value=0.0, step=0.001, format="%.3f")

    elif classifier == "Random Forest":
        rf_max_features_opt  = st.selectbox("max_features", ["sqrt", "log2", "None", 0.25, 0.5, 0.75], index=0)
        rf_min_samples_split = st.number_input("min_samples_split", min_value=2, value=2, step=1)
        rf_min_samples_leaf  = st.number_input("min_samples_leaf",  min_value=1, value=1, step=1)
        rf_bootstrap         = st.checkbox("bootstrap", value=True)
        rf_oob               = st.checkbox("oob_score", value=False, disabled=not rf_bootstrap)
        rf_n_jobs            = st.number_input("n_jobs", value=-1, step=1)
        rf_class_weight      = st.selectbox("class_weight", ["None", "balanced", "balanced_subsample"], index=0)

    elif classifier == "Gradient Boosting":
        gb_learning_rate     = st.number_input("learning_rate", min_value=1e-4, value=0.1, step=0.01, format="%.4f")
        gb_subsample         = st.number_input("subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
        gb_max_depth         = st.slider("max_depth (trees)", 1, 10, depth, step=1)
        gb_max_features_opt  = st.selectbox("max_features", ["None", "sqrt", "log2"], index=0)
        gb_validation_fraction = st.number_input("validation_fraction (early stopping)", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
        gb_n_iter_no_change  = st.number_input("n_iter_no_change", min_value=0, value=0, step=1)

    elif classifier == "Bagging":
        bg_max_samples       = st.number_input("max_samples (fraction)", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
        bg_max_features      = st.number_input("max_features (fraction)", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
        bg_bootstrap         = st.checkbox("bootstrap", value=True)
        bg_bootstrap_features= st.checkbox("bootstrap_features", value=False)
        bg_oob               = st.checkbox("oob_score", value=False, disabled=not bg_bootstrap)
        bg_n_jobs            = st.number_input("n_jobs", value=-1, step=1)


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------- ES descriptions -------------------------
with st.expander("📝 Descripción breve del método (ES)"):
    if classifier == "Random Forest":
        st.markdown(
            "- **Random Forest**: Ensamble de árboles entrenados sobre **submuestras de datos** y **subconjuntos aleatorios de características** en cada división. "
            "Reduce la varianza y mejora la generalización; la predicción es por voto mayoritario (clasificación) o promedio (regresión)."
        )
    elif classifier == "Bagging":
        st.markdown(
            "- **Bagging (Bootstrap Aggregating)**: Entrena múltiples modelos base (p. ej., árboles) sobre **muestras bootstrap** del conjunto de entrenamiento. "
            "Promedia/combina sus predicciones para **reducir la varianza** y estabilizar el rendimiento."
        )
    elif classifier == "Gradient Boosting":
        st.markdown(
            "- **Gradient Boosting**: Construye un ensamble **secuencial** donde cada nuevo árbol corrige los **errores (residuos)** del ensamble previo, "
            "optimizando una función de pérdida mediante **gradiente**. Suele requerir hiperparámetros de regularización cuidadosos."
        )
    else:
        st.markdown(
            "- **Decision Tree**: Modelo interpretable que particiona el espacio de características mediante reglas si/entonces. "
            "Aquí permitimos **Gini**, **Entropía (Information Gain)** y **Gain Ratio (C4.5)**."
        )

# ------------------------- Evaluation -------------------------
def show_eval(clf, name: str):
    st.markdown(f"### {name}")

    y_pred = clf.predict(X_test)

    y_score = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)
        y_score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
    elif hasattr(clf, "decision_function"):
        s = clf.decision_function(X_test)
        y_score = s

    acc, prec, rec, f1, auc_val, auc_note, rep_df = metrics_block(y_test, y_pred, y_score=y_score, classes_=getattr(clf, "classes_", None))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision (weighted)", f"{prec:.3f}")
    col3.metric("Recall (weighted)", f"{rec:.3f}")
    col4.metric("F1 (weighted)", f"{f1:.3f}")
    col5.metric(f"AUC{auc_note}", f"{auc_val:.3f}" if auc_val is not None else "—")

    cm = confusion_matrix(y_test, y_pred, labels=getattr(clf, "classes_", None))
    tick_labels = y_le.inverse_transform(getattr(clf, "classes_", np.unique(y_test)))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax,
                xticklabels=tick_labels, yticklabels=tick_labels)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    plt.xticks(rotation=45); plt.yticks(rotation=0)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.dataframe(
        rep_df,
        use_container_width=True,
        column_config={
            "Precision": st.column_config.ProgressColumn("Precision", help="TP / (TP + FP)", min_value=0.0, max_value=1.0, format="%.3f"),
            "Recall":    st.column_config.ProgressColumn("Recall",    help="TP / (TP + FN)", min_value=0.0, max_value=1.0, format="%.3f"),
            "F1":        st.column_config.ProgressColumn("F1",        help="2·(P·R)/(P+R)",  min_value=0.0, max_value=1.0, format="%.3f"),
            "Support":   st.column_config.NumberColumn  ("Support",   help="Muestras por clase", format="%d"),
        },
    )

    if hasattr(clf, "classes_") and len(clf.classes_) == 2 and y_score is not None and y_score.ndim == 1:
        plot_roc_streamlit(clf, X_test, y_test, title=f"{name} — ROC Curve")

    with st.expander("🔧 Hiperparámetros del modelo"):
        try:
            st.json(clf.get_params())
        except Exception:
            st.write(clf.__dict__)

# ------------------------- Train -------------------------

gr_params = dict(
    max_depth=5,                 
    min_samples_split=20,
    min_samples_leaf=10,
    categorical_features=categorical_idx,
    random_state=42,
    n_bins=32,                  
    bin_strategy="quantile",     
    max_features="sqrt",         
    min_gain_ratio=1e-3,         
    max_categories=24,           
)

def gr_factory():
    return GainRatioDecisionTree(**gr_params)

if st.button("Train", type="primary"):
    if classifier == "Random Forest":
        if gain_criterion.startswith("gain_ratio"):
            st.warning("Random Forest no soporta Gain Ratio en scikit-learn; usando 'entropy'.")
            rf_criterion = "entropy"
        else:
            rf_criterion = gain_criterion
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=depth, criterion=rf_criterion, random_state=42
        ).fit(X_train, y_train)
        show_eval(clf, "Random Forest")

    elif classifier == "Decision Tree":
        if gain_criterion.startswith("gain_ratio"):
            folds = min(kfold, 3)
            mean, std = stratified_cv_scores(gr_factory, X_train, y_train, n_splits=folds)
            st.write(f"CV accuracy ({folds}-fold): mean={mean:.3f} ± {std:.3f}")
            clf = gr_factory().fit(X_train, y_train)
            show_eval(clf, "Decision Tree (Gain Ratio)")
        else:
            clf = DecisionTreeClassifier(max_depth=depth, criterion=gain_criterion, random_state=42)
            scores = cross_val_score(clf, X_train, y_train, cv=kfold, scoring="accuracy")
            st.write(f"CV accuracy ({kfold}-fold): mean={scores.mean():.3f} ± {scores.std():.3f}")
            clf.fit(X_train, y_train)
            show_eval(clf, "Decision Tree")

    elif classifier == "Gradient Boosting":
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=depth, random_state=42
        ).fit(X_train, y_train)
        show_eval(clf, "Gradient Boosting")

    elif classifier == "Bagging":
        if gain_criterion.startswith("gain_ratio"):
            base = gr_factory()  
        else:
            base = DecisionTreeClassifier(max_depth=depth, criterion=gain_criterion, random_state=42)

        clf = BaggingClassifier(estimator=base, n_estimators=n_estimators, random_state=42).fit(X_train, y_train)
        show_eval(clf, "Bagging")

