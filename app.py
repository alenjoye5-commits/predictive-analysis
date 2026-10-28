"""Regional Sales ML Studio - Streamlit app
Data analysis | Visualization | Predictive analysis | Forecasting
Run:  streamlit run app.py
"""
import hashlib, os, tempfile, warnings
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Regional Sales ML Studio", page_icon="📈", layout="wide")

DEFAULT_FILE = "Denormalize_regional_sales_data.xlsx"
REQUIRED = ["Date", "Units Sold", "Sales", "Cost of sales", "Product Name", "Category", "Brand",
            "Cities", "SalesPerson Name", "Scenario Name"]
DIMS = ["Category", "Brand", "Product Name", "Cities", "SalesPerson Name", "Designation", "Warranty", "Scenario Name"]
METRICS = ["Sales", "Profit", "Units Sold"]


# ----------------------------------------------------------------------------- helpers
def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def plot(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def fmt(x):
    for u, d in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= d:
            return f"{x / d:,.2f}{u}"
    return f"{x:,.0f}"


def download(df, label, name):
    st.download_button(label, df.to_csv(index=False).encode(), name, "text/csv")


# ----------------------------------------------------------------------------- data loading
@st.cache_data(show_spinner="Loading data (first load of a large Excel file can take a minute)...")
def load_data(name: str, data: bytes) -> pd.DataFrame:
    cache = os.path.join(tempfile.gettempdir(), f"sales_{hashlib.md5(data).hexdigest()}.pkl")
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
    else:
        if name.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(data))
        else:  # the sheet may start with blank rows -> detect the header row
            peek = pd.read_excel(BytesIO(data), header=None, nrows=15)
            hdr = next((i for i, r in peek.iterrows()
                        if {"date", "sales"} <= {str(v).strip().lower() for v in r.values}), 0)
            df = pd.read_excel(BytesIO(data), header=hdr)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")].dropna(how="all")
        df.to_pickle(cache)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if "DOJ" in df:
        df["DOJ"] = pd.to_datetime(df["DOJ"])
    df["Profit"] = df["Sales"] - df["Cost of sales"]
    df["Margin %"] = np.where(df["Sales"] > 0, df["Profit"] / df["Sales"] * 100, 0)
    df["Year"], df["Month"] = df["Date"].dt.year, df["Date"].dt.month
    df["Quarter"], df["DayOfWeek"] = df["Date"].dt.quarter, df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["YearMonth"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    return df.sort_values("Date").reset_index(drop=True)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("📈 Sales ML Studio")
up = st.sidebar.file_uploader("Upload dataset (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
if up is not None:
    raw_name, raw = up.name, up.getvalue()
else:
    here = os.path.dirname(os.path.abspath(__file__))
    path = next((p for p in (os.path.join(here, DEFAULT_FILE), DEFAULT_FILE) if os.path.exists(p)), None)
    if path is None:
        st.info(f"Upload your dataset in the sidebar, or place **{DEFAULT_FILE}** next to app.py.")
        st.stop()
    raw_name, raw = path, open(path, "rb").read()
try:
    DF = load_data(raw_name, raw)
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()

st.sidebar.subheader("Filters (Analysis & Visualization)")
d0, d1 = DF["Date"].min().date(), DF["Date"].max().date()
dr = st.sidebar.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
f = DF
if isinstance(dr, (tuple, list)) and len(dr) == 2:
    f = f[(f["Date"] >= pd.Timestamp(dr[0])) & (f["Date"] <= pd.Timestamp(dr[1]))]
for col, lab in [("Scenario Name", "Scenario"), ("Cities", "City"), ("Category", "Category"), ("Brand", "Brand")]:
    sel = st.sidebar.multiselect(lab, sorted(DF[col].unique()))
    if sel:
        f = f[f[col].isin(sel)]
st.sidebar.caption(f"{len(f):,} of {len(DF):,} rows selected. Predictive & Forecasting tabs have their own controls.")
if f.empty:
    st.warning("No rows match the current filters.")
    st.stop()

st.title("Regional Sales ML Studio")
t1, t2, t3, t4 = st.tabs(["📊 Data Analysis", "📈 Visualization", "🤖 Predictive Analysis", "🔮 Forecasting"])

# ============================================================================= 1. DATA ANALYSIS
with t1:
    c = st.columns(5)
    c[0].metric("Total Sales", fmt(f["Sales"].sum()))
    c[1].metric("Total Profit", fmt(f["Profit"].sum()))
    c[2].metric("Profit Margin", f"{f['Profit'].sum() / f['Sales'].sum() * 100:.1f}%")
    c[3].metric("Units Sold", fmt(f["Units Sold"].sum()))
    c[4].metric("Avg Sale / Order", fmt(f["Sales"].mean()))

    st.subheader("Dataset overview")
    a, b = st.columns([2, 3])
    with a:
        prof = pd.DataFrame({"dtype": f.dtypes.astype(str), "nulls": f.isna().sum(), "unique": f.nunique()})
        show_df(prof)
    with b:
        n = st.slider("Preview rows", 5, 100, 10)
        show_df(f.head(n))
    st.write("**Summary statistics**")
    show_df(f[["Units Sold", "Sales", "Cost of sales", "Profit", "Margin %"]].describe().T)

    st.subheader("Data quality")
    q = st.columns(4)
    q[0].metric("Missing cells", int(DF.isna().sum().sum()))
    q[1].metric("Duplicate rows", int(DF.duplicated().sum()))
    if "Sale price" in DF:
        q[2].metric("Sales ≠ Units×Price", int((DF["Sales"] != DF["Units Sold"] * DF["Sale price"]).sum()))
    num = DF[["Units Sold", "Sales", "Cost of sales", "Profit"]]
    q1, q3 = num.quantile(.25), num.quantile(.75)
    iqr = q3 - q1
    out = ((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).sum().rename("IQR outliers")
    q[3].metric("Sales outliers (IQR)", int(out["Sales"]))

    st.subheader("Correlation")
    ncols = [c for c in ["Units Sold", "Sales", "Cost of sales", "Profit", "Sale price", "Cost price", "Age", "Year", "Month"]
             if c in f.columns]
    plot(px.imshow(f[ncols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                   title="Correlation matrix"))

    st.subheader("Group-by explorer")
    g1, g2, g3 = st.columns(3)
    by = g1.multiselect("Group by", [d for d in DIMS if d in f.columns] + ["Year", "Quarter", "Month"], default=["Category"])
    mets = g2.multiselect("Metrics", ["Sales", "Profit", "Units Sold", "Cost of sales"], default=["Sales", "Profit"])
    agg = g3.selectbox("Aggregation", ["sum", "mean", "median", "count", "max", "min"])
    if by and mets:
        gb = f.groupby(by)[mets].agg(agg).reset_index().sort_values(mets[0], ascending=False)
        show_df(gb)
        download(gb, "⬇ Download table", "groupby.csv")

    st.subheader("Actual vs Budget")
    if f["Scenario Name"].nunique() > 1:
        dim = st.selectbox("Compare by", [d for d in ["Category", "Brand", "Cities", "SalesPerson Name", "Year"] if d in f.columns])
        pv = f.pivot_table(index=dim, columns="Scenario Name", values="Sales", aggfunc="sum")
        if {"Actual", "Budget"} <= set(pv.columns):
            pv["Variance"] = pv["Actual"] - pv["Budget"]
            pv["Variance %"] = pv["Variance"] / pv["Budget"] * 100
            show_df(pv.reset_index().style.format({c: "{:,.0f}" for c in ["Actual", "Budget", "Variance"]} | {"Variance %": "{:+.1f}%"}))
            plot(px.bar(pv.reset_index(), x=dim, y=["Actual", "Budget"], barmode="group", title=f"Actual vs Budget by {dim}"))
    else:
        st.info("Select both scenarios (or clear the scenario filter) to compare Actual vs Budget.")

    st.subheader("Year-over-year growth")
    yy = f.groupby("Year")[["Sales", "Profit", "Units Sold"]].sum()
    yy["Sales growth %"] = yy["Sales"].pct_change() * 100
    show_df(yy.reset_index().style.format("{:,.1f}", subset=["Sales growth %"]))

# ============================================================================= 2. VISUALIZATION
with t2:
    m = st.radio("Metric", METRICS, horizontal=True, key="vm")
    c1, c2 = st.columns([3, 1])
    fr = c2.selectbox("Trend frequency", ["Monthly", "Quarterly", "Yearly", "Weekly"])
    roll = c2.checkbox("3-period rolling mean", False)
    rule = {"Monthly": "MS", "Quarterly": "QS", "Yearly": "YS", "Weekly": "W"}[fr]
    tr = f.set_index("Date").groupby("Scenario Name")[m].resample(rule).sum().reset_index()
    if roll:
        tr[m] = tr.groupby("Scenario Name")[m].transform(lambda s: s.rolling(3, min_periods=1).mean())
    with c1:
        plot(px.line(tr, x="Date", y=m, color="Scenario Name", title=f"{m} trend ({fr.lower()})"))

    a, b = st.columns(2)
    with a:
        d = f.groupby("Category")[m].sum().sort_values().reset_index()
        plot(px.bar(d, x=m, y="Category", orientation="h", title=f"{m} by category"))
    with b:
        d = f.groupby("Cities")[m].sum().reset_index()
        plot(px.pie(d, names="Cities", values=m, hole=.45, title=f"{m} share by city"))

    a, b = st.columns(2)
    with a:
        plot(px.sunburst(f.groupby(["Category", "Brand"])[m].sum().reset_index(), path=["Category", "Brand"], values=m,
                         title="Category → Brand"))
    with b:
        top = f.groupby("Product Name")[m].sum().nlargest(10).sort_values().reset_index()
        plot(px.bar(top, x=m, y="Product Name", orientation="h", title="Top 10 products"))

    a, b = st.columns(2)
    with a:
        hm = f.pivot_table(index="Year", columns="Month", values=m, aggfunc="sum")
        plot(px.imshow(hm, aspect="auto", color_continuous_scale="Viridis", title=f"{m}: Year × Month heatmap"))
    with b:
        hm = f.pivot_table(index="Category", columns="Cities", values=m, aggfunc="sum")
        plot(px.imshow(hm, aspect="auto", text_auto=".2s", color_continuous_scale="Blues", title="Category × City"))

    if {"Latitude", "Longitude"} <= set(f.columns):
        geo = f.groupby(["Cities", "Latitude", "Longitude"])[m].sum().reset_index()
        mapfn = getattr(px, "scatter_map", None) or px.scatter_mapbox
        fig = mapfn(geo, lat="Latitude", lon="Longitude", size=m, color=m, hover_name="Cities", zoom=4.3,
                    size_max=45, color_continuous_scale="Turbo", title=f"{m} by city")
        fig.update_layout(map_style="open-street-map") if hasattr(px, "scatter_map") else fig.update_layout(mapbox_style="open-street-map")
        plot(fig)

    a, b = st.columns(2)
    with a:
        sp = f.groupby(["SalesPerson Name"] + (["Designation"] if "Designation" in f else []))[m].sum().reset_index()
        plot(px.bar(sp.sort_values(m), x=m, y="SalesPerson Name", orientation="h",
                    color="Designation" if "Designation" in sp else None, title="Sales-rep performance"))
    with b:
        plot(px.box(f, x="Month", y=m, title=f"Seasonality: {m} per transaction by month"))

    a, b = st.columns(2)
    with a:
        plot(px.histogram(f, x="Sales", nbins=60, log_y=True, title="Distribution of sale amounts (log y)"))
    with b:
        plot(px.box(f, x="Category", y="Margin %", color="Category", title="Margin % by category"))

    st.subheader("Custom chart builder")
    k = st.columns(5)
    kind = k[0].selectbox("Chart", ["bar", "line", "scatter", "box"])
    x = k[1].selectbox("X", ["Year", "Month", "YearMonth"] + DIMS)
    y = k[2].selectbox("Y", ["Sales", "Profit", "Units Sold", "Cost of sales"])
    colr = k[3].selectbox("Color", [None] + DIMS)
    ag = k[4].selectbox("Aggregate", ["sum", "mean", "count"])
    if kind == "box":
        plot(px.box(f, x=x, y=y, color=colr))
    else:
        d = f.groupby([x] + ([colr] if colr and colr != x else []))[y].agg(ag).reset_index()
        plot(getattr(px, kind)(d, x=x, y=y, color=colr if colr else None))


# ============================================================================= 3. PREDICTIVE ANALYSIS
class RelModel(BaseEstimator, RegressorMixin):
    """Pipeline wrapper: optionally predicts target / base (relative mode) and rescales to original units."""
    def __init__(self, pipe=None, rel=False, base_col="_base"):
        self.pipe, self.rel, self.base_col = pipe, rel, base_col

    def fit(self, X, y):
        self.pipe.fit(X, y / X[self.base_col].values if self.rel else y)
        return self

    def predict(self, X):
        p = self.pipe.predict(X)
        return p * X[self.base_col].values if self.rel else p


CATS = ["Product Name", "Category", "Brand", "Warranty", "Cities", "SalesPerson Name", "Scenario Name"]
NUMS = ["Units Sold", "Year", "Month", "Quarter", "DayOfWeek", "DayOfYear", "TimeIdx"]


def add_features(df, refs, target, t0):
    X = df.copy()
    for c in CATS:
        if c not in X:
            X[c] = "NA"
    X["TimeIdx"] = (X["Date"] - t0).dt.days
    if refs is not None:
        key = "price" if target == "Sales" else "margin"
        X["RefUnit"] = X["Product Name"].map(refs[key]).fillna(refs[key].mean())
        X["_base"] = (X["Units Sold"] * X["RefUnit"]).clip(lower=1)
    else:
        X["_base"] = 1.0
    return X


def make_model(name, use_ref):
    cats = [c for c in CATS if c != "Brand" or True]
    nums = NUMS + (["RefUnit", "_base"] if use_ref else [])
    if name == "Ridge Regression":
        pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cats), ("n", StandardScaler(), nums)])
        est = Ridge(alpha=1.0)
    else:
        pre = ColumnTransformer([("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
                                 ("n", "passthrough", nums)])
        est = (RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=42)
               if name == "Random Forest" else
               HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=8, random_state=42))
    return Pipeline([("pre", pre), ("est", est)]), cats + nums


with t3:
    st.subheader("Train a model to predict transaction-level sales")
    st.caption("Uses all data (both scenarios). Features: product, city, sales rep, scenario, units, and date parts. "
               "Time-ordered split: the model is trained on the past and tested on the most recent period, like real forecasting.")
    k = st.columns(5)
    target = k[0].selectbox("Target", ["Sales", "Profit"])
    mname = k[1].selectbox("Model", ["HistGradientBoosting", "Random Forest", "Ridge Regression"])
    test_frac = k[2].slider("Test share (latest data)", 0.1, 0.4, 0.2, 0.05)
    cap = k[3].slider("Max training rows", 10_000, 100_000, 60_000, 10_000)
    use_ref = k[4].checkbox("Product reference-price features", True,
                            help="Adds the product's typical price/margin × units and predicts the ratio to it. "
                                 "Greatly improves accuracy and extrapolation to later periods; turn off to use only raw features.")
    if st.button("🚀 Train model", type="primary"):
        with st.spinner("Training..."):
            d = DF.sort_values("Date")
            cut = int(len(d) * (1 - test_frac))
            tr_df, te_df = d.iloc[:cut], d.iloc[cut:]
            if len(tr_df) > cap:
                tr_df = tr_df.sample(cap, random_state=42).sort_values("Date")
            t0 = DF["Date"].min()
            refs = None
            if use_ref:
                refs = {"price": (tr_df["Sales"] / tr_df["Units Sold"]).groupby(tr_df["Product Name"]).mean(),
                        "margin": (tr_df["Profit"] / tr_df["Units Sold"]).groupby(tr_df["Product Name"]).mean()}
            Xtr, Xte = add_features(tr_df, refs, target, t0), add_features(te_df, refs, target, t0)
            pipe, cols = make_model(mname, use_ref)
            feat = list(dict.fromkeys(cols + ["_base"]))
            model = RelModel(pipe, rel=use_ref).fit(Xtr[feat], Xtr[target])
            pred = model.predict(Xte[feat])
            yt = Xte[target].values
            met = {"MAE": mean_absolute_error(yt, pred), "RMSE": mean_squared_error(yt, pred) ** .5,
                   "R²": r2_score(yt, pred), "WAPE %": np.abs(yt - pred).sum() / np.abs(yt).sum() * 100}
            sub = Xte.sample(min(4000, len(Xte)), random_state=1)
            imp = permutation_importance(model, sub[feat], sub[target], n_repeats=3, random_state=0,
                                         scoring="neg_mean_absolute_error")
            imp = pd.DataFrame({"feature": feat, "importance": imp.importances_mean}).sort_values("importance")
            st.session_state["ml"] = dict(model=model, feat=feat, refs=refs, t0=t0, target=target, name=mname, met=met,
                                          test=te_df.assign(Predicted=pred), imp=imp, use_ref=use_ref)

    ml = st.session_state.get("ml")
    if ml:
        st.success(f"Trained **{ml['name']}** → predicting **{ml['target']}**  (test period: "
                   f"{ml['test']['Date'].min().date()} to {ml['test']['Date'].max().date()})")
        mc = st.columns(4)
        for col, (k_, v) in zip(mc, ml["met"].items()):
            col.metric(k_, f"{v:,.3f}" if k_ == "R²" else f"{v:,.1f}")
        te = ml["test"]
        a, b = st.columns(2)
        with a:
            s = te.sample(min(3000, len(te)), random_state=0)
            fig = px.scatter(s, x=ml["target"], y="Predicted", opacity=.4, title="Actual vs predicted (test sample)")
            mx = float(max(s[ml["target"]].max(), s["Predicted"].max()))
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="perfect", line=dict(dash="dash", color="red")))
            plot(fig)
        with b:
            plot(px.bar(ml["imp"], x="importance", y="feature", orientation="h",
                        title="Permutation importance (increase in MAE when shuffled)"))
        mon = te.groupby("YearMonth")[[ml["target"], "Predicted"]].sum().reset_index()
        plot(px.line(mon, x="YearMonth", y=[ml["target"], "Predicted"], title="Monthly totals in test period: actual vs predicted"))
        download(te[["Date", "Product Name", "Cities", "SalesPerson Name", "Scenario Name", "Units Sold", ml["target"], "Predicted"]],
                 "⬇ Download test predictions", "test_predictions.csv")

        st.subheader(f"Predict a new {ml['target']} value")
        p = st.columns(6)
        prod = p[0].selectbox("Product", sorted(DF["Product Name"].unique()))
        city = p[1].selectbox("City", sorted(DF["Cities"].unique()))
        rep = p[2].selectbox("Sales rep", sorted(DF["SalesPerson Name"].unique()))
        scn = p[3].selectbox("Scenario", sorted(DF["Scenario Name"].unique()))
        units = p[4].number_input("Units", 1, 100, 3)
        dt = p[5].date_input("Date", DF["Date"].max().date() + pd.Timedelta(days=30))
        info = DF[DF["Product Name"] == prod].iloc[0]

        def one(city_=city, prod_=prod, rep_=rep):
            r = DF[DF["Product Name"] == prod_].iloc[0]
            row = pd.DataFrame([{"Date": pd.Timestamp(dt), "Product Name": prod_, "Category": r["Category"], "Brand": r["Brand"],
                                 "Warranty": r.get("Warranty", "NA"), "Cities": city_, "SalesPerson Name": rep_,
                                 "Scenario Name": scn, "Units Sold": units, "Year": dt.year, "Month": dt.month,
                                 "Quarter": (dt.month - 1) // 3 + 1, "DayOfWeek": dt.weekday(), "DayOfYear": dt.timetuple().tm_yday}])
            return float(ml["model"].predict(add_features(row, ml["refs"], ml["target"], ml["t0"])[ml["feat"]])[0])

        val = one()
        st.metric(f"Predicted {ml['target']}", f"{val:,.0f}")
        wi = pd.DataFrame({"City": sorted(DF["Cities"].unique())})
        wi["Predicted"] = [one(city_=c_) for c_ in wi["City"]]
        plot(px.bar(wi, x="City", y="Predicted", title=f"What-if: same order in each city ({prod}, {units} units)"))
    else:
        st.info("Choose settings and click **Train model**.")

# ============================================================================= 4. FORECASTING
RULES = {"Monthly": ("MS", 12), "Quarterly": ("QS", 4), "Weekly": ("W", 52)}
MODELS = ["Seasonal Naive", "Linear Trend + Seasonality", "Holt-Winters", "SARIMA", "Gradient Boosting (YoY ratio)"]


def fc_snaive(y, h, m):
    v = y.values
    return np.array([v[-m + (i % m)] for i in range(h)]), None


def fc_linear(y, h, m):
    n = len(y)

    def X(t):
        return np.column_stack([np.ones(len(t)), t] + [(t % m == k).astype(float) for k in range(1, m)])
    t = np.arange(n)
    beta = np.linalg.lstsq(X(t), y.values, rcond=None)[0]
    return X(np.arange(n, n + h)) @ beta, None


def fc_hw(y, h, m):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(y.values, trend="add", damped_trend=True, seasonal="mul" if (y > 0).all() else "add",
                               seasonal_periods=m, initialization_method="estimated").fit()
    return fit.forecast(h), None


def fc_sarima(y, h, m):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    sc = y.mean()
    fit = SARIMAX(y.values / sc, order=(1, 1, 1), seasonal_order=(0, 1, 1, m),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=150)
    f_ = fit.get_forecast(h)
    return f_.predicted_mean * sc, f_.conf_int(alpha=.05) * sc


def fc_gbm(y, h, m):
    v = list(y.values.astype(float))
    n = len(v)
    r = np.array([v[i] / v[i - m] for i in range(m, n)])
    L = 3

    def feats(rr, pos):
        return [rr[-1], rr[-2], rr[-3], np.mean(rr[-L:]), pos % m]
    Xs = np.array([feats(r[:i], i + m) for i in range(L, len(r))])
    ys = r[L:]
    mod = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=.05, subsample=.8, random_state=0).fit(Xs, ys)
    rr, out = list(r), []
    for i in range(h):
        ratio = float(mod.predict(np.array([feats(np.array(rr), n + i)]))[0])
        val = ratio * v[n + i - m]
        v.append(val)
        rr.append(ratio)
        out.append(val)
    return np.array(out), None


FUNCS = {"Seasonal Naive": fc_snaive, "Linear Trend + Seasonality": fc_linear, "Holt-Winters": fc_hw,
         "SARIMA": fc_sarima, "Gradient Boosting (YoY ratio)": fc_gbm}


def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return {"MAE": np.mean(np.abs(a - p)), "RMSE": np.sqrt(np.mean((a - p) ** 2)),
            "MAPE %": np.mean(np.abs((a - p) / np.where(a == 0, np.nan, a))) * 100, "Bias": np.mean(p - a)}


@st.cache_data(show_spinner="Back-testing and fitting models...")
def run_forecast(y: pd.Series, h: int, hb: int, m: int, rule: str, models: tuple):
    idx_future = pd.date_range(y.index[-1], periods=h + 1, freq=rule)[1:]
    train, test = y.iloc[:-hb], y.iloc[-hb:]
    bt, fut, rows, errs = {}, {}, [], {}
    for nm in models:
        try:
            pb, _ = FUNCS[nm](train, hb, m)
            pf, ci = FUNCS[nm](y, h, m)
            bt[nm], fut[nm] = np.asarray(pb, float), (np.asarray(pf, float), None if ci is None else np.asarray(ci, float))
        except Exception as e:
            errs[nm] = str(e)[:120]
    if len(bt) >= 2:
        bt["Ensemble (average)"] = np.mean(list(bt.values()), axis=0)
        fut["Ensemble (average)"] = (np.mean([v[0] for k, v in fut.items() if k in bt and k != "Ensemble (average)"], axis=0), None)
    for nm, p in bt.items():
        rows.append({"Model": nm, **metrics(test.values, p)})
    res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True) if rows else pd.DataFrame()
    return dict(res=res, bt=bt, fut=fut, errs=errs, idx_future=idx_future, test_idx=test.index)


with t4:
    st.subheader("Forecast future sales")
    k = st.columns(5)
    fm = k[0].selectbox("Metric to forecast", METRICS, key="fm")
    fq = k[1].selectbox("Frequency", list(RULES), key="fq")
    scn_f = k[2].selectbox("Scenario", ["Actual", "Budget", "All"], key="fs")
    scope = k[3].selectbox("Forecast for", ["Total"] + [d for d in ["Category", "Brand", "Cities", "SalesPerson Name", "Product Name", "Warranty"] if d in DF])
    val = k[4].selectbox("Value", sorted(DF[scope].unique()), key="fv") if scope != "Total" else None

    rule, m = RULES[fq]
    s = DF if scn_f == "All" else DF[DF["Scenario Name"] == scn_f]
    if val is not None:
        s = s[s[scope] == val]
    y = s.set_index("Date")[fm].resample(rule).sum().astype(float)
    y = y[y.index >= y[y > 0].index.min()] if (y > 0).any() else y

    k = st.columns(3)
    hdef = {"Monthly": 12, "Quarterly": 4, "Weekly": 26}[fq]
    h = k[0].slider("Forecast horizon (periods)", 1, {"Monthly": 36, "Quarterly": 12, "Weekly": 104}[fq], hdef)
    hb = k[1].slider("Back-test periods (held out)", max(2, m // 4), max(3, min(m * 2, len(y) // 3)), min(m, max(3, len(y) // 4)))
    avail = [mm for mm in MODELS if len(y) >= (2 * m + hb if mm in ("Holt-Winters", "SARIMA") else m + hb + 6)]
    chosen = k[2].multiselect("Models", MODELS, default=avail)
    st.caption(f"Series: **{len(y)} {fq.lower()} periods** ({y.index.min().date()} → {y.index.max().date()}). "
               "Models need ≥ 2 seasonal cycles of history (Holt-Winters/SARIMA) and are skipped if the series is too short.")
    chosen = [c_ for c_ in chosen if c_ in avail]
    if len(y) < m + hb + 6 or not chosen:
        st.warning("Not enough history for the selected frequency/back-test window. Try a coarser frequency or a smaller window.")
    else:
        R = run_forecast(y, h, hb, m, rule, tuple(chosen))
        for nm, e in R["errs"].items():
            st.warning(f"{nm} failed: {e}")
        if R["res"].empty:
            st.stop()
        st.write(f"**Back-test leaderboard** (last {hb} periods held out)")
        show_df(R["res"].style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE %": "{:.2f}", "Bias": "{:,.0f}"}))
        best = R["res"].iloc[0]["Model"]
        pick = st.selectbox("Model used for the final forecast", list(R["res"]["Model"]), index=0,
                            help="Default is the lowest back-test RMSE.")
        pf, ci = R["fut"][pick]
        rmse = float(R["res"].set_index("Model").loc[pick, "RMSE"])
        lo, hi = (ci[:, 0], ci[:, 1]) if ci is not None else (pf - 1.96 * rmse, pf + 1.96 * rmse)
        fdf = pd.DataFrame({"Date": R["idx_future"], "Forecast": pf, "Lower 95%": np.maximum(lo, 0), "Upper 95%": hi})

        fig = go.Figure()
        fig.add_scatter(x=y.index, y=y.values, name="History", line=dict(color="#1f77b4"))
        for nm, p in R["bt"].items():
            fig.add_scatter(x=R["test_idx"], y=p, name=f"Back-test: {nm}", line=dict(dash="dot", width=1),
                            visible=True if nm == pick else "legendonly")
        fig.add_scatter(x=fdf["Date"], y=fdf["Upper 95%"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fdf["Date"], y=fdf["Lower 95%"], fill="tonexty", fillcolor="rgba(255,127,14,.2)",
                        line=dict(width=0), name="95% interval", hoverinfo="skip")
        fig.add_scatter(x=fdf["Date"], y=fdf["Forecast"], name=f"Forecast ({pick})", line=dict(color="#ff7f0e", width=3))
        fig.update_layout(title=f"{fm} forecast – {val or 'Total'} ({scn_f}) – {fq.lower()}", hovermode="x unified")
        plot(fig)

        last_h = y.iloc[-h:] if len(y) >= h else y
        c = st.columns(3)
        c[0].metric(f"Forecast total (next {h})", fmt(fdf["Forecast"].sum()))
        c[1].metric(f"vs last {len(last_h)} periods", f"{(fdf['Forecast'].sum() / last_h.sum() - 1) * 100:+.1f}%")
        c[2].metric("Best back-test model", best)
        show_df(fdf.style.format({"Forecast": "{:,.0f}", "Lower 95%": "{:,.0f}", "Upper 95%": "{:,.0f}"}))
        download(fdf, "⬇ Download forecast", "forecast.csv")
        st.caption("Intervals: SARIMA's native 95% interval; for other models forecast ± 1.96 × back-test RMSE (an approximation). "
                   "Forecasts assume past patterns continue; they are not guarantees.")
