import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    Sequential = None

st.set_page_config(page_title="Heat Exchanger Fouling Dashboard", layout="wide")
st.title("Heat Exchanger Fouling Prediction Dashboard")
st.caption("CSV data cleaning • Random Forest • XGBoost • LSTM forecasting")

with st.expander("Data requirements and limitations"):
    st.markdown("""
    - Supervised classification needs a real historical fouling label column (Yes/No or 1/0).
    - LSTM needs timestamped, chronologically ordered, labelled historical data with enough rows.
    - Numeric missing features are filled with their median; categorical missing features with their most frequent value. Fouling labels are never imputed.
    - The heat-transfer-coefficient (HTC/U) threshold is equipment- and unit-specific. Configure only a threshold validated for your exchanger.
    - Forecasting holds the latest sensor values constant because future sensor inputs are unknown; treat it as a scenario, not a guaranteed fouling date.
    - Prototype only: not a safety system or a replacement for engineering validation.
    """)

file = st.sidebar.file_uploader("Upload historical CSV", type=["csv"])
if file is None:
    st.info("Upload your historical heat-exchanger CSV using the sidebar.")
    st.markdown("Suggested columns: timestamp, heat_transfer_coefficient, pressure_drop, flow_rate, fouling_status.")
    st.stop()
try:
    raw = pd.read_csv(file)
except Exception as e:
    st.error(f"Cannot read CSV: {e}")
    st.stop()
if raw.empty:
    st.error("CSV contains no rows.")
    st.stop()
cols = list(raw.columns)
def guess(words):
    for c in cols:
        if any(w in str(c).lower().replace(" ", "_") for w in words):
            return c
    return "(none)"
opts = ["(none)"] + cols
def choose(label, words):
    g = guess(words)
    return st.sidebar.selectbox(label, opts, index=opts.index(g) if g in opts else 0)
time_col = choose("Timestamp column", ["timestamp", "datetime", "date", "time"])
label_col = choose("Historical fouling label", ["fouling_status", "fouling_present", "fouling", "target", "label"])
htc_col = choose("Heat-transfer coefficient", ["heat_transfer_coefficient", "heat_transfer", "htc", "u_value"])
dp_col = choose("Pressure drop", ["pressure_drop", "pressuredrop", "delta_p"])
flow_col = choose("Flow rate", ["flow_rate", "flowrate", "mass_flow"])
time_col = None if time_col == "(none)" else time_col
label_col = None if label_col == "(none)" else label_col
htc_col = None if htc_col == "(none)" else htc_col
dp_col = None if dp_col == "(none)" else dp_col
flow_col = None if flow_col == "(none)" else flow_col

st.header("1. Data input and cleaning")
a,b,c = st.columns(3)
a.metric("Rows", len(raw)); b.metric("Columns", len(cols)); c.metric("Missing cells", int(raw.isna().sum().sum()))
st.subheader("Uploaded data preview")
st.dataframe(raw.head(30), use_container_width=True)
st.subheader("Missing values before cleaning")
st.dataframe(raw.isna().sum().rename("Missing cells").to_frame(), use_container_width=True)

clean = raw.copy()
if time_col:
    parsed = pd.to_datetime(clean[time_col], errors="coerce")
    if parsed.notna().any():
        clean[time_col] = parsed
for col in clean.columns:
    if col == label_col:
        continue
    if pd.api.types.is_numeric_dtype(clean[col]):
        med = clean[col].median()
        if pd.notna(med):
            clean[col] = clean[col].fillna(med)
    else:
        modes = clean[col].mode(dropna=True)
        if not modes.empty:
            clean[col] = clean[col].fillna(modes.iloc[0])
st.subheader("Missing values after cleaning")
st.dataframe(clean.isna().sum().rename("Missing cells").to_frame(), use_container_width=True)
st.dataframe(clean.head(30), use_container_width=True)
st.download_button("Download cleaned CSV", clean.to_csv(index=False).encode(), "cleaned_heat_exchanger_data.csv", "text/csv")

def labels(s):
    z = s.astype(str).str.strip().str.lower()
    result = pd.Series(np.nan, index=s.index, dtype=float)
    result[z.isin(["yes","y","true","1","fouling","fouled","present","positive"])] = 1
    result[z.isin(["no","n","false","0","clean","not fouled","not_fouled","absent","negative"])] = 0
    n = pd.to_numeric(s, errors="coerce")
    result[n == 1] = 1; result[n == 0] = 0
    return result

st.divider()
st.header("2. Fouling detection — Random Forest and XGBoost")
use_rule = st.checkbox("Also apply a configured HTC threshold rule")
threshold = None
direction = "below"
if use_rule:
    if htc_col:
        threshold = st.number_input("HTC threshold (same units as your data)", value=2.0)
        direction = st.selectbox("Fouling rule", ["below", "above"])
        st.caption("Example: 'HTC below threshold indicates possible fouling' is valid only if supported by your engineering baseline.")
    else:
        st.warning("Select an HTC column in the sidebar to use the threshold rule.")

trained_models = {}
if not label_col:
    st.warning("Select a historical fouling label column to train models. Without historical labels, supervised Yes/No prediction cannot be trained reliably.")
else:
    y0 = labels(clean[label_col])
    valid = y0.notna()
    y = y0[valid].astype(int)
    X = clean.loc[valid].drop(columns=[label_col] + ([time_col] if time_col else []), errors="ignore")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if len(y) < 10 or y.nunique() < 2 or X.shape[1] == 0:
        st.error("Need at least 10 labelled rows, both fouling and clean classes, and usable sensor features.")
    else:
        counts = y.value_counts()
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y if counts.min() >= 2 else None)
            nums = Xtr.select_dtypes(include=np.number).columns.tolist()
            cats = [c for c in Xtr.columns if c not in nums]
            trans = []
            if nums: trans.append(("num", Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]), nums))
            if cats: trans.append(("cat", Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), cats))
            prep = ColumnTransformer(trans)
            candidates = [("Random Forest", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42))]
            if XGBClassifier is not None:
                candidates.append(("XGBoost", XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,subsample=.9,colsample_bytree=.9,eval_metric="logloss",random_state=42)))
            for name, model in candidates:
                pipe = Pipeline([("prep", prep), ("model", model)])
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
                trained_models[name] = pipe
                st.subheader(name)
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Accuracy", f"{accuracy_score(yte,pred):.3f}")
                m2.metric("Precision", f"{precision_score(yte,pred,zero_division=0):.3f}")
                m3.metric("Recall", f"{recall_score(yte,pred,zero_division=0):.3f}")
                m4.metric("F1", f"{f1_score(yte,pred,zero_division=0):.3f}")
                cm = confusion_matrix(yte,pred,labels=[0,1])
                fig,ax = plt.subplots()
                ax.imshow(cm)
                ax.set_xticks([0,1],["Predicted No","Predicted Yes"]); ax.set_yticks([0,1],["Actual No","Actual Yes"])
                for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
                ax.set_title("Confusion matrix"); st.pyplot(fig); plt.close(fig)
            st.subheader("Latest row prediction")
            latest = X.iloc[[-1]]
            cards = st.columns(len(trained_models))
            for i,(name,model) in enumerate(trained_models.items()):
                p = model.predict_proba(latest)[0]
                cls = int(np.argmax(p))
                cards[i].metric(name, "Yes — Fouling" if cls else "No — Clean", f"Confidence {p[cls]:.1%}")
            if threshold is not None and htc_col:
                htc_values = pd.to_numeric(clean[htc_col], errors="coerce").dropna()
                if len(htc_values):
                    latest_htc = float(htc_values.iloc[-1])
                    yes = latest_htc < threshold if direction == "below" else latest_htc > threshold
                    st.info(f"Configured rule: latest HTC={latest_htc:g}, threshold={threshold:g}. Rule indication: {'Yes — possible fouling' if yes else 'No — threshold not crossed'}. Validate units and threshold.")
        except Exception as e:
            st.error(f"Model training error: {e}")

st.divider()
st.header("3. Historical trends and future fouling forecast (LSTM)")
plot_df = clean.copy()
if time_col:
    plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors="coerce")
    plot_df = plot_df.sort_values(time_col)
xaxis = plot_df[time_col] if time_col else np.arange(len(plot_df))
for col in [htc_col, dp_col, flow_col]:
    if col:
        fig,ax = plt.subplots(figsize=(10,3))
        ax.plot(xaxis, pd.to_numeric(plot_df[col], errors="coerce"))
        ax.set_title(col); ax.set_xlabel(time_col or "Row/time step"); ax.set_ylabel(col); ax.grid(alpha=.3)
        fig.autofmt_xdate(); st.pyplot(fig); plt.close(fig)
if label_col:
    fig,ax = plt.subplots(figsize=(10,2.5))
    ax.step(xaxis, labels(plot_df[label_col]), where="post")
    ax.set_yticks([0,1],["No fouling","Fouling"]); ax.set_title("Historical fouling occurrence"); ax.grid(alpha=.3)
    fig.autofmt_xdate(); st.pyplot(fig); plt.close(fig)

if Sequential is None:
    st.warning("TensorFlow is unavailable, so LSTM forecasting is disabled. Install tensorflow-cpu from requirements.txt and redeploy.")
elif not time_col or not label_col:
    st.warning("Choose both timestamp and historical fouling label columns to enable LSTM.")
else:
    lookback = st.number_input("Look-back window (past rows)", min_value=5, max_value=200, value=20)
    horizon = st.number_input("Forecast horizon (future steps)", min_value=1, max_value=100, value=10)
    if st.button("Train LSTM and forecast", type="primary"):
        try:
            d = clean.copy()
            d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
            d = d.dropna(subset=[time_col]).sort_values(time_col)
            yseq = labels(d[label_col])
            sensor_cols = [c for c in [htc_col,dp_col,flow_col] if c]
            sensor_cols += [c for c in d.select_dtypes(include=np.number).columns if c not in sensor_cols and c != label_col]
            sensor_cols = list(dict.fromkeys(sensor_cols))
            seq = d[sensor_cols].apply(pd.to_numeric,errors="coerce")
            seq = seq.fillna(seq.median()).fillna(0)
            combined = seq.copy(); combined["_target"] = yseq.to_numpy()
            combined = combined.dropna(subset=["_target"])
            if len(combined) <= int(lookback)+10 or combined["_target"].nunique() < 2:
                st.error("Not enough chronological labelled rows. Add more history or reduce look-back.")
            else:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(combined[sensor_cols])
                target = combined["_target"].to_numpy(dtype=np.float32)
                xx,yy = [],[]
                for i in range(int(lookback),len(combined)):
                    xx.append(scaled[i-int(lookback):i]); yy.append(target[i])
                xx=np.asarray(xx,dtype=np.float32); yy=np.asarray(yy,dtype=np.float32)
                split=int(len(xx)*.8)
                if split < 5 or len(xx)-split < 2:
                    st.error("Not enough sequences for train/test evaluation.")
                else:
                    net=Sequential([LSTM(32,input_shape=(xx.shape[1],xx.shape[2])),Dropout(.2),Dense(16,activation="relu"),Dense(1,activation="sigmoid")])
                    net.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
                    net.fit(xx[:split],yy[:split],epochs=30,batch_size=32,validation_split=.15,callbacks=[EarlyStopping(patience=5,restore_best_weights=True)],verbose=0)
                    testp=net.predict(xx[split:],verbose=0).ravel()
                    testy=yy[split:].astype(int); testpred=(testp>=.5).astype(int)
                    st.write(f"LSTM holdout accuracy: {accuracy_score(testy,testpred):.3f} | precision: {precision_score(testy,testpred,zero_division=0):.3f} | recall: {recall_score(testy,testpred,zero_division=0):.3f}")
                    window=scaled[-int(lookback):].copy(); probs=[]
                    for _ in range(int(horizon)):
                        probs.append(float(net.predict(window[None,:,:],verbose=0)[0,0]))
                        window=np.vstack([window[1:],window[-1]])
                    fig,ax=plt.subplots(figsize=(10,4))
                    ax.plot(np.arange(1,int(horizon)+1),probs,marker="o")
                    ax.axhline(.5,linestyle="--"); ax.set_ylim(0,1)
                    ax.set_xlabel("Future step (not calendar time)"); ax.set_ylabel("Fouling probability")
                    ax.set_title("LSTM scenario forecast"); ax.grid(alpha=.3); st.pyplot(fig); plt.close(fig)
                    first=next((i+1 for i,p in enumerate(probs) if p>=.5),None)
                    st.warning(f"Scenario first crosses 0.5 at future step {first}." if first else "Scenario does not cross 0.5 in the selected horizon.")
                    st.caption("Future sensor readings are unknown, so the model repeats the latest sensor state. This is a scenario, not a reliable calendar-date prediction.")
        except Exception as e:
            st.error(f"LSTM failed: {e}")

st.caption("Validate target labels, sensor units, thresholds, chronological testing, and model performance before operational use.")
