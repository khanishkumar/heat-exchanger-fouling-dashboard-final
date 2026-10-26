# Heat Exchanger Fouling Prediction Dashboard

## Includes
1. CSV upload, data preview, missing-data report, imputation, and cleaned CSV download.
2. Random Forest and XGBoost classification for Yes/No fouling using historical labels.
3. Optional configured heat-transfer-coefficient threshold rule.
4. Heat-transfer coefficient, pressure-drop, flow-rate, and historical fouling graphs.
5. Optional LSTM forecast for timestamped historical sensor data with fouling labels.

## Required data
Use one row per measurement, a real fouling label column (Yes/No or 1/0), and timestamp for LSTM. Labels are not imputed. Sensor units and threshold must be validated for your particular exchanger.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Upload app.py, requirements.txt, and README.md to the root of a GitHub repository. On https://share.streamlit.io create an app, choose the repository, branch `main`, and main file `app.py`. TensorFlow may make the first deployment take longer.

## Limitation
LSTM forecast holds the latest sensor values constant because future sensor values are unknown. It is a scenario forecast, not a guaranteed future fouling date. Prototype only; validate before operational use.
