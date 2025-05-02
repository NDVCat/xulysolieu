from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import io

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# 📌 Cấu hình
MODEL_DIR = "models"
ANOMALY_MODEL_PATH = "combined_models.pkl"

PRESERVED_COLUMNS = ['UniqueId', 'Date', 'Method']
EXPECTED_COLUMNS = ['DayOn', 'Qoil', 'Qgas', 'Qwater', 'GOR', 'ChokeSize',
                    'Press_WH', 'Oilrate', 'LiqRate', 'GasRate']

# 📌 Tải mô hình phát hiện bất thường
try:
    if not os.path.exists(ANOMALY_MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy tệp {ANOMALY_MODEL_PATH}")
    
    combined_models = joblib.load(ANOMALY_MODEL_PATH)
    scaler = combined_models["scaler"]
    pca = combined_models["pca"]
    iso_forest = combined_models["isolation_forest"]

    print("✅ Đã tải mô hình phát hiện bất thường!")
except Exception as e:
    raise RuntimeError(f"❌ Lỗi khi tải mô hình phát hiện bất thường: {e}")

# 📌 Tải mô hình dự đoán riêng lẻ
def load_model(col_name):
    model_path = os.path.join(MODEL_DIR, f"{col_name}.pkl")
    return joblib.load(model_path) if os.path.exists(model_path) else None

# 📌 Xử lý dữ liệu đầu vào
def preprocess_input(df):
    df = df[[col for col in PRESERVED_COLUMNS + EXPECTED_COLUMNS if col in df.columns]]
    df.replace({"...": np.nan, "null": np.nan, "NaN": np.nan, "": np.nan}, inplace=True)

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df[EXPECTED_COLUMNS] = df[EXPECTED_COLUMNS].astype(float)
    return df

# 📌 Dự đoán các giá trị thiếu
def predict_missing_values(df):
    forecast_mask = pd.DataFrame(False, index=df.index, columns=EXPECTED_COLUMNS)
    forecasted_info = []

    for idx, row in df.iterrows():
        missing_cols = row[EXPECTED_COLUMNS].isnull()
        if missing_cols.any():
            missing_cols_list = missing_cols[missing_cols].index.tolist()
            if 'Qoil' in missing_cols_list:
                missing_cols_list.remove('Qoil')
                missing_cols_list.insert(0, 'Qoil')

            for col in missing_cols_list:
                model = load_model(col)
                if model is not None:
                    try:
                        input_features = [f for f in EXPECTED_COLUMNS if f != col]
                        input_data = pd.DataFrame([row[input_features].values], columns=input_features)
                        predicted_value = model.predict(input_data)[0]
                        df.at[idx, col] = predicted_value
                        forecast_mask.at[idx, col] = True

                        forecasted_info.append({
                            'row_index': idx,
                            'column': col,
                            'predicted_value': predicted_value
                        })
                    except Exception as e:
                        print(f"❌ Lỗi khi dự đoán {col} tại dòng {idx}: {e}")

    df["is_forecasted"] = forecast_mask.any(axis=1).astype(int)
    forecasted_columns = forecast_mask.apply(lambda row: ", ".join(row.index[row]), axis=1)
    df["forecasted_columns"] = forecasted_columns
    return df, forecasted_info

# 📌 Phát hiện bất thường
def detect_anomalies(df):
    try:
        numeric_cols = [col for col in EXPECTED_COLUMNS if col in scaler.feature_names_in_]
        df_anomaly = df[numeric_cols].dropna()

        if df_anomaly.empty:
            print("⚠️ Không có dòng nào đủ dữ liệu để phát hiện bất thường.")
            df["anomaly"] = np.nan
            df["anomaly_label"] = "unknown"
            return df

        # Chuẩn hóa, PCA, Isolation Forest
        df_scaled = pd.DataFrame(scaler.transform(df_anomaly), columns=numeric_cols)
        pca_result = pca.transform(df_scaled)
        pca_df = pd.DataFrame(pca_result[:, :2], columns=["PC1", "PC2"])
        anomaly_pred = iso_forest.predict(pca_df)

        df.loc[df_anomaly.index, "anomaly"] = anomaly_pred
        df["anomaly_label"] = df["anomaly"].map({1: "normal", -1: "anomaly"})
        df["anomaly_label"].fillna("unknown", inplace=True)

        return df
    except Exception as e:
        print(f"❌ Lỗi khi phát hiện bất thường: {e}")
        df["anomaly"] = np.nan
        df["anomaly_label"] = "error"
        return df

# 📌 API xử lý dữ liệu
@app.route('/process', methods=['POST'])
def process_data():
    try:
        if request.content_type == 'text/csv':
            csv_text = request.data.decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_text))
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                return jsonify({"error": "Unsupported file format"}), 400
        else:
            return jsonify({"error": "No valid input found (file or CSV text)"}), 400

        df = preprocess_input(df)
        df, forecasted_info = predict_missing_values(df)
        df = detect_anomalies(df)

        result = {
            "status": "success",
            "data": df.to_dict(orient='records'),
            "forecasted_info": forecasted_info,
            "anomaly_stats": {
                "total_records": len(df),
                "normal": int((df["anomaly"] == 1).sum()),
                "anomaly": int((df["anomaly"] == -1).sum()),
                "unknown": int((df["anomaly_label"] == "unknown").sum()),
                "error": int((df["anomaly_label"] == "error").sum())
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

# 📌 Khởi chạy
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
