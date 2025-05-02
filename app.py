from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

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

# 📌 Hàm tải mô hình dự báo giá trị thiếu (chỉ khi cần)
def load_model(col_name):
    model_path = os.path.join(MODEL_DIR, f"{col_name}.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

# 📌 Tiền xử lý đầu vào
def preprocess_input(df):
    df = df[[col for col in PRESERVED_COLUMNS + EXPECTED_COLUMNS if col in df.columns]]
    df.replace({"...": np.nan, "null": np.nan, "NaN": np.nan, "": np.nan}, inplace=True)

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df[EXPECTED_COLUMNS] = df[EXPECTED_COLUMNS].astype(float)
    return df

# 📌 Dự đoán giá trị thiếu (dynamic model loading)
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
        df_scaled = pd.DataFrame(scaler.transform(df[numeric_cols]), columns=numeric_cols)
        pca_result = pca.transform(df_scaled)
        pca_df = pd.DataFrame(pca_result[:, :2], columns=["PC1", "PC2"])
        anomalies = iso_forest.predict(pca_df[["PC1", "PC2"]])
        df["anomaly"] = anomalies
        df["anomaly_label"] = df["anomaly"].map({1: "normal", -1: "anomaly"})
        return df
    except Exception as e:
        print(f"❌ Lỗi khi phát hiện bất thường: {e}")
        return df

# 📌 API chính
@app.route('/process', methods=['POST'])
def process_data():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

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
                "anomaly": int((df["anomaly"] == -1).sum())
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
