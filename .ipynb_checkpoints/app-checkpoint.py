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

# 📌 Định nghĩa các cột đầu vào cần thiết
PRESERVED_COLUMNS = ['UniqueId', 'Date', 'Method']
EXPECTED_COLUMNS = ['DayOn', 'Qoil', 'Qgas', 'Qwater', 'GOR', 'ChokeSize', 
                   'Press_WH', 'Oilrate', 'LiqRate', 'GasRate']

# 📌 Tải tất cả mô hình
MODEL_FILES = {
    "prediction_models": "reverse_prediction_models.pkl",
    "anomaly_detection": "combined_models.pkl"
}

# Kiểm tra và tải mô hình
try:
    # Tải mô hình dự đoán giá trị thiếu
    if not os.path.exists(MODEL_FILES["prediction_models"]):
        raise FileNotFoundError(f"⚠️ Không tìm thấy tệp '{MODEL_FILES['prediction_models']}'!")
    models = joblib.load(MODEL_FILES["prediction_models"])
    
    # Tải mô hình phát hiện bất thường
    if not os.path.exists(MODEL_FILES["anomaly_detection"]):
        raise FileNotFoundError(f"⚠️ Không tìm thấy tệp '{MODEL_FILES['anomaly_detection']}'!")
    combined_models = joblib.load(MODEL_FILES["anomaly_detection"])
    scaler = combined_models["scaler"]
    pca = combined_models["pca"]
    iso_forest = combined_models["isolation_forest"]
    
    print("✅ Đã tải tất cả mô hình thành công!")
except Exception as e:
    raise RuntimeError(f"❌ Lỗi khi tải mô hình: {e}")

# 📌 Hàm tiền xử lý dữ liệu đầu vào
def preprocess_input(df):
    # Chọn các cột cần thiết
    df = df[[col for col in PRESERVED_COLUMNS + EXPECTED_COLUMNS if col in df.columns]]
    
    # Chuyển các giá trị không hợp lệ thành NaN
    df.replace({"...": np.nan, "null": np.nan, "NaN": np.nan, "": np.nan}, inplace=True)
    
    # Đảm bảo tất cả các cột cần thiết đều có trong DataFrame
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
            
    # Chuyển đổi kiểu dữ liệu
    df[EXPECTED_COLUMNS] = df[EXPECTED_COLUMNS].astype(float)
    
    return df

# 📌 Hàm dự đoán giá trị thiếu
def predict_missing_values(df):
    forecast_mask = pd.DataFrame(False, index=df.index, columns=EXPECTED_COLUMNS)
    forecasted_info = []
    
    for idx, row in df.iterrows():
        missing_cols = row[EXPECTED_COLUMNS].isnull()
        if missing_cols.any():
            missing_cols_list = missing_cols[missing_cols].index.tolist()
            
            # Ưu tiên dự đoán Qoil trước
            if 'Qoil' in missing_cols_list:
                missing_cols_list.remove('Qoil')
                missing_cols_list.insert(0, 'Qoil')
            
            for col in missing_cols_list:
                if col in models:
                    try:
                        # Lấy các features đầu vào cho mô hình
                        input_features = [f for f in EXPECTED_COLUMNS if f != col]
                        input_data = pd.DataFrame([row[input_features].values], columns=input_features)
                        
                        # Dự đoán giá trị thiếu
                        predicted_value = models[col].predict(input_data)[0]
                        df.at[idx, col] = predicted_value
                        forecast_mask.at[idx, col] = True
                        
                        forecasted_info.append({
                            'row_index': idx,
                            'column': col,
                            'predicted_value': predicted_value
                        })
                    except Exception as e:
                        print(f"❌ Lỗi khi dự đoán {col} tại dòng {idx}: {e}")
    
    df["is_forecasted"] = forecast_mask.any(axis=1)
    forecasted_columns = forecast_mask.apply(lambda row: ", ".join(row.index[row]), axis=1)
    df["forecasted_columns"] = forecasted_columns
    
    return df, forecasted_info

# 📌 Hàm phát hiện bất thường
def detect_anomalies(df):
    try:
        # Chuẩn hóa dữ liệu
        numeric_cols = [col for col in EXPECTED_COLUMNS if col in scaler.feature_names_in_]
        df_scaled = pd.DataFrame(scaler.transform(df[numeric_cols]), columns=numeric_cols)
        
        # Áp dụng PCA
        pca_result = pca.transform(df_scaled)
        pca_df = pd.DataFrame(pca_result[:, :2], columns=["PC1", "PC2"])
        
        # Phát hiện bất thường
        anomalies = iso_forest.predict(pca_df[["PC1", "PC2"]])
        df["anomaly"] = anomalies
        df["anomaly_label"] = df["anomaly"].map({1: "normal", -1: "anomaly"})
        
        return df
    except Exception as e:
        print(f"❌ Lỗi khi phát hiện bất thường: {e}")
        return df

# 📌 API xử lý file CSV đầu vào
@app.route('/process', methods=['POST'])
def process_data():
    try:
        # Kiểm tra dữ liệu đầu vào
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Đọc file (hỗ trợ cả Excel và CSV)
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                return jsonify({"error": "Unsupported file format"}), 400
        except Exception as e:
            return jsonify({"error": f"Error reading file: {str(e)}"}), 400
        
        # Tiền xử lý dữ liệu
        df = preprocess_input(df)
        
        # Dự đoán giá trị thiếu
        df, forecasted_info = predict_missing_values(df)
        
        # Phát hiện bất thường
        df = detect_anomalies(df)
        
        # Chuẩn bị kết quả
        result = {
            "status": "success",
            "data": df.to_dict(orient='records'),
            "forecasted_info": forecasted_info,
            "anomaly_stats": {
                "total_records": len(df),
                "normal": len(df[df["anomaly"] == 1]),
                "anomaly": len(df[df["anomaly"] == -1])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)