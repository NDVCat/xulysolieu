from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import io

app = Flask(__name__)

# 📌 Định nghĩa các cột đầu vào cần thiết
EXPECTED_COLUMNS = ['DayOn','Qoil','Qgas','Qwater','GOR','ChokeSize','Press_WH','Oilrate','LiqRate','GasRate']

# 📌 Tải tất cả mô hình từ tệp duy nhất
MODEL_FILE = "reverse_prediction_models.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"⚠️ Không tìm thấy tệp '{MODEL_FILE}'!")

try:
    models = joblib.load(MODEL_FILE)
    if not isinstance(models, dict):
        raise ValueError("⚠️ Dữ liệu trong tệp không phải là dictionary chứa các model!")
except Exception as e:
    raise RuntimeError(f"❌ Lỗi khi tải mô hình: {e}")

print(f"✅ Đã tải {len(models)} mô hình:", list(models.keys()))

# 📌 Hàm tiền xử lý dữ liệu đầu vào
def preprocess_input(df):
    # Chuyển các giá trị không hợp lệ thành NaN
    df.replace({"...": np.nan, "null": np.nan, "NaN": np.nan, "": np.nan}, inplace=True)

    # Đảm bảo tất cả các cột cần thiết đều có trong DataFrame
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Chuyển đổi kiểu dữ liệu
    for col in EXPECTED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Lưu lại vị trí các giá trị bị thiếu
    missing_positions = {col: df[df[col].isnull()].index.tolist() for col in EXPECTED_COLUMNS}

    # Thay thế NaN bằng giá trị trung vị của mỗi cột
    for col in EXPECTED_COLUMNS:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)

    return df, missing_positions

# 📌 API xử lý file CSV đầu vào và dự đoán kết quả
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if not request.data:
            return jsonify({"error": "No CSV data provided"}), 400

        csv_data = request.data.decode('utf-8')
        print("📥 Received CSV Data:\n", csv_data[:500])

        if not csv_data.strip():
            return jsonify({"error": "Empty CSV data received"}), 400

        try:
            df = pd.read_csv(io.StringIO(csv_data), encoding='utf-8-sig', skip_blank_lines=True)
        except Exception as e:
            return jsonify({"error": f"CSV Parsing Error: {str(e)}"}), 400

        print("📊 Parsed DataFrame:\n", df.head())

        # Tiền xử lý dữ liệu và xác định vị trí các giá trị thiếu
        df, missing_positions = preprocess_input(df)

        # 📌 Dự đoán lại các giá trị đã bị thay thế bằng trung vị
        predictions = {}
        for target_col, model in models.items():
            if target_col in missing_positions and missing_positions[target_col]:
                missing_indices = missing_positions[target_col]

                try:
                    feature_df = df.loc[missing_indices, model.feature_names_in_]
                    print(f"🔹 Dự đoán lại giá trị cho {target_col}...")
                    predicted_values = model.predict(feature_df)

                    # Cập nhật giá trị đã dự đoán vào DataFrame
                    df.loc[missing_indices, target_col] = predicted_values

                    predictions[target_col] = predicted_values.tolist()
                except Exception as e:
                    return jsonify({'error': f'Model prediction error for {target_col}: {str(e)}'}), 500

        response = {
            'message': 'CSV processed successfully',
            'predictions': df.to_dict(orient='records')
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
