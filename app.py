from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import io

app = Flask(__name__)

# 📌 Định nghĩa các cột đầu vào cần thiết
EXPECTED_COLUMNS = ['Qgas', 'Qwater', 'Oilrate', 'LiqRate', 'DayOn']

# 📌 Tải tất cả mô hình từ tệp duy nhất
MODEL_FILE = "reverse_prediction_models.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"⚠️ Không tìm thấy tệp '{MODEL_FILE}'!")

# Load models từ tệp .pkl
try:
    models = joblib.load(MODEL_FILE)
    if not isinstance(models, dict):
        raise ValueError("⚠️ Dữ liệu trong tệp không phải là dictionary chứa các model!")
except Exception as e:
    raise RuntimeError(f"❌ Lỗi khi tải mô hình từ '{MODEL_FILE}': {e}")

print(f"✅ Đã tải {len(models)} mô hình:", list(models.keys()))


# 📌 Hàm tiền xử lý dữ liệu đầu vào
def preprocess_input(df):
    # Điền giá trị thiếu cho cột số
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().all():
            df[col] = 0.0  # Nếu toàn bộ cột không có giá trị, điền 0
        else:
            df[col] = df[col].fillna(df[col].mean())  # Điền giá trị trung bình nếu có dữ liệu

    # Điền giá trị thiếu cho cột dạng object
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')

    # Đảm bảo các cột EXPECTED_COLUMNS có trong dataframe
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0  # Nếu thiếu cột nào, điền mặc định là 0

    df[EXPECTED_COLUMNS] = df[EXPECTED_COLUMNS].astype(float)
    return df


# 📌 API xử lý file CSV đầu vào và dự đoán kết quả
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if not request.data:
            return jsonify({"error": "No CSV data provided"}), 400

        csv_data = request.data.decode('utf-8')  
        print("📥 Received CSV Data:\n", csv_data[:500])  # Log giới hạn 500 ký tự

        # Kiểm tra dữ liệu có phải CSV hợp lệ không
        if not csv_data.strip():
            return jsonify({"error": "Empty CSV data received"}), 400

        # Thử đọc CSV
        try:
            df = pd.read_csv(io.StringIO(csv_data), encoding='utf-8-sig', skip_blank_lines=True)
        except Exception as e:
            return jsonify({"error": f"CSV Parsing Error: {str(e)}"}), 400

        print("📊 Parsed DataFrame:\n", df.head())

        df = preprocess_input(df)

        # 📌 Dự đoán dữ liệu mới
        predictions = {}
        for target_col, model in models.items():
            try:
                feature_df = df[EXPECTED_COLUMNS]  
                print(f"🔹 Dự đoán giá trị cho {target_col}...")
                df[f'Predicted_{target_col}'] = model.predict(feature_df)
                predictions[target_col] = df[f'Predicted_{target_col}'].tolist()
            except Exception as e:
                return jsonify({'error': f'Model prediction error for {target_col}: {str(e)}'}), 500

        # Chuyển đổi dữ liệu sang CSV string
        response = {
            'message': 'CSV processed successfully',
            'predictions': df.to_dict(orient='records')  # Trả về danh sách JSON
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
