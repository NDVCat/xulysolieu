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

# Load models từ tệp .pkl
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

    # Đảm bảo thứ tự cột đúng với khi huấn luyện
    df = df.reindex(columns=EXPECTED_COLUMNS)

    # Chuyển đổi kiểu dữ liệu
    for col in EXPECTED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Lưu lại vị trí các giá trị bị thiếu
    missing_positions = {col: df[df[col].isnull()].index.tolist() for col in EXPECTED_COLUMNS}

    # Sử dụng mô hình để xử lý giá trị thiếu
    for target_col, model in models.items():
        missing_rows = df[df[target_col].isnull()]
        if not missing_rows.empty:
            print(f"🔍 Đang xử lý giá trị thiếu cho {target_col}...")
            print("📊 Input cho mô hình:", missing_rows[EXPECTED_COLUMNS])
            filled_values = model.predict(missing_rows[EXPECTED_COLUMNS])
            df.loc[missing_rows.index, target_col] = filled_values

    return df, missing_positions


# 📌 API xử lý file CSV đầu vào và dự đoán kết quả
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if not request.data:
            return jsonify({"error": "No CSV data provided"}), 400

        csv_data = request.data.decode('utf-8')
        print("📥 Received CSV Data:\n", csv_data[:500])

        # Kiểm tra dữ liệu hợp lệ
        if not csv_data.strip():
            return jsonify({"error": "Empty CSV data received"}), 400

        # Đọc dữ liệu CSV
        try:
            df = pd.read_csv(io.StringIO(csv_data), encoding='utf-8-sig', skip_blank_lines=True)
        except Exception as e:
            return jsonify({"error": f"CSV Parsing Error: {str(e)}"}), 400

        print("📊 Parsed DataFrame:\n", df.head())

        # Tiền xử lý dữ liệu
        df, missing_positions = preprocess_input(df)

        # Chuyển đổi dữ liệu dự đoán thành JSON
        response = {
            'message': 'CSV processed successfully',
            'predictions': df.to_dict(orient='records'),
            'missing_positions': missing_positions
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
