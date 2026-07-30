"""Feature engineering tái sử dụng cho dự án Telco Customer Churn.

Dùng chung giữa các notebook (02_feature_engineering, 03_modeling, 04_explainability)
để tránh copy-paste logic xử lý dữ liệu ở nhiều nơi.

Quy ước:
- `customerID` được set làm index xuyên suốt (không đưa vào feature matrix), giữ lại để
  tra cứu/giải thích cho từng khách hàng cụ thể ở bước SHAP.
- Các cột dịch vụ add-on có giá trị "No internet service" / "No phone service" được gộp
  về "No" vì bản chất là phụ thuộc vào InternetService/PhoneService, không phải hạng mục
  độc lập (xem data dictionary trong 01_eda.ipynb).
"""
from __future__ import annotations

import pandas as pd

RAW_DATA_PATH = "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Các cột dịch vụ add-on có giá trị "No internet service"
ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
# Cột có giá trị "No phone service"
PHONE_DEPENDENT_COLS = ["MultipleLines"]

TENURE_BINS = [0, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]

# Sau khi gộp "No internet/phone service" -> "No", các cột này chỉ còn Yes/No
BINARY_YESNO_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
# Cột nominal nhiều hạng mục -> one-hot encode
NOMINAL_COLS = ["InternetService", "Contract", "PaymentMethod", "tenure_bucket"]

# Cột continuous/count trong X đã encode (dùng cho SMOTENC — phần còn lại của X đều là
# categorical/binary, không nên nội suy tuyến tính như SMOTE thường làm).
CONTINUOUS_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "NumAddonServices"]


def get_categorical_feature_indices(X: pd.DataFrame) -> list[int]:
    """Trả về vị trí cột (index) của các feature categorical/binary trong X đã encode.

    Dùng cho `SMOTENC` (imbalanced-learn). Lý do cần hàm này: `SMOTE` thường nội suy
    tuyến tính trên MỌI cột kể cả cột one-hot (0/1) -> tạo giá trị phân số phi thực tế
    (vd. 0.3 cho `Contract_Two year`), và khi ép kiểu dtype gốc (int) sẽ bị TRUNCATE về
    0 một cách có hệ thống (thiên lệch dữ liệu tổng hợp). `SMOTENC` xử lý đúng bằng cách
    lấy mode của k-neighbors cho các cột categorical thay vì nội suy.
    """
    categorical_cols = [c for c in X.columns if c not in CONTINUOUS_COLS]
    return [X.columns.get_loc(c) for c in categorical_cols]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc CSV gốc từ Kaggle, chưa xử lý gì."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges sang numeric, impute missing, set customerID làm index."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # 11 dòng missing đều là khách hàng mới (tenure=0) -> chưa phát sinh phí, impute = 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df = df.set_index("customerID")
    return df


def consolidate_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Gộp 'No internet service' / 'No phone service' về 'No' để giảm hạng mục thừa."""
    df = df.copy()
    for col in ADDON_COLS:
        df[col] = df[col].replace("No internet service", "No")
    for col in PHONE_DEPENDENT_COLS:
        df[col] = df[col].replace("No phone service", "No")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các feature phái sinh: tenure_bucket, NumAddonServices, AvgMonthlySpend."""
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True
    )
    df["NumAddonServices"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    # Chi tiêu trung bình/tháng tính trên toàn bộ lịch sử — khác với MonthlyCharges hiện tại,
    # hữu ích cho khách hàng đã đổi gói theo thời gian. tenure=0 -> dùng mẫu số 1 để tránh chia 0.
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    return df


def encode_target(df: pd.DataFrame) -> pd.Series:
    """Trả về nhãn churn dạng 0/1 (1 = Yes/rời bỏ)."""
    return (df["Churn"] == "Yes").astype(int)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode toàn bộ feature: binary Yes/No -> 0/1, gender -> 0/1, one-hot cho nominal.

    Giả định df đã qua `clean_data` + `consolidate_service_columns` + `engineer_features`.
    Loại bỏ cột `Churn` (nhãn, xử lý riêng bằng `encode_target`).
    """
    df = df.copy()
    df = df.drop(columns=["Churn"])

    df["gender"] = (df["gender"] == "Female").astype(int)
    for col in BINARY_YESNO_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    df = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=True)
    # get_dummies trả về cột bool -> ép về int cho đồng nhất dtype toàn bộ X
    # (tránh lỗi/quirk dtype với XGBoost/LightGBM/SHAP về sau).
    dummy_cols = [c for c in df.columns if df[c].dtype == bool]
    df[dummy_cols] = df[dummy_cols].astype(int)
    return df


def build_dataset(path: str = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Chạy toàn bộ pipeline, trả về (X, y, df_readable).

    - X: feature matrix đã encode, sẵn sàng cho modeling.
    - y: nhãn churn (0/1).
    - df_readable: bản đã clean + feature engineering nhưng CHƯA encode — dùng để tra cứu/
      hiển thị dễ đọc (vd. khi giải thích SHAP cho một khách hàng cụ thể ở Ngày 6).
    """
    df = load_raw_data(path)
    df = clean_data(df)
    df = consolidate_service_columns(df)
    df = engineer_features(df)

    y = encode_target(df)
    X = encode_features(df)
    return X, y, df
