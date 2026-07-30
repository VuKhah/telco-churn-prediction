"""Evaluation utilities tái sử dụng cho dự án Telco Customer Churn.

Dùng ở `03_modeling.ipynb` (so sánh model, chọn threshold theo chi phí) và có thể tái
sử dụng ở `04_explainability.ipynb` khi cần tính lại metrics/chi phí cho model cuối cùng.

Quy ước:
- `y_proba` luôn là xác suất dự đoán lớp churn (1 = Yes/rời bỏ).
- Các hàm liên quan chi phí (`expected_cost`, `threshold_cost_sweep`,
  `best_threshold_by_cost`) nhận `fn_cost`/`fp_cost` dạng array-like (chi phí riêng theo
  từng khách hàng, vd. dựa trên MonthlyCharges) hoặc số vô hướng (chi phí cố định).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Tính bộ metrics chuẩn cho bài toán churn (imbalanced) tại 1 threshold.

    - `roc_auc`, `pr_auc` (average precision): không phụ thuộc threshold, đo khả năng
      phân tách tổng thể của model — dùng để so sánh/xếp hạng các model.
    - `precision`/`recall`/`f1`/`accuracy`: phụ thuộc threshold được chọn.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def _as_cost_array(cost, n: int) -> np.ndarray:
    if hasattr(cost, "__len__"):
        return np.asarray(cost)
    return np.full(n, cost, dtype=float)


def expected_cost(y_true, y_proba, threshold: float, fn_cost, fp_cost) -> float:
    """Tổng chi phí kỳ vọng tại 1 threshold cho trước.

    - False Negative (bỏ sót khách sẽ churn, không can thiệp): tốn `fn_cost` (mất khách
      hàng — doanh thu tương lai bị mất).
    - False Positive (dự đoán nhầm khách sẽ churn, can thiệp không cần thiết): tốn
      `fp_cost` (chi phí ưu đãi giữ chân lãng phí).
    True Positive/True Negative không tính thêm chi phí trong khuôn khổ đơn giản hoá này
    (giả định ưu đãi cho đúng khách sắp churn là khoản đầu tư hợp lý, không phải "chi phí").
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    fn_cost = _as_cost_array(fn_cost, len(y_true))
    fp_cost = _as_cost_array(fp_cost, len(y_true))

    is_fn = (y_true == 1) & (y_pred == 0)
    is_fp = (y_true == 0) & (y_pred == 1)
    return float(fn_cost[is_fn].sum() + fp_cost[is_fp].sum())


def threshold_cost_sweep(y_true, y_proba, fn_cost, fp_cost, thresholds=None) -> pd.DataFrame:
    """Quét nhiều threshold, trả về DataFrame (threshold, total_cost, precision, recall, f1)."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        m = evaluate_predictions(y_true, y_proba, threshold=t)
        cost = expected_cost(y_true, y_proba, t, fn_cost, fp_cost)
        rows.append(
            {
                "threshold": t,
                "total_cost": cost,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        )
    return pd.DataFrame(rows)


def best_threshold_by_cost(y_true, y_proba, fn_cost, fp_cost, thresholds=None) -> tuple[float, pd.DataFrame]:
    """Trả về (threshold tối ưu theo chi phí, DataFrame kết quả sweep đầy đủ)."""
    df = threshold_cost_sweep(y_true, y_proba, fn_cost, fp_cost, thresholds)
    best_row = df.loc[df["total_cost"].idxmin()]
    return float(best_row["threshold"]), df


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Ước lượng sai số chuẩn (SE) của ROC-AUC theo công thức Hanley & McNeil (1982).

    Dùng để trả lời câu hỏi "chênh lệch ROC-AUC giữa 2 model trên cùng 1 validation set
    có ý nghĩa thống kê hay chỉ là nhiễu do cỡ mẫu?" — nếu chênh lệch nhỏ hơn nhiều so
    với SE này thì không đủ căn cứ để kết luận model có AUC cao hơn thực sự tốt hơn.
    Đây là ước lượng gần đúng (coi 2 model độc lập, bỏ qua tương quan do đánh giá trên
    cùng 1 tập dữ liệu) — bảo thủ theo hướng an toàn: tương quan dương giữa 2 model thực
    tế sẽ làm SE của phần CHÊNH LỆCH nhỏ hơn nữa, nên nếu công thức này đã kết luận
    "không có ý nghĩa" thì kết luận đó càng chắc chắn đúng.
    """
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    variance = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(variance))


def get_feature_importance(model, feature_names) -> pd.Series:
    """Trích xuất feature importance/coefficient từ model đã fit, chuẩn hoá về Series.

    Hỗ trợ: model có `feature_importances_` (tree-based) hoặc `coef_` (linear, kể cả khi
    nằm trong `sklearn.pipeline.Pipeline` — sẽ tự lấy step cuối cùng).
    """
    est = model
    if hasattr(model, "steps"):  # sklearn Pipeline
        est = model.steps[-1][1]

    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(est.coef_).ravel()
    else:
        raise ValueError(f"Model {type(est)} không có feature_importances_ hoặc coef_")

    return pd.Series(values, index=feature_names).sort_values(ascending=False)
