# Telco Customer Churn
# Tiến độ dự án

> File theo dõi tiến độ theo ngày, cập nhật liên tục trong suốt quá trình làm.
> Xem `README.md` để biết kế hoạch 7 ngày đầy đủ. Mỗi mục dưới đây theo cùng cấu trúc:
> Học về / Kết quả.

## Trạng thái tổng quan

| Ngày | Nội dung |
|---|---|
| 1-2 | Setup + EDA (`01_eda.ipynb`) |
| 3 | Feature engineering (`02_feature_engineering.ipynb`) |
| 4-5 | Modeling & đánh giá (`03_modeling.ipynb`) |
| 6 | SHAP & đề xuất kinh doanh (`04_explainability.ipynb`) |
| 7 | Báo cáo cuối, hoàn thiện README |
| 8 | Sửa lỗi data dredging ở segment EDA |
| 9 | Sensitivity analysis giả định chi phí FN:FP |
| 10 | Cross-validation thực sự, kiểm định thống kê giữa các mô hình |

## Chi tiết theo mốc đã làm

### Ngày 1-2: Setup + EDA (`01_eda.ipynb`)
- **Học về:** dựng môi trường cho dự án, lấy dữ liệu qua API. Cách đo mức ảnh hưởng của từng yếu tố lên churn bằng kiểm định thống kê thay vì chỉ biểu đồ, và cách gộp nhiều yếu tố rủi ro thành một nhóm khách cụ thể.
- **Kết quả:**
  - 7.043 khách hàng, churn rate tổng 26,5%. 
  - Yếu tố ảnh hưởng mạnh nhất: loại hợp đồng. 
  - Tìm thấy nhóm khoảng 410 khách hàng (5,8%) có churn 76%, gấp gần 3 lần trung bình.

### Ngày 3: Feature engineering (`02_feature_engineering.ipynb`)
- **Học** cách chuẩn bị dữ liệu trước khi vào mô hình, vì sao cần tách riêng 3 tập
  dữ liệu để đánh giá khách quan, và cách xử lý dữ liệu mất cân bằng mà không làm méo dữ liệu.
- **Kết quả:** dữ liệu chia 60/20/20, giữ tỷ lệ churn ~26,5% ở cả 3 tập.

### Ngày 4-5: Modeling & đánh giá (`03_modeling.ipynb`)
- **Học** cách so sánh nhiều mô hình công bằng, không dùng chung 1 tập dữ liệu để vừa chọn mô hình vừa báo cáo kết quả. Cách chọn ngưỡng dự đoán theo chi phí thực tế thay vì mặc định.
- **Kết quả:** ROC-AUC 0,845 trên test set. Ngưỡng tối ưu 0,15 tiết kiệm ước tính 84,7%
  chi phí so với không dùng mô hình.

### Ngày 6: SHAP & đề xuất kinh doanh (`04_explainability.ipynb`)
- **Học** cách giải thích dự đoán mô hình cho từng khách hàng cụ thể, hiểu yếu tố nào kéo rủi ro lên, và từ kết quả đưa ra đề xuất chăm sóc khách hàng.
- **Kết quả:** xác nhận lại các yếu tố ảnh hưởng churn mạnh nhất từ bước EDA (cước hàng tháng, Fiber optic, thời gian gắn bó, hợp đồng dài hạn), khớp qua 3 cách phân tích độc
  lập.

### Ngày 7: Báo cáo cuối, README
- **Kết quả:** Viết bản tóm tắt kinh doanh hoàn chỉnh.

### Ngày 8: Sửa lỗi data dredging ở segment EDA
- **Vấn đề:** Một lỗi phân tích dữ liệu: tìm nhóm rủi ro bằng cách dò trên toàn bộ dữ liệu rồi báo cáo ngay trên chính dữ liệu đó. 
- **Sửa:** tách riêng dữ liệu định nghĩa, xác nhận, và kiểm tra cuối.
- **Kết quả:** nhóm 410 khách hàng được xác thực lại, churn ổn định 75-79% qua cả 3 tập,
  kèm khoảng tin cậy rõ ràng.

### Ngày 9: Sensitivity analysis giả định chi phí FN:FP
- **Tìm hiểu lại** cách kiểm tra một kết luận có phụ thuộc quá nhiều vào 1 giả định kinh doanh không, bằng cách thử nhiều kịch bản chi phí khác nhau.
- **Kết quả:** kết luận "nên dùng ngưỡng thấp hơn hẳn 0,5" vẫn đúng ở nhiều kịch bản. Chỉ con số tiết kiệm cụ thể dao động tuỳ giả định.

### Ngày 10: Cross-validation thực sự, kiểm định thống kê giữa các mô hình
- **Học về:** cách đánh giá mô hình đáng tin cậy hơn bằng kiểm định chéo nhiều lần, và
  cách kiểm tra 2 mô hình có khác biệt thật sự hay chỉ ngẫu nhiên.
- **Kết quả:** kết quả cuối gần như không đổi (ROC-AUC test 0,8452), củng cố quyết định
  chọn Logistic Regression vì đơn giản, dễ giải thích.
