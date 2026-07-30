# Tóm tắt kinh doanh 
# Dự đoán khách hàng rời bỏ (Churn)

## Vấn đề

Công ty viễn thông đang mất khoảng 26,5% khách hàng mỗi kỳ do rời bỏ dịch vụ. Giữ
chân một khách hàng hiện tại luôn rẻ hơn nhiều so với tìm khách hàng mới, nhưng đội CSKH
không có đủ ngân sách để ưu đãi cho tất cả mọi người. 

Câu hỏi đặt ra: nên ưu tiên ngân sách
giữ chân cho ai, và làm gì với từng nhóm?

## Phát hiện chính

**1. Một nhóm nhỏ khách hàng chiếm phần lớn rủi ro.**
Chỉ 5,8% khách hàng (410 người), là những khách hợp đồng theo tháng, dùng internet cáp quang (Fiber optic), thanh toán bằng check điện tử, và mới gia nhập chưa đầy 6 tháng, có tỷ lệ rời bỏ khoảng 76-79%, gấp gần 3 lần mức trung bình toàn công ty. Con số này đã được kiểm tra chéo trên 3 nhóm dữ liệu độc lập để tránh trường hợp "nhìn thấy" đúng nhóm khách hàng khớp. Kết quả ổn định (75%, 75%, 79%) ở cả 3 nhóm. Có thể xuất danh sách này ngay cho đội CSKH liên hệ ưu tiên.

**2. Những tháng đầu tiên là giai đoạn nguy hiểm nhất.**
Khách hàng mới (dưới 6 tháng) có tỷ lệ rời bỏ 52,9%, cao hơn hẳn khách đã gắn bó lâu (chỉ 9,5% ở nhóm trên 4 năm). Đây là cửa sổ thời gian quan trọng để chăm sóc, tặng ưu đãi, và thuyết phục khách ký hợp đồng dài hạn hơn.

**3. Đăng ký thêm dịch vụ đi kèm (add-on) giúp giữ chân khách hàng.**
Khách hàng đăng ký thêm các dịch vụ bảo mật/hỗ trợ kỹ thuật có xu hướng ở lại lâu hơn. Đây là cơ hội bán thêm (upsell) vừa tăng doanh thu, vừa giảm rủi ro rời bỏ.

**4. Có công cụ dự đoán trước ai sắp rời bỏ, với độ tin cậy đã kiểm chứng.**
Một mô hình máy học dự đoán trước xác suất một khách hàng sẽ rời bỏ, dựa trên hồ sơ sử
dụng dịch vụ của họ.
## Giá trị kinh doanh ước tính

Nếu công ty dùng mô hình này để quyết định ai cần được chăm sóc/ưu đãi, thay vì chăm sóc dàn trải tất cả khách, chi phí giữ chân ước tính giảm:
- 84,7% so với không có công cụ hỗ trợ.
- 35,9% so với cách tiếp cận mặc định (coi mọi khách có trên 50% khả năng rời bỏ). Bài toán này cần ưu tiên bắt được càng nhiều khách sắp rời bỏ càng tốt,
  chấp nhận thi thoảng chăm sóc nhầm một số khách không thực sự có ý định rời đi, vì chi
  phí chăm sóc nhầm rẻ hơn nhiều so với chi phí mất một khách hàng.

## Đề xuất hành động

| Ưu tiên | Đối tượng | Hành động |
|---|---|---|
| 1 (cao nhất) | 410 khách khớp đủ 4 điều kiện rủi ro (xem trên) | Liên hệ trực tiếp, ưu đãi 1 tháng cước miễn phí đổi cam kết hợp đồng 1 năm |
| 2 | Mọi khách hợp đồng theo tháng, trong 3 tháng đầu | Quy trình chăm sóc/ưu đãi chủ động ở giai đoạn onboarding |
| 3 | Khách dùng Fiber optic nói chung | Khảo sát chất lượng dịch vụ, đối chiếu với đối thủ cạnh tranh |
| 4 | Khách cước cao, chưa dùng dịch vụ đi kèm | Upsell bảo mật/hỗ trợ kỹ thuật, tăng doanh thu và giảm rủi ro rời bỏ |

## Giới hạn 

Đây là phân tích trên dữ liệu lịch sử ở một thời điểm, dựa trên giả định chi phí đơn giản
hoá.
---
*Chi tiết kỹ thuật đầy đủ: xem `README.md` và các notebook trong `notebooks/`
(`01_eda.ipynb` → `02_feature_engineering.ipynb` → `03_modeling.ipynb` →
`04_explainability.ipynb`).*
