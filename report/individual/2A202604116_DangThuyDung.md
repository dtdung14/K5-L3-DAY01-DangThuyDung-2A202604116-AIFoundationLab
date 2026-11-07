# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202604116**
- **Họ và tên: Đặng Thùy Dung**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

Phương pháp	F1 Test	F1 SUB 40	p95 (ms/tin)	USD / 1k tin
1. BoW + Logistic Regression	0.868	0.667	0.11	$0.00
2. BoW + MLP (1 hidden layer)	0.852	0.667	0.15	$0.00
3a. Encoder (MiniLM) + kNN	0.875	0.820	4.20	$0.00
3b. Encoder (MiniLM) + LogReg	0.912	0.885	0.25	$0.00
3c. Encoder (MiniLM) + MLP	0.910	0.885	0.32	$0.00


## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Mô hình tụt nhiều nhất: Bag-of-Words + Logistic Regression sụt giảm F1 thê thảm nhất ở nhóm case khó SUB=40 (chỉ đạt 0.667 so với 0.885 của Embedding). Nhóm case này tập trung vào các tin nhắn cố tình viết né từ khóa (Spam ẩn) hoặc chứa từ nhạy cảm nhưng vô hại (Ham bẫy).

• Case cụ thể 1 (Ham bẫy): "Hey are you free for lunch tomorrow? My treat"
	• Cơ chế sai của BoW: Đếm từ thấy chữ "free" - từ khóa mang trọng số dương cực lớn của lớp Spam - nên lập tức báo động giả và chặn nhầm tin thường.


## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

• Thiết kế 4 ca kiểm thử bẫy ngữ nghĩa nâng cao (2 spam né từ khóa bằng từ đồng nghĩa/kịch bản lừa đảo; 2 ham chứa từ nhạy cảm free/win nhưng bối cảnh an toàn).
• Cách làm: Viết trực tiếp mã nguồn mảng MY_CHALLENGE_CASES, truyền qua bộ trích xuất vector của encoder MiniLM, áp dụng chuẩn hóa Z-score rồi chạy kiểm thử song song trên hai đầu phân loại BoW và Encoder + LogReg (3b).
• Kết quả: BoW chỉ đúng 2/4 ca (bị lừa chặn nhầm tin thường và lọt lưới tin rác ngân hàng). Encoder + LogReg đạt độ chính xác tuyệt đối 4/4.
• Giải thích:
	• Với tin spam né từ khóa giả danh shipper lấy OTP hoặc khóa thẻ debit, Encoder bắt được bản chất hành vi chiếm đoạt thông tin nên chặn đứng.
	• Với tin ham chứa chữ free (tặng sách giáo trình ở thư viện đại học) và chữ win (chúc mừng đội bóng vô địch), Encoder nhận diện bối cảnh giáo dục và thể thao an toàn nên không chặn nhầm, khắc phục hoàn toàn điểm yếu đếm từ thô của BoW.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
• Phương án chọn: 3b. Encoder (MiniLM) + LogReg.
• Dẫn chứng số liệu:
	• Mô hình đạt chỉ số F1 Test vượt trội (0.912) và xử lý các ca bẫy ngữ nghĩa rất tốt (F1 SUB = 0.885).
	• Về tốc độ, thời gian suy luận p95 cực kỳ ấn tượng chỉ mất 0.25 ms/tin, bỏ xa mức trần cho phép 50 ms của nhà mạng.
	• Về tài chính: Chi phí vận hành tối ưu 0 USD / 1k tin do mô hình MiniLM hoàn toàn chạy local (on-premise) trên hạ tầng CPU có sẵn, không mất tiền mua token API đắt đỏ của bên thứ ba ($10.50/ngày nếu dùng LLM như GPT).
• Chấp nhận đánh đổi: Chấp nhận một khoản chi phí đầu tư hạ tầng phần cứng ban đầu (RAM tối thiểu để tải mô hình Encoder nặng khoảng 100MB lên bộ nhớ liên tục) và tốn thêm một khoảng thời gian nhỏ (~0.05ms) cho bước mã hóa văn bản thô thành vector trước khi đưa vào hàm tính toán tuyến tính của LogReg.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
• Lượt 1: Hỏi cách tối ưu hàm vectorize rồi Nhận được vòng lặp dùng set() rồi Sửa lại bằng cách tích hợp thêm bộ lọc biểu thức chính quy re.findall(r'\w+') để loại bỏ triệt để dấu câu gây fail test bài học.
• Lượt 2: Hỏi cách xử lý lỗi _lower rồi Nhận ra mình gõ nhầm thuộc tính ẩn text._lower() \ rồi Chỉnh sửa ngay về phương thức chuẩn của Python là text.lower().
• Lượt 3: Hỏi cách tính toán đạo hàm ma trận trong 5 dòng PyTorch rồi Nhận được khối lệnh chuẩn rồi Sửa lại cấu trúc mảng bằng lệnh .squeeze(1) nhằm đồng bộ kích thước chiều vector giữa đầu ra và nhãn thật.
• Lượt 4 (Tối ưu hóa phép toán ma trận không gian nhúng): Hỏi cách tìm kiếm hàng xóm gần nhất hiệu năng cao cho thuật toán kNN mà không dùng vòng lặp Python duyệt tuyến tính qua 3.600 dòng \(\rightarrow \) Nhận được kỹ thuật vector hóa nâng cao bằng cách cấu hình axis=1, keepdims=True trong hàm np.linalg.norm \(\rightarrow \) Triển khai phép nhân tích vô hướng trực tiếp np.dot trên hai ma trận đã chuẩn hóa để ép toàn bộ luồng chạy xuống tầng mã máy C, tối ưu độ trễ xử lý.

 Trước đây mình luôn nghĩ mạng nơ-ron sâu hay Deep Learning phức tạp thì lúc nào cũng tốt hơn mô hình toán cổ điển. Hôm nay mình hiểu ra rằng bản chất của dữ liệu (Data Representation) mới quyết định tất cả. Khi dữ liệu đã được xử lý thành một không gian nhúng (Embedding) giàu ngữ nghĩa và phân tách quá tốt, một thuật toán tuyến tính đơn giản như Logistic Regression vừa chạy nhanh gấp hàng chục lần, vừa không bị Overfitting, lại mang lại hiệu quả vượt trội so với các mạng nơ-ron tầng ẩn phức tạp.
