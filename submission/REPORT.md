# Báo cáo Day 09 — Lương Hoàng Bảo Trân, 2A202603077

## 1. Kết luận trước

Pipeline nhận 193/224 ảnh, loại 31 ảnh ở intake. Sau mô phỏng duyệt 80 ảnh, kết luận là **Hold**: 57/193 ảnh còn cảnh báo (29,53%), cao hơn ngưỡng 10%. Dataset `clean_v1` chỉ được xuất làm bản nháp để thử nghiệm, chưa được phát hành.

## 2. Lỗi và phân việc

Ở lần kiểm đầu, `02_error_list_v1` ghi 67 lỗi bắt buộc và 385 cảnh báo. Trong đó intake loại 31 ảnh; 36 lỗi bắt buộc và 385 cảnh báo thuộc 193 ảnh được nhận vào pipeline.

| Mã lỗi | Số lỗi/dòng | Xử lý |
|---|---:|---|
| `CORRUPT_IMAGE` / `BAD_SIZE` | 5 | Máy loại tại intake |
| `LEAK_TEST` / `DUP_IMAGE_EXACT` / `DUP_IMAGE_NEAR` | 26 | Máy loại ảnh rò rỉ/trùng tại intake |
| `MISSING_LABEL_FILE` | 6 | Cần người gán nhãn; máy không tự bịa nhãn |
| `BAD_CLASS_ID`, `BAD_CLASS_NAME`, `BAD_FORMAT`, `BOX_OUT_OF_IMAGE`, `BOX_ZERO`, `DUP_BOX` | 30 | Máy tự sửa được một phần; 35 thay đổi được ghi vào `03_fixes.csv` |
| `BOX_HUGE`, `BOX_TINY`, `CLASS_RARE`, `EMPTY_LABEL` | 24 | Cảnh báo cần người xem ngữ nghĩa |
| `POSSIBLE_FALSE_POS`, `POSSIBLE_LOOSE_BOX`, `POSSIBLE_MISSING`, `POSSIBLE_WRONG_CLASS` | 361 | Máy chỉ cảnh báo; người cần xác minh nhãn/model |

Hàng đợi có 142 ảnh; mô phỏng đã duyệt 80 ảnh theo tổng điểm ưu tiên giảm dần. Điểm cộng theo trọng số từng cảnh báo và các sửa cần xác nhận (`drop_line`, `clip_box`, `needs_label`); vì vậy `img_0160` đứng đầu với priority 109.727 do có cảnh báo thiếu box và sai lớp với độ tin cậy cao. Kết quả này là mô phỏng, không thay cho việc người thật xem ảnh.

## 3. Calibration (2 ngưỡng)

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do |
|---|---:|---:|---|---|
| `duplicates.phash_dist` | 2 | 16 | `DUP_IMAGE_NEAR`: 1.00 / 0.50 → 1.00 / 0.83; `DUP_IMAGE_EXACT`: 1.00 / 1.00 → 1.00 / 1.00 | Bắt thêm ảnh gần trùng mà vẫn giữ precision 1.00; mức 20 làm recall của mã `DUP_IMAGE_EXACT` giảm còn 0.33 do các nhóm trùng chồng nhau và thứ tự ưu tiên phân loại. |
| `second_opinion.conf` | 0.25 | 0.1 | `POSSIBLE_MISSING`: 0.50 / 1.00 → 0.40 / 1.00; `POSSIBLE_LOOSE_BOX`: 0.29 / 0.40 → 0.38 / 0.60 | Ưu tiên recall cho cảnh báo để người xem; chấp nhận nhiều cảnh báo giả hơn. |

Điểm có thể gây lệch: calib chỉ có 6 lỗi gần trùng nên recall 0.83 là 5/6 và còn sót một lỗi. Ở `conf=0.1`, `POSSIBLE_MISSING` có 12 FP so với 8 FP tại 0.25; kết quả trên 39 ảnh không đảm bảo phản ánh raw.

## 4. Cửa chặn và điểm mù

Chạy pipeline với ngân sách duyệt 0 cho kết luận **Rework**: còn 6 lỗi bắt buộc và cảnh báo ở 131/193 ảnh (67,88%); `work/exp_nb` không được xuất. Trong lượt pipeline chuẩn, mô phỏng duyệt 80 ảnh làm hết lỗi bắt buộc nhưng vẫn còn 57/193 ảnh cảnh báo, nên kết quả là **Hold**.

Hai giới hạn có số liệu: (1) với `phash_dist=16`, checker tìm được 5/6 ảnh gần trùng trong calib, vẫn bỏ sót 1; các biến đổi ảnh ngoài những dạng đã chèn có thể lọt. (2) `second_opinion` không phải đáp án chuẩn: ở `conf=0.1`, `POSSIBLE_FALSE_POS` bắt đủ 5 lỗi chèn nhưng có 11 FP, nên bất đồng với model cần người kiểm chứng thay vì tự động sửa.

Ghi nhận khi xem 10 ảnh đã vẽ nhãn:

| Ảnh | Quan sát |
|---|---|
| `img_0001` | Box khít và nhãn `person` đúng; có 1 người nhìn thấy nhưng thiếu box. Không thấy trùng hoặc vấn đề box nhỏ/lớn, cắt hay tràn ảnh. |
| `img_0030` | Có box không bao hết vật; nhãn `person` đúng. Ghi nhận 5 người thiếu box và 1 box quá nhỏ. Không thấy trùng. |
| `img_0058` | Có box không bao đúng vật, 1 vật thiếu box và 1 box thừa; box thừa cũng có thể là box quá nhỏ. Không thấy trùng. |
| `img_0074` | Một box ở mép phải không bao đúng vật, một box khác lệch khỏi vật; nhãn đúng. Không thấy trùng. |
| `img_0099` | Người đàn ông ở mép trái ảnh không được đánh box; không thấy ảnh trùng. |
| `img_0127` | Box 1 gán `person` nhưng không khít vật; còn nhiều người khác không được đánh box. Không thấy ảnh trùng. |
| `img_0160` | Có 1 nhãn không khớp thực tế và ghi nhận thêm vật cần chú ý; không trùng, không có vấn đề kích thước box. |
| `img_0183` | Có 1 nhãn sai; không trùng, không có vấn đề kích thước box. |
| `img_0204` | Có box lệch; nhãn đúng, không ghi nhận vật thiếu box hay ảnh trùng, không có vấn đề kích thước box. |
| `img_0220` | Một xe `car` bị gán nhãn `truck`; người ở mép trái và xe bị khuất sau xe màu cam không được đánh box. Không thấy ảnh trùng. |

Các ví dụ nhìn trực tiếp cho thấy điểm mù của kiểm tra tự động: thiếu người ở `img_0001`/`img_0030`/`img_0099`, nhãn xe sai (`car` thành `truck`) và vật thể bị khuất hoặc ở mép ảnh thiếu box trong `img_0220`, nhãn sai trong `img_0160`/`img_0183`, và box lệch hoặc không khít vật trong `img_0058`/`img_0074`/`img_0127`/`img_0204`. Đây là lý do cảnh báo của model cần người xác minh.

## 5. Train: raw vs clean (tùy chọn)

Chưa chạy train Colab/GPU; chưa có kết quả trong `submission/results/`. Vì vậy chưa thể kết luận clean có cải thiện mAP so với raw hay không.

## 6. Quyết định và ghi chú AI

Đề xuất hiện tại là **chưa phát hành**: cần người thật xem các cảnh báo còn lại, xác minh các nhãn cần gán/sửa, chạy lại pipeline rồi chỉ phát hành nếu cổng cho Release. AI đã hỗ trợ triển khai các module Part 2, tổng hợp calibration và điền báo cáo từ `submission/reports/`; pipeline và test là bằng chứng kỹ thuật, không thay thế xác nhận hình ảnh của người gán nhãn.

**Bạn cần làm thủ công trước khi nộp:** kiểm tra họ tên/MSHV ở tiêu đề README và báo cáo; xem lại các cặp ảnh bị báo trùng; nếu muốn lấy điểm train, chạy notebook Colab và bổ sung bảng kết quả. Sau các bổ sung, chạy `make check-core` (và `make check` nếu đã train).
