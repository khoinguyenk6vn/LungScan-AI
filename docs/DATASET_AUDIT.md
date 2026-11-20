# Dataset Audit — kiểm tra dữ liệu ảnh CT phổi

> Ngày kiểm tra: 21/11/2026 · Người phụ trách: Nguyễn Lê Thành (phần dữ liệu)
> Code: `scripts/audit_dataset.py` (kiểm tra), `scripts/build_dataset.py` (tạo bảng + chia fold),
> `tests/test_no_patient_leakage.py` (kiểm tra rò rỉ). Ảnh gốc để ở `data/raw/`, không đưa lên Git.

## 1. Kết luận nhanh

| | Quyết định |
|---|---|
| **Dữ liệu chính** (train / validation / test) | **IQ-OTH/NCCD Patient-Level** — 1.073 ảnh sau khi bỏ trùng, 89 mã bệnh nhân gom thành 82 nhóm |
| **Bài thi phụ** (external test) | Ảnh của bộ **Advanced Imaging (Bangladesh) không có trong IQ-OTH** — 1.476 ảnh, chỉ benign / malignant |
| **Không dùng** | Bộ **Federated Learning** — toàn bộ ảnh là bản sao của 2 bộ trên |

Lý do chính: 3 bộ chép ảnh của nhau rất nhiều. **6.241 file chỉ có 2.553 ảnh khác nhau.** Toàn bộ ảnh *normal* của 2 bộ
phụ là ảnh lấy lại từ IQ-OTH (bệnh viện Iraq), còn ảnh mới thật sự của bộ Bangladesh chỉ có *benign* / *malignant*.
Nếu gộp tất cả, mọi ảnh *normal* đến từ Iraq còn một phần ảnh bệnh đến từ Bangladesh → mô hình có thể học "nhận ra
bệnh viện" thay vì nhận ra tổn thương (shortcut learning). Vì vậy dữ liệu chính chỉ dùng 1 nguồn, nguồn còn lại dùng
để đánh giá khả năng tổng quát hóa.

## 2. Thông tin từng dataset

| Mục | IQ-OTH/NCCD Patient-Level | Advanced Imaging | Federated Learning |
|---|---|---|---|
| Nguồn | [Mendeley dhzdww85gy v1](https://data.mendeley.com/datasets/dhzdww85gy/1) | [Mendeley zr4fddh833 v1](https://data.mendeley.com/datasets/zr4fddh833/1) | [Mendeley 2tp26b6bzw v2](https://data.mendeley.com/datasets/2tp26b6bzw/2) |
| DOI | 10.17632/dhzdww85gy.1 | 10.17632/zr4fddh833.1 | 10.17632/2tp26b6bzw.2 |
| Giấy phép | CC BY 4.0 | CC BY 4.0 | CC BY 4.0 |
| Bệnh viện (theo mô tả) | Iraq (IQ-OTH / NCCD), máy Siemens SOMATOM | NIDCH, Dhaka, Bangladesh | Không ghi rõ; chia 4 "Client" |
| Số file ảnh | 1.102 | 1.933 | 3.206 |
| Normal / Benign / Malignant | 420 / 115 / 567 | 416 / 467 / 1.050 | 986 / 912 / 1.308 |
| Ảnh khác nhau (bỏ trùng nội bộ) | 1.073 | 1.885 | 1.987 |
| Mã bệnh nhân | **Có** (89 bệnh nhân, thư mục riêng + 2 file CSV) | Không | Không |
| Định dạng | JPEG, RGB | JPEG, RGB | JPEG, RGB |
| Kích thước | 512×512 (1.041 ảnh); còn lại 623×512, 801×512, vài ảnh lẻ | 512×512 (1.932 ảnh) | 512×512 (3.204 ảnh) |
| Nhãn | Theo chẩn đoán từng ca (cấp bệnh nhân) | Theo mô tả: bác sĩ chẩn đoán hình ảnh NIDCH gán và kiểm tra | Không ghi rõ |
| Bounding box / mask | Không | Không | Không |
| Ảnh hỏng | 0 | 0 | 0 |
| Vai trò | **Dữ liệu chính** | **Bài thi phụ** (phần không trùng IQ-OTH) | **Không dùng** |

Ghi chú: mô tả trên Mendeley của bộ Federated ghi ảnh 224×224 grayscale, nhưng file thực tế là 512×512 RGB.

## 3. Kết quả kiểm tra trùng lặp (SHA-256, trùng từng byte)

| Phát hiện | Số lượng |
|---|---|
| Ảnh khác nhau trên tổng 6.241 file | 2.553 |
| Ảnh *normal* của Advanced lấy lại từ IQ-OTH | 416 / 416 |
| Ảnh của Federated có trong IQ-OTH | 986 normal + 120 benign |
| Ảnh khác nhau của Advanced cũng có trong Federated | 1.885 / 1.885 |
| Ảnh mới thật sự (không có trong IQ-OTH) | 1.480 (benign 457, malignant 1.023, **normal 0**) |
| Ảnh y hệt nhau nhưng bị gán 2 nhãn (benign ↔ malignant) | 4 (đều thuộc bộ Bangladesh) |
| IQ-OTH: ảnh y hệt nằm ở 2 mã bệnh nhân khác nhau | 29 ảnh, ở 6 nhóm mã |

Các cặp mã bệnh nhân IQ-OTH có chung ảnh (nhiều khả năng là cùng 1 người bị tách mã):
PatMA22–PatMA26 (6 ảnh), PatMA23–PatMA27 (8), PatNO04–PatNO32 (2), PatNO18–PatNO33–PatNO34 (2 + 2 + 4),
PatNO19–PatNO31 (2), PatNO24–PatNO35 (3).

IQ-OTH khớp hoàn toàn với `Dataset_Metadata.csv` (1.102 dòng) và `Patient_Summary.csv` (89 bệnh nhân):
không thiếu ảnh, không thừa ảnh, không sai mã bệnh nhân hay nhãn.

**Giới hạn:** chỉ phát hiện bản sao y hệt. Ảnh bị nén hoặc resize lại sẽ không bị bắt. Đã thử dHash 64 bit nhưng
với ảnh CT (nền đen, hình dạng giống nhau) báo nhầm quá nhiều nên không dùng.

## 4. Xử lý và chia dữ liệu (`scripts/build_dataset.py`)

**Dữ liệu chính — IQ-OTH:**
1. Bỏ 29 ảnh lặp (giữ 1 bản) → 1.073 ảnh.
2. Các mã bệnh nhân có chung ảnh được gom thành 1 nhóm (`group_id`) → 82 nhóm. Cả nhóm luôn nằm chung 1 fold.
3. Chia **5 fold phân tầng theo nhãn**, đơn vị chia là nhóm bệnh nhân, `seed = 42`.

| Fold | Normal (nhóm / ảnh) | Benign (nhóm / ảnh) | Malignant (nhóm / ảnh) |
|---|---|---|---|
| 0 | 7 / 84 | 3 / 23 | 6 / 121 |
| 1 | 7 / 83 | 3 / 24 | 7 / 110 |
| 2 | 7 / 78 | 3 / 22 | 6 / 108 |
| 3 | 7 / 78 | 3 / 23 | 6 / 107 |
| 4 | 8 / 82 | 3 / 23 | 6 / 107 |

Cách dùng khi train: mỗi lần chạy lấy 1 fold làm test; validation lấy từ các nhóm thuộc 4 fold còn lại
(cũng chia theo `group_id`); không dùng fold test để chọn siêu tham số.

**Bài thi phụ — Bangladesh:** từ 1.933 ảnh của Advanced Imaging, bỏ ảnh có trong IQ-OTH, ảnh lặp và 4 ảnh bị gán
2 nhãn → **1.476 ảnh** (benign 456, malignant 1.020). Không có ảnh normal, nên chỉ dùng để đo khả năng phân biệt
benign / malignant trên ảnh của bệnh viện khác.

**Đầu ra** (`data/processed/`, không đưa lên Git — ai cần thì chạy lại script):
- `iq_oth_folds.csv`: `path, label, patient_id, group_id, fold, sha256`
- `external_bangladesh.csv`: `path, label, sha256`

`path` tính từ thư mục `data/raw/`.

**Kiểm tra rò rỉ** (`tests/test_no_patient_leakage.py`, 6/6 đạt): mỗi bệnh nhân và mỗi nhóm chỉ ở 1 fold;
không còn ảnh trùng; fold nào cũng đủ 3 nhãn; bài thi phụ không trùng dữ liệu chính; mọi đường dẫn tồn tại.
Đã thử cố tình chuyển 1 ảnh sang fold khác — test báo lỗi đúng.

## 5. Hạn chế cần ghi trong báo cáo

- **Ít dữ liệu:** dữ liệu chính chỉ 1.073 ảnh / 82 nhóm bệnh nhân, từ 1 bệnh viện.
- **Mất cân bằng:** benign chỉ 15 mã bệnh nhân (≈ 3 nhóm mỗi fold) → kết quả lớp benign dao động mạnh giữa các fold;
  cần class weights / sampler và báo cáo theo từng lớp (Recall, F1), không chỉ Accuracy.
- **Bài thi phụ thiếu lớp normal:** không đánh giá được đủ 3 lớp trên bệnh viện khác.
- **Kích thước ảnh không đồng nhất** ở IQ-OTH (61 ảnh không phải 512×512) → cần resize thống nhất khi tiền xử lý.
- **Không có vị trí tổn thương:** không dataset nào có bounding box / mask → không làm được bài toán Detection,
  Grad-CAM chỉ để giải thích, không phải khoanh vùng.

## 6. Cách chạy lại từ đầu

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
# Tải 3 dataset vào data/raw/ (giữ nguyên cấu trúc thư mục như mục 2)
.venv\Scripts\python.exe -X utf8 scripts\audit_dataset.py
.venv\Scripts\python.exe -X utf8 scripts\build_dataset.py
.venv\Scripts\python.exe -X utf8 tests\test_no_patient_leakage.py
```

Thư mục ảnh cần có trong `data/raw/`:
- `iq_oth_patient_level/The IQ-OTHNCCD Patient-Level Annotated Lung Cancer Dataset/` (giải nén file `.rar` gốc)
- `mendeley_advanced_imaging_full/An Image Dataset of Advanced Imaging Techniques for Lung Cancer Diagnosis/`
  (giải nén file zip "Download All" — tải từng file qua API chỉ được 1.000 file/thư mục, sẽ thiếu ảnh)
- `mendeley_federated/LungData/` (chỉ cần nếu muốn chạy lại phần kiểm tra trùng)
