"""Kiểm tra (audit) 3 dataset CT phổi trước khi gộp.

Việc làm:
  1. Đếm ảnh theo dataset / nhãn, đọc thử từng ảnh (phát hiện ảnh lỗi).
  2. Thống kê định dạng, kích thước, kênh màu.
  3. Tìm ảnh trùng hệt từng byte (SHA-256) trong cùng dataset và giữa các dataset,
     ảnh trùng mà khác nhãn, ảnh trùng giữa 2 bệnh nhân khác nhau (IQ-OTH).
  4. Đối chiếu bộ IQ-OTH/NCCD với 2 file metadata CSV đi kèm.

Giới hạn: chỉ bắt được bản sao y hệt. Ảnh bị nén / resize lại sẽ không bị phát hiện
(đã thử dHash 64 bit nhưng với ảnh CT báo nhầm quá nhiều nên không dùng).

Kết quả ghi vào reports/metrics/audit_images.csv (mỗi ảnh 1 dòng).

Chạy:  .venv\\Scripts\\python.exe -X utf8 scripts\\audit_dataset.py
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "metrics"
DUOI_ANH = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Tên thư mục nhãn trong từng dataset -> nhãn chuẩn (giữ nguyên lỗi chính tả gốc của tác giả)
NHAN = {
    "NORMAL": "normal", "BENGIN": "benign", "MALIGMNET": "malignant",
    "Normal cases": "normal", "Bengin cases": "benign", "Malignant cases": "malignant",
    "Normal": "normal", "Benign": "benign", "Malignant": "malignant",
}

IQ_DIR = RAW / "iq_oth_patient_level" / "The IQ-OTHNCCD Patient-Level Annotated Lung Cancer Dataset"
DATASETS = {
    "iq_oth": IQ_DIR / "Dataset",  # <Nhãn>/<Mã bệnh nhân>/<ảnh>
    "advanced": RAW / "mendeley_advanced_imaging_full"
    / "An Image Dataset of Advanced Imaging Techniques for Lung Cancer Diagnosis"
    / "Lung Cancer Diagnosis Dataset" / "Dataset",  # <Nhãn>/<ảnh>
    "federated": RAW / "mendeley_federated" / "LungData",  # Client1..4/<Nhãn>/<ảnh>
}


@dataclass
class AnhInfo:
    dataset: str
    rel_path: str
    label: str
    patient_id: str  # chỉ IQ-OTH có; dataset khác để trống
    sub_source: str  # Client1..4 của bộ federated
    fmt: str
    mode: str
    width: int
    height: int
    sha256: str


def quet(name: str, base: Path) -> tuple[list[AnhInfo], list[str]]:
    anh, loi = [], []
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in DUOI_ANH:
            continue
        rel = p.relative_to(base)
        label_dir = next((x for x in rel.parts if x in NHAN), None)
        if label_dir is None:
            loi.append(f"{name}: không xác định nhãn {rel}")
            continue
        try:
            with Image.open(p) as im:
                im.load()
                anh.append(AnhInfo(
                    name, rel.as_posix(), NHAN[label_dir],
                    rel.parts[1] if name == "iq_oth" else "",
                    rel.parts[0] if name == "federated" else "",
                    im.format or "", im.mode, im.width, im.height,
                    hashlib.sha256(p.read_bytes()).hexdigest()))
        except Exception as e:  # ảnh hỏng / không đọc được
            loi.append(f"{name}: không đọc được {rel} ({e})")
    return anh, loi


def bao_cao_trung(anh: list[AnhInfo]) -> None:
    theo_sha: dict[str, list[AnhInfo]] = defaultdict(list)
    for a in anh:
        theo_sha[a.sha256].append(a)
    print(f"Tổng {len(anh)} file ảnh -> chỉ {len(theo_sha)} ảnh KHÁC NHAU")

    sha_cua = {ds: {a.sha256 for a in anh if a.dataset == ds} for ds in DATASETS}
    for ds in DATASETS:
        so_file = sum(1 for a in anh if a.dataset == ds)
        chung = {o: len(sha_cua[ds] & sha_cua[o]) for o in DATASETS if o != ds}
        print(f"{ds:10s} {so_file:5d} file, {len(sha_cua[ds]):5d} ảnh khác nhau | trùng với bộ khác: {chung}")

    for ds in DATASETS:
        if ds == "iq_oth":
            continue
        c = Counter(a.label for a in anh if a.dataset == ds and a.sha256 in sha_cua["iq_oth"])
        print(f"{ds}: ảnh lấy lại từ IQ-OTH theo nhãn {dict(c)}")

    khac_nhan = Counter(tuple(sorted({a.label for a in g})) for g in theo_sha.values()
                        if len({a.label for a in g}) > 1)
    print(f"Ảnh trùng hệt nhưng bị gán KHÁC NHÃN: {sum(khac_nhan.values())} {dict(khac_nhan)}")

    nhieu_bn = sum(1 for g in theo_sha.values()
                   if len({a.patient_id for a in g if a.dataset == "iq_oth"}) > 1)
    print(f"IQ-OTH: ảnh trùng hệt xuất hiện ở 2+ bệnh nhân khác nhau: {nhieu_bn}")

    moi = Counter(Counter(a.label for a in g).most_common(1)[0][0]
                  for h, g in theo_sha.items() if h not in sha_cua["iq_oth"])
    print(f"Ảnh mới thật sự (không có trong IQ-OTH): {sum(moi.values())} {dict(moi)}")


def doi_chieu_iq(anh: list[AnhInfo]) -> None:
    """So ảnh thực tế của IQ-OTH với Dataset_Metadata.csv và Patient_Summary.csv (phân cách bằng tab)."""
    meta = IQ_DIR / "Metadata"
    with open(meta / "Dataset_Metadata.csv", encoding="utf-8-sig") as f:
        rows = {r["Image_ID"]: r for r in csv.DictReader(f, delimiter="\t")}
    tren_dia = {Path(a.rel_path).name: a for a in anh if a.dataset == "iq_oth"}
    sai = [k for k in rows.keys() & tren_dia.keys()
           if rows[k]["Patient_ID"] != tren_dia[k].patient_id or NHAN[rows[k]["Category"]] != tren_dia[k].label]
    print(f"Dataset_Metadata.csv: {len(rows)} dòng | thiếu trên đĩa: {len(rows.keys() - tren_dia.keys())} "
          f"| không có trong CSV: {len(tren_dia.keys() - rows.keys())} | sai mã bệnh nhân/nhãn: {len(sai)}")
    with open(meta / "Patient_Summary.csv", encoding="utf-8-sig") as f:
        bn = list(csv.DictReader(f, delimiter="\t"))
    dem = Counter(a.patient_id for a in anh if a.dataset == "iq_oth")
    lech = [r["Patient_ID"] for r in bn if int(r["Number_of_Images"]) != dem.get(r["Patient_ID"], 0)]
    print(f"Patient_Summary.csv: {len(bn)} bệnh nhân | số ảnh lệch so với thực tế: {len(lech)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    anh, loi = [], []
    for name, base in DATASETS.items():
        a, l = quet(name, base)
        anh += a
        loi += l

    print("=== 1. SỐ ẢNH THEO DATASET / NHÃN ===")
    for ds in DATASETS:
        c = Counter(a.label for a in anh if a.dataset == ds)
        bn = {a.patient_id for a in anh if a.dataset == ds and a.patient_id}
        print(f"{ds:10s} tổng {sum(c.values()):5d} | normal {c['normal']:4d} | benign {c['benign']:4d} "
              f"| malignant {c['malignant']:4d}" + (f" | {len(bn)} bệnh nhân" if bn else ""))
    print(f"Ảnh lỗi / không xác định nhãn: {len(loi)}")
    for x in loi[:20]:
        print("  -", x)

    print("\n=== 2. ĐỊNH DẠNG / KÍCH THƯỚC / KÊNH MÀU ===")
    for ds in DATASETS:
        sub = [a for a in anh if a.dataset == ds]
        print(f"{ds}: định dạng {dict(Counter(a.fmt for a in sub))} | mode {dict(Counter(a.mode for a in sub))}")
        print(f"   kích thước phổ biến: {Counter((a.width, a.height) for a in sub).most_common(5)}")

    print("\n=== 3. ẢNH TRÙNG HỆT (SHA-256) ===")
    bao_cao_trung(anh)

    print("\n=== 4. ĐỐI CHIẾU IQ-OTH VỚI METADATA ===")
    doi_chieu_iq(anh)

    with open(OUT / "audit_images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(anh[0])))
        w.writeheader()
        w.writerows(asdict(a) for a in anh)
    print(f"\nĐã ghi: {OUT / 'audit_images.csv'}")


if __name__ == "__main__":
    main()
