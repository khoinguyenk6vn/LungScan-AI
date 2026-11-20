"""Kiểm tra bảng chia fold không rò rỉ dữ liệu.

Chạy được bằng pytest hoặc trực tiếp:  .venv\\Scripts\\python.exe -X utf8 tests\\test_no_patient_leakage.py
Cần chạy scripts/build_dataset.py trước để có data/processed/*.csv.
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


def doc(ten: str) -> list[dict]:
    with open(PROCESSED / ten, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mot_fold(rows: list[dict], cot: str) -> None:
    folds = defaultdict(set)
    for r in rows:
        folds[r[cot]].add(r["fold"])
    sai = {k: v for k, v in folds.items() if len(v) > 1}
    assert not sai, f"{cot} nằm ở nhiều fold: {list(sai.items())[:5]}"


def test_benh_nhan_chi_o_1_fold():
    _mot_fold(doc("iq_oth_folds.csv"), "patient_id")


def test_nhom_chi_o_1_fold():
    _mot_fold(doc("iq_oth_folds.csv"), "group_id")


def test_khong_co_anh_trung():
    rows = doc("iq_oth_folds.csv")
    assert len({r["sha256"] for r in rows}) == len(rows), "Còn ảnh trùng y hệt trong dữ liệu chính"


def test_moi_fold_du_3_nhan():
    labels = defaultdict(set)
    for r in doc("iq_oth_folds.csv"):
        labels[r["fold"]].add(r["label"])
    assert len(labels) == 5
    assert all(v == {"normal", "benign", "malignant"} for v in labels.values()), dict(labels)


def test_bai_thi_phu_khong_trung_du_lieu_chinh():
    chinh = {r["sha256"] for r in doc("iq_oth_folds.csv")}
    phu = doc("external_bangladesh.csv")
    assert not chinh & {r["sha256"] for r in phu}, "Bài thi phụ có ảnh trùng dữ liệu chính"
    assert len({r["sha256"] for r in phu}) == len(phu), "Bài thi phụ còn ảnh trùng"


def test_duong_dan_ton_tai():
    for ten in ("iq_oth_folds.csv", "external_bangladesh.csv"):
        thieu = [r["path"] for r in doc(ten) if not (RAW / r["path"]).is_file()]
        assert not thieu, f"{ten}: {len(thieu)} đường dẫn không tồn tại, ví dụ {thieu[:3]}"


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    loi = 0
    for ten, fn in tests:
        try:
            fn()
            print(f"OK    {ten}")
        except AssertionError as e:
            loi += 1
            print(f"LỖI   {ten}: {e}")
    print(f"\n{len(tests) - loi}/{len(tests)} kiểm tra đạt")
    raise SystemExit(1 if loi else 0)
