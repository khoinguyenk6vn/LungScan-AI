"""Tạo bảng dữ liệu sạch và chia 5 fold theo bệnh nhân (không rò rỉ).

Quyết định (xem docs/DATASET_AUDIT.md):
  - Dữ liệu chính: IQ-OTH/NCCD Patient-Level (1 bệnh viện, có mã bệnh nhân).
      * Bỏ ảnh lặp y hệt (giữ 1 bản).
      * Các mã bệnh nhân có chung ảnh y hệt -> gom thành 1 nhóm (group_id),
        cả nhóm luôn nằm chung 1 fold.
      * Chia 5 fold phân tầng theo nhãn, đơn vị chia là nhóm bệnh nhân.
  - Bài thi phụ (external test): ảnh của bộ Advanced Imaging (Bangladesh) KHÔNG có trong
    IQ-OTH — chỉ gồm benign / malignant; bỏ ảnh lặp và ảnh bị gán 2 nhãn.
    Bộ Federated không dùng: toàn bộ ảnh của nó là bản sao của 2 bộ trên.

Đầu ra (data/processed/, không đưa lên Git):
  - iq_oth_folds.csv         : path, label, patient_id, group_id, fold, sha256
  - external_bangladesh.csv  : path, label, sha256

Chạy:  .venv\\Scripts\\python.exe -X utf8 scripts\\build_dataset.py
"""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict

from audit_dataset import DATASETS, RAW, ROOT, AnhInfo, quet

OUT = ROOT / "data" / "processed"
SO_FOLD = 5
SEED = 42
NHAN = ("normal", "benign", "malignant")


def gom_nhom_benh_nhan(anh: list[AnhInfo]) -> dict[str, str]:
    """Union-find: 2 mã bệnh nhân có chung 1 ảnh y hệt -> cùng nhóm. Trả về patient_id -> group_id."""
    cha = {a.patient_id: a.patient_id for a in anh}

    def goc(x: str) -> str:
        while cha[x] != x:
            cha[x] = cha[cha[x]]
            x = cha[x]
        return x

    dau_tien: dict[str, str] = {}
    for a in anh:
        if a.sha256 in dau_tien:
            ra, rb = goc(dau_tien[a.sha256]), goc(a.patient_id)
            if ra != rb:
                cha[max(ra, rb)] = min(ra, rb)
        else:
            dau_tien[a.sha256] = a.patient_id
    return {p: goc(p) for p in cha}


def chia_fold(nhom: dict[str, tuple[str, int]]) -> dict[str, int]:
    """Chia nhóm vào fold, phân tầng theo nhãn: mỗi nhãn chia đều số nhóm, rồi cân số ảnh.

    nhom: group_id -> (nhãn, số ảnh).
    """
    rng = random.Random(SEED)
    fold_cua: dict[str, int] = {}
    for nhan in NHAN:
        ds = [g for g, (l, _) in nhom.items() if l == nhan]
        rng.shuffle(ds)
        ds.sort(key=lambda g: -nhom[g][1])  # nhóm nhiều ảnh xếp trước cho dễ cân
        so_nhom, so_anh = [0] * SO_FOLD, [0] * SO_FOLD
        for g in ds:
            f = min(range(SO_FOLD), key=lambda i: (so_nhom[i], so_anh[i]))
            fold_cua[g] = f
            so_nhom[f] += 1
            so_anh[f] += nhom[g][1]
    return fold_cua


def ghi_csv(path, fields, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Dữ liệu chính: IQ-OTH ----
    iq, loi = quet("iq_oth", DATASETS["iq_oth"])
    if loi:
        raise SystemExit(f"IQ-OTH có {len(loi)} ảnh lỗi, chạy audit_dataset.py trước: {loi[:3]}")
    group_of = gom_nhom_benh_nhan(iq)
    nhan_nhom = {}
    for a in iq:
        g = group_of[a.patient_id]
        if nhan_nhom.setdefault(g, a.label) != a.label:
            raise SystemExit(f"Nhóm {g} lẫn 2 nhãn — cần xem lại dữ liệu")

    da_co, sach = set(), []
    for a in iq:  # bỏ ảnh lặp, giữ bản đầu tiên (thứ tự đường dẫn đã sắp xếp)
        if a.sha256 not in da_co:
            da_co.add(a.sha256)
            sach.append(a)

    nhom = {g: (nhan_nhom[g], sum(1 for a in sach if group_of[a.patient_id] == g)) for g in set(group_of.values())}
    fold_cua = chia_fold(nhom)
    base = DATASETS["iq_oth"].relative_to(RAW).as_posix()
    rows = [{"path": f"{base}/{a.rel_path}", "label": a.label, "patient_id": a.patient_id,
             "group_id": group_of[a.patient_id], "fold": fold_cua[group_of[a.patient_id]], "sha256": a.sha256}
            for a in sach]
    ghi_csv(OUT / "iq_oth_folds.csv", list(rows[0]), rows)

    print(f"IQ-OTH: {len(iq)} ảnh -> bỏ {len(iq) - len(sach)} ảnh lặp -> {len(sach)} ảnh | "
          f"{len(group_of)} mã bệnh nhân -> {len(nhom)} nhóm")
    print("fold | " + " | ".join(f"{n:>9s} (nhóm/ảnh)" for n in NHAN))
    for f in range(SO_FOLD):
        cot = []
        for n in NHAN:
            gs = [g for g, (l, _) in nhom.items() if l == n and fold_cua[g] == f]
            cot.append(f"{len(gs):>9d} / {sum(nhom[g][1] for g in gs):<10d}")
        print(f"  {f}  | " + " | ".join(cot))

    # ---- Bài thi phụ: ảnh Bangladesh không có trong IQ-OTH ----
    adv, loi = quet("advanced", DATASETS["advanced"])
    if loi:
        raise SystemExit(f"Advanced có {len(loi)} ảnh lỗi: {loi[:3]}")
    nhan_theo_sha = defaultdict(set)
    for a in adv:
        nhan_theo_sha[a.sha256].add(a.label)
    sha_iq = {a.sha256 for a in iq}
    da_co, ext = set(), []
    for a in adv:
        if a.sha256 in sha_iq or a.sha256 in da_co or len(nhan_theo_sha[a.sha256]) > 1:
            continue
        da_co.add(a.sha256)
        ext.append(a)
    base = DATASETS["advanced"].relative_to(RAW).as_posix()
    ghi_csv(OUT / "external_bangladesh.csv", ["path", "label", "sha256"],
            [{"path": f"{base}/{a.rel_path}", "label": a.label, "sha256": a.sha256} for a in ext])
    print(f"\nBài thi phụ (Bangladesh): {len(adv)} ảnh -> {len(ext)} ảnh dùng được "
          f"{dict(Counter(a.label for a in ext))}")
    print(f"Đã ghi: {OUT / 'iq_oth_folds.csv'} và {OUT / 'external_bangladesh.csv'}")


if __name__ == "__main__":
    main()
