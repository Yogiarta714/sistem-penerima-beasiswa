from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import fuzzy_engine as fe
from .fuzzy_engine import (
    GRID_STEP,
    INPUT_VARIABLES,
    OUTPUT_VARIABLE,
    FuzzyInferenceError,
    InputValidationError,
    Rule,
)
from .data import SKENARIO_PENGUJIAN

TOL_MU = 1e-9              # toleransi derajat keanggotaan & alpha
TOL_CENTROID_DISKRIT = 1e-6  # program vs manual pada grid yang sama (0,1)
TOL_CENTROID_KONTINU = 0.05  # grid 0,1 vs pembanding grid 0,01 / analitik (galat diskretisasi)
TOL_MU_UJI = 1e-9


# 1. Pengujian skenario
def _ringkas_fuzzifikasi(fuzzified: Mapping[str, Mapping[str, float]]) -> str:
    potongan = []
    for key, var in INPUT_VARIABLES.items():
        isi = ", ".join(f"{nama[0]}={mu:.2f}" for nama, mu in fuzzified[key].items())
        potongan.append(f"{var.label.split()[0]}[{isi}]")
    return " | ".join(potongan)


def jalankan_skenario_pengujian(rules: Sequence[Rule]) -> pd.DataFrame:
    """Menjalankan semua skenario; status LULUS bila semua pemeriksaan terpenuhi."""
    baris = []
    for sk in SKENARIO_PENGUJIAN:
        masukan = sk["inputs"]
        gagal: List[str] = []
        skor, kategori, fuzz_txt, n_aktif = float("nan"), "Error", "-", 0
        try:
            r = fe.infer(masukan, rules)
            skor, kategori, n_aktif = r.centroid, r.category, len(r.active_activations)
            fuzz_txt = _ringkas_fuzzifikasi(r.fuzzified)

            semua_mu = [mu for d in r.fuzzified.values() for mu in d.values()]
            if not all(0.0 <= mu <= 1.0 for mu in semua_mu):
                gagal.append("μ di luar [0,1]")
            if any(max(d.values()) <= 0 for d in r.fuzzified.values()):
                gagal.append("ada variabel tanpa μ > 0")
            if n_aktif < 1:
                gagal.append("tidak ada aturan aktif")
            if not (0.0 <= skor <= 100.0):
                gagal.append("skor di luar 0–100")
            if sk["harapan_kategori"] and kategori not in sk["harapan_kategori"]:
                gagal.append(
                    f"kategori {kategori} di luar harapan {'/'.join(sk['harapan_kategori'])}"
                )
            for (vkey, himp), nilai in sk["harapan_mu"].items():
                mu = r.fuzzified[vkey][himp]
                if abs(mu - nilai) > TOL_MU_UJI:
                    gagal.append(f"μ {vkey}.{himp}={mu:.4f} ≠ {nilai}")
        except (InputValidationError, FuzzyInferenceError) as err:
            gagal.append(f"error: {err}")
        i = sk["inputs"]
        baris.append(
            {
                "No": sk["No"],
                "Skenario": sk["Skenario"],
                "Kelompok": sk["Kelompok"],
                "IPK": i["ipk"],
                "Penghasilan": i["penghasilan"],
                "Tanggungan": i["tanggungan"],
                "Prestasi": i["prestasi"],
                "Hasil Fuzzifikasi": fuzz_txt,
                "Aturan Aktif": n_aktif,
                "Skor": round(skor, 4) if skor == skor else skor,
                "Kategori": kategori,
                "Harapan": " / ".join(sk["harapan_kategori"]),
                "Status": "LULUS" if not gagal else "GAGAL",
                "Keterangan Gagal": "; ".join(gagal),
            }
        )
    return pd.DataFrame(baris)


# 2. Uji integritas model
def cek_integritas(rules: Sequence[Rule]) -> pd.DataFrame:
    """Pemeriksaan struktural model; setiap baris: Pemeriksaan, Status, Detail."""
    hasil: List[Tuple[str, bool, str]] = []

    # (a) nilai pada titik batas dan puncak setiap fungsi keanggotaan
    semua_var = list(INPUT_VARIABLES.values()) + [OUTPUT_VARIABLE]
    salah = []
    for var in semua_var:
        for s in var.sets:
            a, b, c, d = s.trapezoid_params()
            harapan = {a: 0.0 if b > a else 1.0, b: 1.0, c: 1.0, d: 0.0 if d > c else 1.0}
            for x, h in harapan.items():
                if abs(float(s.membership(x)) - h) > 1e-12:
                    salah.append(f"{var.label}.{s.name}({x:g})")
    hasil.append(("Nilai MF pada titik a, b, c, d sesuai definisi", not salah,
                "Semua sesuai" if not salah else "Salah: " + ", ".join(salah)))

    # (b) rentang 0..1 dan tanpa celah pada seluruh semesta
    for var in semua_var:
        lo, hi = var.universe
        x = np.linspace(lo, hi, 4001)
        mat = np.vstack([s.membership(x) for s in var.sets])
        ok_rentang = bool(mat.min() >= 0.0 and mat.max() <= 1.0)
        maks_per_titik = mat.max(axis=0)
        ok_celah = bool(maks_per_titik.min() > 0.0)
        jumlah = mat.sum(axis=0)
        hasil.append((f"{var.label}: μ ∈ [0,1]", ok_rentang,
                    f"min={mat.min():.3f}, maks={mat.max():.3f}"))
        hasil.append((f"{var.label}: tanpa celah (maks μ > 0 di seluruh semesta)", ok_celah,
                    f"μ gabungan terkecil = {maks_per_titik.min():.4f}"))
        hasil.append((f"{var.label}: overlap memadai (Σμ = 1 pada seluruh semesta)",
                    bool(np.allclose(jumlah, 1.0, atol=1e-9)),
                    f"Σμ min={jumlah.min():.6f}, maks={jumlah.max():.6f}"))

    # (c) penanganan input di luar semesta
    ditolak = 0
    uji_luar = [("ipk", -0.01), ("ipk", 4.01), ("penghasilan", -1), ("penghasilan", 15_000_001),
                ("tanggungan", -1), ("tanggungan", 9), ("prestasi", -0.5), ("prestasi", 100.5)]
    for key, nilai in uji_luar:
        masukan = {"ipk": 3.0, "penghasilan": 5e6, "tanggungan": 3, "prestasi": 50}
        masukan[key] = nilai
        try:
            fe.validate_inputs(masukan)
        except InputValidationError:
            ditolak += 1
    hasil.append(("Input di luar semesta ditolak dengan pesan error", ditolak == len(uji_luar),
                f"{ditolak} dari {len(uji_luar)} nilai di luar semesta ditolak"))

    # (d) kelengkapan basis aturan
    from .rules import cek_kelengkapan
    kel = cek_kelengkapan(rules)
    hasil.append(("Basis aturan: 81 kombinasi terwakili tepat satu kali", bool(kel["lengkap"]),
                f"{kel['jumlah_aturan']} aturan, {kel['jumlah_kombinasi_unik']} kombinasi unik, "
                f"{kel['duplikat']} duplikat, {len(kel['kombinasi_hilang'])} hilang"))

    # (e) setiap kombinasi nilai yang valid mengaktifkan minimal satu aturan
    titik = {k: np.linspace(v.universe[0], v.universe[1], 7) for k, v in INPUT_VARIABLES.items()}
    tanpa_aturan, total = 0, 0
    for i in titik["ipk"]:
        for p in titik["penghasilan"]:
            for t in titik["tanggungan"]:
                for pr in titik["prestasi"]:
                    total += 1
                    fz = fe.fuzzify_inputs({"ipk": i, "penghasilan": p, "tanggungan": t, "prestasi": pr})
                    if not any(a.active for a in fe.evaluate_rules(fz, rules)):
                        tanpa_aturan += 1
    hasil.append(("Setiap input valid mengaktifkan ≥ 1 aturan (grid 7⁴ titik)", tanpa_aturan == 0,
                f"{total} kombinasi diuji, {tanpa_aturan} tanpa aturan aktif"))

    # (f) inferensi acak berseed tetap: tidak ada error, skor dalam 0..100
    rng = np.random.default_rng(2024)
    galat, di_luar = 0, 0
    for _ in range(300):
        masukan = {
            "ipk": float(rng.uniform(0, 4)),
            "penghasilan": float(rng.uniform(0, 15_000_000)),
            "tanggungan": float(rng.uniform(0, 8)),
            "prestasi": float(rng.uniform(0, 100)),
        }
        try:
            r = fe.infer(masukan, rules)
            if not (0 <= r.centroid <= 100):
                di_luar += 1
        except (InputValidationError, FuzzyInferenceError):
            galat += 1
    hasil.append(("300 input acak (seed 2024): tanpa error & skor 0–100", galat == 0 and di_luar == 0,
                f"{galat} error, {di_luar} skor di luar rentang"))

    # (g) operator: AND=MIN, implikasi=MIN, agregasi=MAX, grid sama
    contoh = {"ipk": 3.2, "penghasilan": 4_000_000, "tanggungan": 2, "prestasi": 70}
    r = fe.infer(contoh, rules)
    ok_and = all(abs(a.alpha - min(a.memberships.values())) < 1e-15 for a in r.activations)
    ok_imp = all(
        np.allclose(r.clipped[a.rule.id], np.minimum(a.alpha, fe.OUTPUT_MU[a.rule.consequent]))
        for a in r.active_activations
    )
    ok_agr = np.allclose(r.aggregated, np.max(np.vstack(list(r.clipped.values())), axis=0))
    ok_grid = r.aggregated.shape == fe.Z_GRID.shape and fe.Z_GRID.size == 1001 \
        and abs(fe.Z_GRID[1] - fe.Z_GRID[0] - 0.1) < 1e-12
    ok_cent = abs(r.centroid - float(np.sum(fe.Z_GRID * r.aggregated) / np.sum(r.aggregated))) < 1e-12
    hasil.append(("Operator AND = MIN", ok_and, "α = min(μ1..μ4) untuk seluruh 81 aturan"))
    hasil.append(("Implikasi = MIN (pemotongan setinggi α)", bool(ok_imp), "kurva = min(α, μ_output)"))
    hasil.append(("Agregasi = MAX atas seluruh aturan aktif", bool(ok_agr),
                f"{len(r.active_activations)} aturan aktif digabung"))
    hasil.append(("Grid output 0–100, interval 0,1, sama untuk agregasi & defuzzifikasi", bool(ok_grid),
                f"{fe.Z_GRID.size} titik"))
    hasil.append(("Centroid = Σ(z·μ)/Σμ", bool(ok_cent), f"Z* = {r.centroid:.6f}"))

    return pd.DataFrame(
        [{"Pemeriksaan": n, "Status": "LULUS" if ok else "GAGAL", "Detail": d} for n, ok, d in hasil]
    )


# 3. Validasi perhitungan manual (Python murni, tanpa NumPy)
KASUS_VALIDASI: List[dict] = [
    {
        "nama": "Kasus 1 – Semua derajat keanggotaan 0,5",
        "inputs": {"ipk": 3.0, "penghasilan": 3_500_000, "tanggungan": 4, "prestasi": 65},
        "deskripsi": "Seluruh variabel tepat di tengah overlap; 2⁴ = 16 aturan aktif dengan α = 0,5.",
    },
    {
        "nama": "Kasus 2 – Derajat campuran",
        "inputs": {"ipk": 3.2, "penghasilan": 4_000_000, "tanggungan": 2, "prestasi": 70},
        "deskripsi": "Derajat keanggotaan berbeda-beda sehingga α setiap aturan berbeda.",
    },
    {
        "nama": "Kasus 3 – Satu aturan aktif (batas atas semesta)",
        "inputs": {"ipk": 4.0, "penghasilan": 0, "tanggungan": 8, "prestasi": 100},
        "deskripsi": "Hanya satu aturan aktif dengan α = 1; centroid dapat dihitung analitik (bentuk tertutup).",
    },
    {
        "nama": "Kasus 4 – Konflik IPK rendah dan prestasi tinggi",
        "inputs": {"ipk": 1.8, "penghasilan": 1_000_000, "tanggungan": 6, "prestasi": 90},
        "deskripsi": "IPK di overlap Rendah–Sedang, ekonomi sulit, prestasi tinggi.",
    },
]


def _mu_manual(x: float, params: Tuple[float, ...]) -> float:
    """Derajat keanggotaan dengan percabangan if/else biasa (independen dari NumPy)."""
    if len(params) == 3:
        a, b, c = params
        b, c, d = b, b, c
    else:
        a, b, c, d = params
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def fuzzifikasi_manual(inputs: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
    return {
        key: {s.name: _mu_manual(float(inputs[key]), s.params) for s in var.sets}
        for key, var in INPUT_VARIABLES.items()
    }


def inferensi_manual(
    inputs: Mapping[str, float], rules: Sequence[Rule], step: float = GRID_STEP
) -> Dict[str, object]:
    """Mamdani manual: loop biasa. `step` menentukan grid (0,1 = sama dengan program)."""
    fz = fuzzifikasi_manual(inputs)

    aktif = []
    for rule in rules:
        mus = [fz[key][himp] for key, himp in rule.antecedents]
        alpha = min(mus)
        if alpha > 0:
            aktif.append({"id": rule.id, "mus": mus, "alpha": alpha, "konsekuen": rule.consequent})

    n = int(round(100 / step))
    zs = [i * step for i in range(n + 1)]
    mu_out = {s.name: [_mu_manual(z, s.params) for z in zs] for s in OUTPUT_VARIABLE.sets}

    pembilang, penyebut = 0.0, 0.0
    agregasi = []
    for i, z in enumerate(zs):
        tertinggi = 0.0
        for a in aktif:
            potong = min(a["alpha"], mu_out[a["konsekuen"]][i])  # implikasi MIN
            if potong > tertinggi:                               # agregasi MAX
                tertinggi = potong
        agregasi.append(tertinggi)
        pembilang += z * tertinggi
        penyebut += tertinggi

    centroid = pembilang / penyebut if penyebut > 0 else float("nan")
    return {
        "fuzzifikasi": fz,
        "aktif": aktif,
        "agregasi": agregasi,
        "pembilang": pembilang,
        "penyebut": penyebut,
        "centroid": centroid,
        "step": step,
    }


def centroid_analitik_satu_himpunan(himpunan, alpha: float) -> float:
    """
    Centroid kontinu bentuk tertutup untuk satu himpunan output yang dipotong setinggi alpha.
    Daerah dipecah menjadi segitiga kiri, persegi panjang, dan segitiga kanan.
    """
    a, b, c, d = himpunan.trapezoid_params()
    x1 = a + alpha * (b - a)       # titik potong sisi naik dengan garis y = alpha
    x2 = d - alpha * (d - c)       # titik potong sisi turun dengan garis y = alpha
    bagian = []                    # (luas, titik berat)
    if x1 > a:
        bagian.append((0.5 * (x1 - a) * alpha, a + 2.0 / 3.0 * (x1 - a)))
    if x2 > x1:
        bagian.append((alpha * (x2 - x1), (x1 + x2) / 2.0))
    if d > x2:
        bagian.append((0.5 * (d - x2) * alpha, d - 2.0 / 3.0 * (d - x2)))
    luas = sum(l for l, _ in bagian)
    return sum(l * t for l, t in bagian) / luas


def jalankan_validasi_manual(kasus: Mapping[str, object], rules: Sequence[Rule]) -> Dict[str, object]:
    """Membandingkan hasil manual dengan hasil program untuk satu kasus."""
    masukan = kasus["inputs"]
    program = fe.infer(masukan, rules)
    manual = inferensi_manual(masukan, rules, step=GRID_STEP)
    halus = inferensi_manual(masukan, rules, step=0.01)

    baris = []

    def tambah(komponen, m, p, tol):
        selisih = abs(m - p)
        baris.append(
            {
                "Komponen": komponen,
                "Hasil Manual": m,
                "Hasil Program": p,
                "Selisih Absolut": selisih,
                "Toleransi": tol,
                "Status": "VALID" if selisih <= tol else "TIDAK VALID",
            }
        )

    for key, var in INPUT_VARIABLES.items():
        for nama in var.set_names:
            tambah(f"μ {var.label} – {nama}", manual["fuzzifikasi"][key][nama],
                program.fuzzified[key][nama], TOL_MU)

    alpha_program = {a.rule.id: a.alpha for a in program.active_activations}
    id_manual = {a["id"] for a in manual["aktif"]}
    for a in manual["aktif"]:
        tambah(f"α-predikat R{a['id']} ({a['konsekuen']})", a["alpha"],
            alpha_program.get(a["id"], float("nan")), TOL_MU)
    tambah("Jumlah aturan aktif", float(len(manual["aktif"])), float(len(alpha_program)), 0.0)
    if id_manual != set(alpha_program):
        baris.append({"Komponen": "Himpunan aturan aktif", "Hasil Manual": float("nan"),
                    "Hasil Program": float("nan"), "Selisih Absolut": float("nan"),
                    "Toleransi": 0.0, "Status": "TIDAK VALID"})

    tambah("Σ μ agregasi (penyebut)", manual["penyebut"], program.denominator, TOL_CENTROID_DISKRIT)
    tambah("Σ z·μ agregasi (pembilang)", manual["pembilang"], program.numerator,
           TOL_CENTROID_DISKRIT * max(1.0, manual["penyebut"]))
    tambah("Centroid Z* (grid 0,1)", manual["centroid"], program.centroid, TOL_CENTROID_DISKRIT)
    tambah("Centroid Z* vs pembanding grid 0,01", halus["centroid"], program.centroid, TOL_CENTROID_KONTINU)

    analitik = None
    if len(manual["aktif"]) == 1:
        a = manual["aktif"][0]
        analitik = centroid_analitik_satu_himpunan(OUTPUT_VARIABLE.get_set(a["konsekuen"]), a["alpha"])
        tambah("Centroid Z* vs rumus analitik (kontinu)", analitik, program.centroid, TOL_CENTROID_KONTINU)

    tabel = pd.DataFrame(baris)
    return {
        "program": program,
        "manual": manual,
        "halus": halus,
        "analitik": analitik,
        "tabel": tabel,
        "valid": bool((tabel["Status"] == "VALID").all()),
    }