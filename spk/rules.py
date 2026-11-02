"""
Basis aturan fuzzy lengkap: 3 x 3 x 3 x 3 = 81 aturan.

Konsekuen setiap aturan tidak ditulis acak, melainkan ditentukan oleh pemetaan berbobot:

    1. Setiap kategori input diberi "tingkat manfaat" 0, 1, atau 2:
        - IPK, Prestasi            : Rendah=0, Sedang=1, Tinggi=2        (makin tinggi makin layak)
        - Penghasilan Orang Tua    : Tinggi=0, Sedang=1, Rendah=2        (makin rendah makin diprioritaskan)
        - Jumlah Tanggungan        : Sedikit=0, Sedang=1, Banyak=2       (makin banyak makin diprioritaskan)
    2. Skor bobot = jumlah (bobot variabel x tingkat manfaat), rentang 0..200.
        Bobot (asumsi perancang): IPK 35, Penghasilan 30, Prestasi 20, Tanggungan 15 (total 100).
    3. Skor bobot dipetakan ke konsekuen dengan ambang:
        < 80 Tidak Layak | 80-109 Cukup Layak | 110-149 Layak | >= 150 Sangat Layak
    4. Aturan veto akademik: IPK "Rendah" membatasi konsekuen maksimum "Cukup Layak"
        (syarat akademik minimum tidak dapat sepenuhnya ditutup oleh faktor lain).
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Sequence

import pandas as pd

from .fuzzy_engine import INPUT_VARIABLES, OUTPUT_VARIABLE, Rule

# Urutan variabel pada antecedent (sama dengan contoh format pada tugas)
URUTAN_VARIABEL = ("ipk", "penghasilan", "tanggungan", "prestasi")

BOBOT: Dict[str, int] = {"ipk": 35, "penghasilan": 30, "tanggungan": 15, "prestasi": 20}

TINGKAT_MANFAAT: Dict[str, Dict[str, int]] = {
    "ipk": {"Rendah": 0, "Sedang": 1, "Tinggi": 2},
    "penghasilan": {"Tinggi": 0, "Sedang": 1, "Rendah": 2},
    "tanggungan": {"Sedikit": 0, "Sedang": 1, "Banyak": 2},
    "prestasi": {"Rendah": 0, "Sedang": 1, "Tinggi": 2},
}

# (batas atas eksklusif skor bobot, konsekuen); sisanya -> "Sangat Layak"
AMBANG_KONSEKUEN = ((80, "Tidak Layak"), (110, "Cukup Layak"), (150, "Layak"))
KONSEKUEN_TERTINGGI = "Sangat Layak"
BATAS_KONSEKUEN_IPK_RENDAH = "Cukup Layak"   # aturan veto akademik

URUTAN_KONSEKUEN: List[str] = OUTPUT_VARIABLE.set_names


def skor_bobot(antecedents: Dict[str, str]) -> int:
    """Skor bobot 0..200 dari satu kombinasi kategori input."""
    return sum(BOBOT[k] * TINGKAT_MANFAAT[k][antecedents[k]] for k in URUTAN_VARIABEL)


def tentukan_konsekuen(antecedents: Dict[str, str]) -> str:
    """Pemetaan kombinasi antecedent -> konsekuen (bobot + ambang + veto IPK rendah)."""
    skor = skor_bobot(antecedents)
    konsekuen = KONSEKUEN_TERTINGGI
    for batas, nama in AMBANG_KONSEKUEN:
        if skor < batas:
            konsekuen = nama
            break
    if antecedents["ipk"] == "Rendah":
        maks = URUTAN_KONSEKUEN.index(BATAS_KONSEKUEN_IPK_RENDAH)
        if URUTAN_KONSEKUEN.index(konsekuen) > maks:
            konsekuen = BATAS_KONSEKUEN_IPK_RENDAH
    return konsekuen


def bangun_basis_aturan() -> List[Rule]:
    """Membangun 81 aturan dengan penomoran R1..R81 (IPK paling lambat berubah)."""
    daftar_himpunan = [INPUT_VARIABLES[k].set_names for k in URUTAN_VARIABEL]
    aturan: List[Rule] = []
    for nomor, kombinasi in enumerate(product(*daftar_himpunan), start=1):
        ant = dict(zip(URUTAN_VARIABEL, kombinasi))
        aturan.append(
            Rule(
                id=nomor,
                antecedents=tuple((k, ant[k]) for k in URUTAN_VARIABEL),
                consequent=tentukan_konsekuen(ant),
            )
        )
    return aturan


RULES: List[Rule] = bangun_basis_aturan()


def rules_to_dataframe(rules: Sequence[Rule] = RULES) -> pd.DataFrame:
    """Tabel basis aturan untuk ditampilkan dan difilter."""
    baris = []
    for r in rules:
        ant = r.antecedent_dict()
        baris.append(
            {
                "No": r.id,
                "Aturan": f"R{r.id}",
                "IPK": ant["ipk"],
                "Penghasilan": ant["penghasilan"],
                "Tanggungan": ant["tanggungan"],
                "Prestasi": ant["prestasi"],
                "Skor Bobot (0–200)": skor_bobot(ant),
                "Kesimpulan": r.consequent,
            }
        )
    return pd.DataFrame(baris)


def cek_kelengkapan(rules: Sequence[Rule] = RULES) -> Dict[str, object]:
    """Memeriksa bahwa seluruh 81 kombinasi terwakili tepat satu kali."""
    seharusnya = set(product(*[INPUT_VARIABLES[k].set_names for k in URUTAN_VARIABEL]))
    ada = [tuple(r.antecedent_dict()[k] for k in URUTAN_VARIABEL) for r in rules]
    konsekuen_valid = all(r.consequent in URUTAN_KONSEKUEN for r in rules)
    return {
        "jumlah_aturan": len(rules),
        "jumlah_kombinasi_unik": len(set(ada)),
        "kombinasi_hilang": sorted(seharusnya - set(ada)),
        "duplikat": len(ada) - len(set(ada)),
        "semua_konsekuen_valid": konsekuen_valid,
        "lengkap": set(ada) == seharusnya and len(ada) == len(seharusnya) and konsekuen_valid,
    }


def distribusi_konsekuen(rules: Sequence[Rule] = RULES) -> pd.DataFrame:
    hitung = {nama: 0 for nama in URUTAN_KONSEKUEN}
    for r in rules:
        hitung[r.consequent] += 1
    return pd.DataFrame({"Konsekuen": list(hitung), "Jumlah Aturan": list(hitung.values())})