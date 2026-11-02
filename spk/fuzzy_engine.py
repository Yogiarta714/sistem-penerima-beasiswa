"""
Implementasi manual Fuzzy Mamdani menggunakan NumPy.

Tahapan yang diimplementasikan (urut):
    1. Fuzzifikasi           -> derajat keanggotaan setiap himpunan input
    2. Operasi AND           -> alpha-predikat = MIN(mu1, mu2, mu3, mu4)
    3. Implikasi             -> MIN (memotong himpunan output setinggi alpha)
    4. Agregasi              -> MAX atas seluruh aturan aktif
    5. Defuzzifikasi         -> centroid diskrit  Z* = sum(z * mu) / sum(mu)

Grid output dinyatakan eksplisit: 0 sampai 100 dengan interval 0,1 (1001 titik).
Agregasi dan defuzzifikasi memakai grid yang sama (Z_GRID).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

# 1. Grid diskrit output
GRID_MIN = 0.0
GRID_MAX = 100.0
GRID_STEP = 0.1
GRID_POINTS = int(round((GRID_MAX - GRID_MIN) / GRID_STEP)) + 1  # 1001 titik
Z_GRID = np.linspace(GRID_MIN, GRID_MAX, GRID_POINTS)
EPS = 1e-12  # batas penyebut centroid dianggap nol


class InputValidationError(ValueError):
    """Input crisp tidak valid (kosong, bukan angka, atau di luar semesta)."""


class FuzzyInferenceError(RuntimeError):
    """Proses inferensi gagal (mis. tidak ada aturan aktif / agregasi kosong)."""


# 2. Fungsi keanggotaan dasar
def trapezoidal_mf(x, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Fungsi keanggotaan trapesium dengan titik a <= b <= c <= d.

        mu(x) = 0                    , x <= a  atau  x >= d
        mu(x) = (x - a) / (b - a)    , a < x < b
        mu(x) = 1                    , b <= x <= c
        mu(x) = (d - x) / (d - c)    , c < x < d

    Bahu kiri (a == b) bernilai 1 pada x >= b; bahu kanan (c == d) bernilai 1
    pada x <= c. Nilai di luar semesta (di luar [a, d]) bernilai 0.
    Menerima skalar atau array; selalu mengembalikan ndarray.
    """
    if not (a <= b <= c <= d) or a == d:
        raise ValueError(f"Parameter tidak valid: a={a}, b={b}, c={c}, d={d}")
    x = np.asarray(x, dtype=float)
    den_kiri = (b - a) if b > a else 1.0  # penyebut aman; cabang ini tidak dipakai bila b == a
    den_kanan = (d - c) if d > c else 1.0
    kiri = np.where(x >= b, 1.0, np.where(x <= a, 0.0, (x - a) / den_kiri))
    kanan = np.where(x <= c, 1.0, np.where(x >= d, 0.0, (d - x) / den_kanan))
    return np.clip(np.minimum(kiri, kanan), 0.0, 1.0)


def triangular_mf(x, a: float, b: float, c: float) -> np.ndarray:
    """
    Fungsi keanggotaan segitiga dengan titik a <= b <= c (puncak di b).
    Dihitung sebagai trapesium khusus (a, b, b, c).
    """
    return trapezoidal_mf(x, a, b, b, c)


def format_angka(v: float) -> str:
    """Format angka untuk teks: ribuan dengan titik, selain itu notasi ringkas."""
    if abs(v) >= 1000 and float(v).is_integer():
        return f"{int(v):,}".replace(",", ".")
    return f"{v:g}"


# 3. Struktur data variabel fuzzy
@dataclass(frozen=True)
class FuzzySet:
    """Satu himpunan fuzzy (kategori linguistik) beserta parameternya."""

    name: str
    kind: str                    # "triangular" atau "trapezoidal"
    params: Tuple[float, ...]    # (a, b, c) atau (a, b, c, d)

    def membership(self, x) -> np.ndarray:
        if self.kind == "triangular":
            return triangular_mf(x, *self.params)
        if self.kind == "trapezoidal":
            return trapezoidal_mf(x, *self.params)
        raise ValueError(f"Jenis fungsi keanggotaan tidak dikenal: {self.kind}")

    def trapezoid_params(self) -> Tuple[float, float, float, float]:
        """Parameter dalam bentuk (a, b, c, d); segitiga menjadi (a, b, b, c)."""
        if self.kind == "triangular":
            a, b, c = self.params
            return a, b, b, c
        return self.params  # type: ignore[return-value]

    def equation_lines(self) -> List[str]:
        """Persamaan matematis fungsi keanggotaan dalam bentuk teks."""
        a, b, c, d = self.trapezoid_params()
        f = format_angka
        baris: List[str] = []
        if b > a:
            baris.append(f"μ(x) = 0 ; x ≤ {f(a)}")
            baris.append(f"μ(x) = (x − {f(a)}) / ({f(b)} − {f(a)}) ; {f(a)} < x < {f(b)}")
        if b == c:
            baris.append(f"μ(x) = 1 ; x = {f(b)}")
        else:
            baris.append(f"μ(x) = 1 ; {f(b)} ≤ x ≤ {f(c)}")
        if d > c:
            baris.append(f"μ(x) = ({f(d)} − x) / ({f(d)} − {f(c)}) ; {f(c)} < x < {f(d)}")
            baris.append(f"μ(x) = 0 ; x ≥ {f(d)}")
        return baris


@dataclass(frozen=True)
class FuzzyVariable:
    """Variabel fuzzy: semesta pembicaraan + daftar himpunan linguistik."""

    key: str
    label: str
    unit: str
    universe: Tuple[float, float]
    sets: Tuple[FuzzySet, ...]
    rationale: str = ""
    interpretation: str = ""

    @property
    def set_names(self) -> List[str]:
        return [s.name for s in self.sets]

    def get_set(self, name: str) -> FuzzySet:
        for s in self.sets:
            if s.name == name:
                return s
        raise KeyError(f"Himpunan '{name}' tidak ada pada variabel '{self.label}'")

    def fuzzify(self, x: float) -> Dict[str, float]:
        """Derajat keanggotaan nilai crisp x pada setiap himpunan."""
        return {s.name: float(s.membership(x)) for s in self.sets}


# 4. Definisi variabel
INPUT_VARIABLES: Dict[str, FuzzyVariable] = {
    "ipk": FuzzyVariable(
        key="ipk",
        label="IPK",
        unit="skala 0–4",
        universe=(0.0, 4.0),
        sets=(
            FuzzySet("Rendah", "trapezoidal", (0.0, 0.0, 1.5, 2.5)),
            FuzzySet("Sedang", "triangular", (1.5, 2.5, 3.5)),
            FuzzySet("Tinggi", "trapezoidal", (2.5, 3.5, 4.0, 4.0)),
        ),
        rationale=(
            "Titik 1,5 / 2,5 / 3,5 dipilih sebagai asumsi perancang: IPK ≤ 1,5 dianggap pasti "
            "rendah, 2,5 sebagai pusat kategori sedang, dan IPK ≥ 3,5 dianggap pasti tinggi "
            "(mendekati batas predikat pujian yang umum di perguruan tinggi). Setiap kategori "
            "bertumpang tindih selebar 1,0 sehingga perubahan kecil IPK tidak menyebabkan "
            "lompatan kategori."
        ),
        interpretation=(
            "IPK 2,0 berada di zona peralihan Rendah–Sedang (masing-masing 0,5); IPK 3,0 berada "
            "di zona peralihan Sedang–Tinggi (masing-masing 0,5)."
        ),
    ),
    "penghasilan": FuzzyVariable(
        key="penghasilan",
        label="Penghasilan Orang Tua",
        unit="Rp per bulan",
        universe=(0.0, 15_000_000.0),
        sets=(
            FuzzySet("Rendah", "trapezoidal", (0.0, 0.0, 2_000_000.0, 5_000_000.0)),
            FuzzySet("Sedang", "triangular", (2_000_000.0, 5_000_000.0, 9_000_000.0)),
            FuzzySet("Tinggi", "trapezoidal", (5_000_000.0, 9_000_000.0, 15_000_000.0, 15_000_000.0)),
        ),
        rationale=(
            "Batas Rp2 juta / Rp5 juta / Rp9 juta adalah asumsi perancang: ≤ Rp2 juta dianggap "
            "pasti rendah (di bawah kisaran upah minimum daerah), Rp5 juta sebagai pusat "
            "kategori sedang, dan ≥ Rp9 juta dianggap pasti tinggi. Mahasiswa dengan penghasilan "
            "rendah mendapat prioritas bantuan lebih besar (diatur pada basis aturan)."
        ),
        interpretation=(
            "Rp3,5 juta berada di zona peralihan Rendah–Sedang (0,5 : 0,5); Rp7 juta berada di "
            "zona peralihan Sedang–Tinggi (0,5 : 0,5)."
        ),
    ),
    "tanggungan": FuzzyVariable(
        key="tanggungan",
        label="Jumlah Tanggungan Keluarga",
        unit="orang",
        universe=(0.0, 8.0),
        sets=(
            FuzzySet("Sedikit", "trapezoidal", (0.0, 0.0, 1.0, 3.0)),
            FuzzySet("Sedang", "triangular", (1.0, 3.0, 5.0)),
            FuzzySet("Banyak", "trapezoidal", (3.0, 5.0, 8.0, 8.0)),
        ),
        rationale=(
            "0–1 orang dianggap pasti sedikit, 3 orang adalah pusat kategori sedang, dan ≥ 5 "
            "orang dianggap pasti banyak. Nilai tanggungan bersifat bilangan bulat pada aplikasi "
            "tetapi mesin tetap menerima nilai desimal."
        ),
        interpretation=(
            "2 tanggungan = 0,5 Sedikit + 0,5 Sedang; 4 tanggungan = 0,5 Sedang + 0,5 Banyak."
        ),
    ),
    "prestasi": FuzzyVariable(
        key="prestasi",
        label="Prestasi Mahasiswa",
        unit="skor 0–100",
        universe=(0.0, 100.0),
        sets=(
            FuzzySet("Rendah", "trapezoidal", (0.0, 0.0, 20.0, 50.0)),
            FuzzySet("Sedang", "triangular", (20.0, 50.0, 80.0)),
            FuzzySet("Tinggi", "trapezoidal", (50.0, 80.0, 100.0, 100.0)),
        ),
        rationale=(
            "Skor prestasi ≤ 20 dianggap pasti rendah, 50 sebagai pusat kategori sedang, dan "
            "≥ 80 dianggap pasti tinggi. Rentang peralihan yang lebar (30 poin) dipilih karena "
            "skor prestasi bersifat subjektif dan tidak memiliki batas tegas."
        ),
        interpretation=(
            "Skor 35 = 0,5 Rendah + 0,5 Sedang; skor 65 = 0,5 Sedang + 0,5 Tinggi."
        ),
    ),
}

OUTPUT_VARIABLE = FuzzyVariable(
    key="kelayakan",
    label="Kelayakan Beasiswa",
    unit="skor 0–100",
    universe=(0.0, 100.0),
    sets=(
        FuzzySet("Tidak Layak", "trapezoidal", (0.0, 0.0, 15.0, 35.0)),
        FuzzySet("Cukup Layak", "triangular", (15.0, 35.0, 55.0)),
        FuzzySet("Layak", "triangular", (35.0, 55.0, 75.0)),
        FuzzySet("Sangat Layak", "trapezoidal", (55.0, 75.0, 100.0, 100.0)),
    ),
    rationale=(
        "Empat himpunan output disusun berurutan dengan puncak di 35, 55, dan 75 (plateau "
        "Tidak Layak di 0–15 dan Sangat Layak di 75–100). Tumpang tindih selebar 20 poin "
        "menjamin transisi halus antar kategori. Karena centroid merupakan rata-rata terbobot, "
        "skor akhir hampir tidak pernah tepat 0 atau 100."
    ),
    interpretation=(
        "Kategori akhir ditentukan dari himpunan output dengan derajat keanggotaan terbesar "
        "pada skor Z*. Batas kategori terjadi di titik potong himpunan yang bersebelahan "
        "(Z* = 25, 45, dan 65)."
    ),
)

INPUT_COLUMNS: Dict[str, str] = {
    "ipk": "IPK",
    "penghasilan": "Penghasilan",
    "tanggungan": "Tanggungan",
    "prestasi": "Prestasi",
}

# Himpunan output dihitung sekali pada grid yang sama untuk implikasi & agregasi
OUTPUT_MU: Dict[str, np.ndarray] = {
    s.name: s.membership(Z_GRID) for s in OUTPUT_VARIABLE.sets
}


# 5. Aturan
@dataclass(frozen=True)
class Rule:
    """Satu aturan IF-THEN: antecedents = ((kunci_variabel, himpunan), ...)."""

    id: int
    antecedents: Tuple[Tuple[str, str], ...]
    consequent: str

    def antecedent_dict(self) -> Dict[str, str]:
        return dict(self.antecedents)

    def text(self) -> str:
        bagian = [
            f"{INPUT_VARIABLES[k].label} {v}" for k, v in self.antecedents
        ]
        return f"R{self.id}: IF " + " AND ".join(bagian) + f" THEN Kelayakan {self.consequent}"


# 6. Hasil inferensi
@dataclass
class RuleActivation:
    rule: Rule
    memberships: Dict[str, float]   # derajat keanggotaan tiap antecedent
    alpha: float                    # alpha-predikat = MIN
    limiting_variable: str          # variabel dengan mu terkecil (pembatas alpha)

    @property
    def active(self) -> bool:
        return self.alpha > 0.0


@dataclass
class InferenceResult:
    inputs: Dict[str, float]
    fuzzified: Dict[str, Dict[str, float]]
    activations: List[RuleActivation]          # seluruh aturan (aktif & tidak)
    clipped: Dict[int, np.ndarray]             # id aturan aktif -> kurva implikasi
    aggregated: np.ndarray                     # hasil agregasi MAX pada Z_GRID
    numerator: float                           # sum(z * mu)
    denominator: float                         # sum(mu)
    centroid: float                            # Z*
    category: str
    category_memberships: Dict[str, float]

    @property
    def active_activations(self) -> List[RuleActivation]:
        return [a for a in self.activations if a.active]

    # ----- tabel bantu untuk tampilan -----
    def fuzzification_table(self) -> pd.DataFrame:
        baris = []
        for key, var in INPUT_VARIABLES.items():
            for nama, mu in self.fuzzified[key].items():
                baris.append(
                    {
                        "Variabel": var.label,
                        "Nilai Crisp": self.inputs[key],
                        "Himpunan": nama,
                        "Derajat Keanggotaan (μ)": mu,
                    }
                )
        return pd.DataFrame(baris)

    def activation_table(self, only_active: bool = True) -> pd.DataFrame:
        baris = []
        for act in self.activations:
            if only_active and not act.active:
                continue
            ant = act.rule.antecedent_dict()
            baris.append(
                {
                    "Aturan": f"R{act.rule.id}",
                    "IPK": f"{ant['ipk']} ({act.memberships['ipk']:.4f})",
                    "Penghasilan": f"{ant['penghasilan']} ({act.memberships['penghasilan']:.4f})",
                    "Tanggungan": f"{ant['tanggungan']} ({act.memberships['tanggungan']:.4f})",
                    "Prestasi": f"{ant['prestasi']} ({act.memberships['prestasi']:.4f})",
                    "α-predikat (MIN)": act.alpha,
                    "Pembatas α": INPUT_VARIABLES[act.limiting_variable].label if act.alpha > 0 else "-",
                    "Konsekuen": act.rule.consequent,
                    "Status": "Aktif" if act.active else "Tidak aktif",
                }
            )
        return pd.DataFrame(baris)

    def implication_table(self) -> pd.DataFrame:
        baris = []
        for act in self.active_activations:
            kurva = self.clipped[act.rule.id]
            baris.append(
                {
                    "Aturan": f"R{act.rule.id}",
                    "Konsekuen": act.rule.consequent,
                    "α-predikat": act.alpha,
                    "Tinggi Potongan (maks μ)": float(kurva.max()),
                    "Luas Daerah Terpotong": float(kurva.sum() * GRID_STEP),
                }
            )
        return pd.DataFrame(baris)

    def aggregation_by_output(self) -> pd.DataFrame:
        baris = []
        for nama in OUTPUT_VARIABLE.set_names:
            alphas = [a.alpha for a in self.active_activations if a.rule.consequent == nama]
            baris.append(
                {
                    "Himpunan Output": nama,
                    "Jumlah Aturan Aktif": len(alphas),
                    "α Maksimum (MAX)": max(alphas) if alphas else 0.0,
                }
            )
        return pd.DataFrame(baris)

    def grid_table(self, step_tampil: float = 5.0) -> pd.DataFrame:
        """Nilai numerik tiap tahap pada grid (setiap `step_tampil` satuan)."""
        lompat = max(1, int(round(step_tampil / GRID_STEP)))
        idx = np.arange(0, GRID_POINTS, lompat)
        data = {"z": Z_GRID[idx]}
        for aid, kurva in self.clipped.items():
            data[f"R{aid}"] = kurva[idx]
        data["Agregasi (MAX)"] = self.aggregated[idx]
        data["z × μ(z)"] = Z_GRID[idx] * self.aggregated[idx]
        return pd.DataFrame(data)


# 7. Tahapan Mamdani
def validate_inputs(inputs: Mapping[str, float]) -> Dict[str, float]:
    """Validasi input crisp. Melempar InputValidationError bila tidak valid."""
    bersih: Dict[str, float] = {}
    for key, var in INPUT_VARIABLES.items():
        if key not in inputs or inputs[key] is None:
            raise InputValidationError(f"Input '{var.label}' belum diisi.")
        try:
            nilai = float(inputs[key])
        except (TypeError, ValueError):
            raise InputValidationError(f"Input '{var.label}' bukan angka: {inputs[key]!r}")
        if not np.isfinite(nilai):
            raise InputValidationError(f"Input '{var.label}' bukan bilangan hingga (NaN/Inf).")
        lo, hi = var.universe
        if nilai < lo or nilai > hi:
            raise InputValidationError(
                f"{var.label} = {nilai:g} berada di luar semesta pembicaraan "
                f"[{format_angka(lo)}, {format_angka(hi)}] ({var.unit})."
            )
        bersih[key] = nilai
    return bersih


def fuzzify_inputs(inputs: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
    """Tahap 1 - Fuzzifikasi seluruh variabel input."""
    return {key: var.fuzzify(inputs[key]) for key, var in INPUT_VARIABLES.items()}


def evaluate_rules(
    fuzzified: Mapping[str, Mapping[str, float]], rules: Sequence[Rule]
) -> List[RuleActivation]:
    """Tahap 2 - Operasi AND (MIN) untuk setiap aturan -> alpha-predikat."""
    hasil: List[RuleActivation] = []
    for rule in rules:
        mus = {key: float(fuzzified[key][himp]) for key, himp in rule.antecedents}
        alpha = min(mus.values())
        pembatas = min(mus, key=mus.get)  # variabel dengan mu terkecil
        hasil.append(RuleActivation(rule, mus, alpha, pembatas))
    return hasil


def implication_min(alpha: float, consequent: str) -> np.ndarray:
    """Tahap 3 - Implikasi MIN: potong himpunan output setinggi alpha."""
    return np.minimum(alpha, OUTPUT_MU[consequent])


def aggregate_max(kurva_implikasi: Sequence[np.ndarray]) -> np.ndarray:
    """Tahap 4 - Agregasi MAX atas seluruh kurva implikasi aturan aktif."""
    if len(kurva_implikasi) == 0:
        raise FuzzyInferenceError("Agregasi kosong: tidak ada kurva implikasi.")
    return np.max(np.vstack(kurva_implikasi), axis=0)


def defuzzify_centroid(aggregated: np.ndarray) -> Tuple[float, float, float]:
    """
    Tahap 5 - Centroid diskrit pada Z_GRID:  Z* = sum(z * mu) / sum(mu).
    Mengembalikan (Z*, pembilang, penyebut). Melempar error bila penyebut ~ 0.
    """
    if aggregated.shape != Z_GRID.shape:
        raise FuzzyInferenceError("Grid agregasi tidak sama dengan grid defuzzifikasi.")
    pembilang = float(np.sum(Z_GRID * aggregated))
    penyebut = float(np.sum(aggregated))
    if penyebut <= EPS:
        raise FuzzyInferenceError(
            "Defuzzifikasi gagal: luas daerah agregasi bernilai nol, skor tidak dapat dihitung."
        )
    z_star = pembilang / penyebut
    if not (GRID_MIN - EPS <= z_star <= GRID_MAX + EPS):
        raise FuzzyInferenceError(f"Hasil centroid di luar rentang 0–100: {z_star}")
    return z_star, pembilang, penyebut


def classify(z_star: float) -> Tuple[str, Dict[str, float]]:
    """Kategori = himpunan output dengan μ terbesar pada Z*; bila seri, ambil kategori lebih tinggi."""
    mus = {s.name: float(s.membership(z_star)) for s in OUTPUT_VARIABLE.sets}
    terbaik, mu_terbaik = "", -1.0
    for nama, mu in mus.items():  # urutan menaik; >= sehingga yang lebih tinggi menang saat seri
        if mu >= mu_terbaik - 1e-12:
            terbaik, mu_terbaik = nama, mu
    return terbaik, mus


def category_boundaries() -> List[Tuple[float, str, str]]:
    """Titik perpindahan kategori sepanjang grid: [(z, kategori_sebelum, kategori_sesudah)]."""
    batas: List[Tuple[float, str, str]] = []
    sebelumnya = classify(float(Z_GRID[0]))[0]
    for z in Z_GRID[1:]:
        kini = classify(float(z))[0]
        if kini != sebelumnya:
            batas.append((float(z), sebelumnya, kini))
            sebelumnya = kini
    return batas


def infer(inputs: Mapping[str, float], rules: Sequence[Rule]) -> InferenceResult:
    """Menjalankan seluruh tahapan Mamdani untuk satu mahasiswa."""
    if not rules:
        raise FuzzyInferenceError("Basis aturan kosong.")
    bersih = validate_inputs(inputs)
    fuzzified = fuzzify_inputs(bersih)
    activations = evaluate_rules(fuzzified, rules)
    aktif = [a for a in activations if a.active]
    if not aktif:
        raise FuzzyInferenceError(
            "Tidak ada aturan yang aktif untuk input ini. Periksa fungsi keanggotaan dan basis aturan."
        )
    clipped = {a.rule.id: implication_min(a.alpha, a.rule.consequent) for a in aktif}
    aggregated = aggregate_max(list(clipped.values()))
    z_star, pembilang, penyebut = defuzzify_centroid(aggregated)
    kategori, mu_kategori = classify(z_star)
    return InferenceResult(
        inputs=bersih,
        fuzzified=fuzzified,
        activations=activations,
        clipped=clipped,
        aggregated=aggregated,
        numerator=pembilang,
        denominator=penyebut,
        centroid=z_star,
        category=kategori,
        category_memberships=mu_kategori,
    )


# 8. Evaluasi massal & sensitivitas (selalu memanggil infer() yang sebenarnya)
def evaluate_dataframe(df: pd.DataFrame, rules: Sequence[Rule]) -> pd.DataFrame:
    """Menambahkan kolom Skor, Kategori, Aturan Aktif, dan Catatan Error pada df."""
    hasil = df.copy()
    skor, kategori, n_aktif, catatan = [], [], [], []
    for _, baris in df.iterrows():
        masukan = {k: baris[kolom] for k, kolom in INPUT_COLUMNS.items()}
        try:
            r = infer(masukan, rules)
            skor.append(round(r.centroid, 4))
            kategori.append(r.category)
            n_aktif.append(len(r.active_activations))
            catatan.append("")
        except (InputValidationError, FuzzyInferenceError) as err:
            skor.append(np.nan)
            kategori.append("Error")
            n_aktif.append(0)
            catatan.append(str(err))
    hasil["Skor Kelayakan"] = skor
    hasil["Kategori"] = kategori
    hasil["Aturan Aktif"] = n_aktif
    hasil["Catatan Error"] = catatan
    return hasil


def sensitivity_analysis(
    base_inputs: Mapping[str, float],
    var_key: str,
    values: Sequence[float],
    rules: Sequence[Rule],
) -> pd.DataFrame:
    """Mengubah satu variabel secara bertahap, variabel lain tetap."""
    baris = []
    for v in values:
        masukan = dict(base_inputs)
        masukan[var_key] = float(v)
        r = infer(masukan, rules)
        baris.append(
            {
                "Nilai Input": float(v),
                "Skor Kelayakan": r.centroid,
                "Kategori": r.category,
                "Aturan Aktif": len(r.active_activations),
            }
        )
    return pd.DataFrame(baris)