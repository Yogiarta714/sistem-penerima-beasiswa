from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .fuzzy_engine import INPUT_VARIABLES

SEED = 42  # seed tetap untuk data acak agar dapat direproduksi
DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "mahasiswa_dummy.csv"
AWALAN_KODE = "MHS"

KOLOM_DATASET = [
   "Kode", "Nama", "IPK", "Penghasilan", "Tanggungan", "Prestasi", "Jenis Kasus", "Keterangan",
]
JENIS_KASUS = ["Normal", "Batas", "Ekstrem", "Outlier", "Konflik"]

def buat_dataset_awal(path: Path = DATASET_PATH) -> pd.DataFrame:
   """
   Membaca dataset dummy dari file CSV (dataset/mahasiswa_dummy.csv).
   File divalidasi: kolom lengkap, kode unik, dan seluruh nilai berada dalam semesta.
   """
   if not Path(path).exists():
      raise FileNotFoundError(
         f"File dataset tidak ditemukan: {path}. Pastikan folder 'dataset' berada di folder yang sama dengan app.py."
      )
   df = pd.read_csv(path, encoding="utf-8")
   kurang = [k for k in KOLOM_DATASET if k not in df.columns]
   if kurang:
      raise ValueError(f"Kolom CSV kurang: {', '.join(kurang)}")
   df = df[KOLOM_DATASET].copy()
   df["Kode"] = df["Kode"].astype(str).str.strip()
   df["Nama"] = df["Nama"].astype(str).str.strip()
   df["Keterangan"] = df["Keterangan"].fillna("").astype(str)
   df["Jenis Kasus"] = df["Jenis Kasus"].fillna("Normal").astype(str)
   for kolom in ["IPK", "Penghasilan", "Prestasi"]:
      df[kolom] = pd.to_numeric(df[kolom], errors="raise").astype(float)
   df["Tanggungan"] = pd.to_numeric(df["Tanggungan"], errors="raise").astype(int)
   if df["Kode"].duplicated().any():
      raise ValueError("Kode mahasiswa pada CSV harus unik.")
   for _, r in df.iterrows():
      galat = validasi_record(r["Nama"], r["IPK"], r["Penghasilan"], r["Tanggungan"], r["Prestasi"])
      if galat:
         raise ValueError(f"Data {r['Kode']} tidak valid: " + " ".join(galat))
   return df.reset_index(drop=True)


def validasi_record(
   nama: str, ipk: float, penghasilan: float, tanggungan: float, prestasi: float
) -> List[str]:
   """Daftar pesan kesalahan untuk satu record (kosong bila valid)."""
   galat: List[str] = []
   if not str(nama).strip():
      galat.append("Nama mahasiswa tidak boleh kosong.")
   nilai = {"ipk": ipk, "penghasilan": penghasilan, "tanggungan": tanggungan, "prestasi": prestasi}
   for key, var in INPUT_VARIABLES.items():
      lo, hi = var.universe
      try:
         v = float(nilai[key])
      except (TypeError, ValueError):
         galat.append(f"{var.label} harus berupa angka.")
         continue
      if not np.isfinite(v) or v < lo or v > hi:
         galat.append(f"{var.label} harus berada pada rentang {lo:g} sampai {hi:,.0f}".replace(",", ".") + ".")
   return galat


def kode_berikutnya(df: pd.DataFrame, awalan: str = AWALAN_KODE) -> str:
   """Kode unik berikutnya, mis. MHS031."""
   angka = [
      int(k[len(awalan):])
      for k in df["Kode"].astype(str)
      if k.startswith(awalan) and k[len(awalan):].isdigit()
   ]
   return f"{awalan}{(max(angka) + 1 if angka else 1):03d}"


def buat_data_acak(jumlah: int, df_saat_ini: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
   """
   Membuat jumlah record acak yang deterministik.
   Seed = SEED + banyak data saat ini, sehingga klik berulang menghasilkan data berbeda
   tetapi urutan keseluruhan tetap dapat direproduksi dari dataset awal.
   """
   rng = np.random.default_rng(seed + len(df_saat_ini))
   baris = []
   sementara = df_saat_ini.copy()
   for _ in range(jumlah):
      kode = kode_berikutnya(sementara)
      rec = {
         "Kode": kode,
         "Nama": f"Mahasiswa Acak {kode[len(AWALAN_KODE):]}",
         "IPK": round(float(rng.uniform(0.0, 4.0)), 2),
         "Penghasilan": float(int(rng.integers(0, 151)) * 100_000),
         "Tanggungan": int(rng.integers(0, 9)),
         "Prestasi": float(int(rng.integers(0, 101))),
         "Jenis Kasus": "Normal",
         "Keterangan": f"Data acak (seed {seed + len(df_saat_ini)}).",
      }
      baris.append(rec)
      sementara = pd.concat([sementara, pd.DataFrame([rec])], ignore_index=True)
   return pd.DataFrame(baris, columns=KOLOM_DATASET)


# Skenario pengujian (oracle = harapan berdasarkan logika/rumus, BUKAN hasil program)
#   harapan_kategori : kategori yang masuk akal secara logika basis aturan
#   harapan_mu       : derajat keanggotaan yang dapat dihitung langsung dari rumus MF
def _s(no, nama, kelompok, ipk, peng, tang, pres, kategori, mu=None, catatan=""):
   return {
      "No": no,
      "Skenario": nama,
      "Kelompok": kelompok,
      "inputs": {"ipk": ipk, "penghasilan": peng, "tanggungan": tang, "prestasi": pres},
      "harapan_kategori": tuple(kategori),
      "harapan_mu": mu or {},
      "Catatan": catatan,
   }


SKENARIO_PENGUJIAN: List[dict] = [
   _s(1, "Semua variabel di batas bawah", "Ekstrem", 0.0, 0, 0, 0,
      ["Tidak Layak"],
      {("ipk", "Rendah"): 1.0, ("penghasilan", "Rendah"): 1.0, ("tanggungan", "Sedikit"): 1.0, ("prestasi", "Rendah"): 1.0},
      "Titik awal semesta; hanya satu aturan aktif."),
   _s(2, "Semua variabel di batas atas", "Ekstrem", 4.0, 15_000_000, 8, 100,
      ["Layak", "Sangat Layak"],
      {("ipk", "Tinggi"): 1.0, ("penghasilan", "Tinggi"): 1.0, ("tanggungan", "Banyak"): 1.0, ("prestasi", "Tinggi"): 1.0},
      "Akademik penuh dan tanggungan penuh, tetapi penghasilan tinggi."),
   _s(3, "Kasus ideal", "Ekstrem", 4.0, 0, 8, 100,
      ["Sangat Layak"],
      {("ipk", "Tinggi"): 1.0, ("penghasilan", "Rendah"): 1.0, ("tanggungan", "Banyak"): 1.0, ("prestasi", "Tinggi"): 1.0},
      "Semua kriteria paling menguntungkan."),
   _s(4, "Kasus terburuk", "Ekstrem", 0.0, 15_000_000, 0, 0,
      ["Tidak Layak"], None, "Semua kriteria paling tidak menguntungkan."),
   _s(5, "IPK tinggi + penghasilan tinggi", "Kombinasi", 3.8, 14_000_000, 1, 55,
      ["Cukup Layak", "Layak"], None, "Akademik baik tetapi tidak membutuhkan bantuan ekonomi."),
   _s(6, "IPK rendah + penghasilan rendah", "Kombinasi", 1.0, 500_000, 3, 40,
      ["Tidak Layak", "Cukup Layak"], None, "Kebutuhan ekonomi tinggi, akademik lemah."),
   _s(7, "Tanggungan banyak + prestasi rendah", "Kombinasi", 3.2, 2_000_000, 8, 5,
      ["Layak", "Sangat Layak"], None,
      "Penghasilan rendah + tanggungan banyak (bobot 45) mengangkat hasil walau prestasi nihil."),
   _s(8, "Prestasi tinggi + IPK rendah", "Konflik", 1.0, 10_000_000, 1, 98,
      ["Tidak Layak", "Cukup Layak"], None, "Prestasi tidak menutup IPK rendah dan ekonomi mampu."),
   _s(9, "IPK tinggi, ekonomi sulit, prestasi rendah", "Konflik", 3.8, 0, 6, 5,
      ["Layak", "Sangat Layak"], None, "Dua kriteria kuat, satu lemah."),
   _s(10, "Akademik sangat baik, ekonomi mampu", "Konflik", 3.95, 14_000_000, 0, 95,
      ["Cukup Layak", "Layak"], None, "Kriteria akademik dan ekonomi berlawanan arah."),
   _s(11, "Akademik kurang, ekonomi sangat butuh", "Konflik", 1.5, 500_000, 8, 10,
      ["Tidak Layak", "Cukup Layak"], None, "Veto akademik membatasi hasil."),
   _s(12, "Batas IPK = 1,5", "Titik Batas", 1.5, 5_000_000, 3, 50,
      ["Tidak Layak", "Cukup Layak"],
      {("ipk", "Rendah"): 1.0, ("ipk", "Sedang"): 0.0, ("ipk", "Tinggi"): 0.0},
      "Ujung plateau Rendah; Sedang baru mulai naik."),
   _s(13, "Batas IPK = 2,5", "Titik Batas", 2.5, 5_000_000, 3, 50,
      ["Cukup Layak"],
      {("ipk", "Rendah"): 0.0, ("ipk", "Sedang"): 1.0, ("ipk", "Tinggi"): 0.0},
      "Puncak Sedang; seluruh variabel di puncak Sedang -> satu aturan aktif."),
   _s(14, "Batas IPK = 3,5", "Titik Batas", 3.5, 5_000_000, 3, 50,
      ["Cukup Layak", "Layak"],
      {("ipk", "Rendah"): 0.0, ("ipk", "Sedang"): 0.0, ("ipk", "Tinggi"): 1.0},
      "Awal plateau Tinggi."),
   _s(15, "Overlap IPK = 2,0", "Overlap", 2.0, 5_000_000, 3, 50,
      ["Cukup Layak"],
      {("ipk", "Rendah"): 0.5, ("ipk", "Sedang"): 0.5, ("ipk", "Tinggi"): 0.0},
      "Tepat di tengah overlap Rendah-Sedang."),
   _s(16, "Overlap penghasilan = Rp3,5 juta", "Overlap", 2.5, 3_500_000, 3, 50,
      ["Cukup Layak", "Layak"],
      {("penghasilan", "Rendah"): 0.5, ("penghasilan", "Sedang"): 0.5, ("penghasilan", "Tinggi"): 0.0},
      "Tepat di tengah overlap Rendah-Sedang."),
   _s(17, "Batas penghasilan = Rp9 juta", "Titik Batas", 2.5, 9_000_000, 3, 50,
      ["Tidak Layak", "Cukup Layak"],
      {("penghasilan", "Rendah"): 0.0, ("penghasilan", "Sedang"): 0.0, ("penghasilan", "Tinggi"): 1.0},
      "Awal plateau Tinggi."),
   _s(18, "Overlap tanggungan = 2 orang", "Overlap", 2.5, 5_000_000, 2, 50,
      ["Cukup Layak"],
      {("tanggungan", "Sedikit"): 0.5, ("tanggungan", "Sedang"): 0.5, ("tanggungan", "Banyak"): 0.0},
      "Tepat di tengah overlap Sedikit-Sedang."),
   _s(19, "Batas tanggungan = 5 orang", "Titik Batas", 2.5, 5_000_000, 5, 50,
      ["Cukup Layak", "Layak"],
      {("tanggungan", "Sedikit"): 0.0, ("tanggungan", "Sedang"): 0.0, ("tanggungan", "Banyak"): 1.0},
      "Awal plateau Banyak."),
   _s(20, "Overlap prestasi = 35", "Overlap", 2.5, 5_000_000, 3, 35,
      ["Cukup Layak"],
      {("prestasi", "Rendah"): 0.5, ("prestasi", "Sedang"): 0.5, ("prestasi", "Tinggi"): 0.0},
      "Tepat di tengah overlap Rendah-Sedang."),
   _s(21, "Semua derajat 0,5 (16 aturan aktif)", "Overlap", 3.0, 3_500_000, 4, 65,
      ["Cukup Layak", "Layak"],
      {("ipk", "Sedang"): 0.5, ("ipk", "Tinggi"): 0.5,
      ("penghasilan", "Rendah"): 0.5, ("penghasilan", "Sedang"): 0.5,
      ("tanggungan", "Sedang"): 0.5, ("tanggungan", "Banyak"): 0.5,
      ("prestasi", "Sedang"): 0.5, ("prestasi", "Tinggi"): 0.5},
      "Seluruh variabel di tengah overlap; kombinasi 2^4 = 16 aturan aktif."),
   _s(22, "Penghasilan Rp0 dan tanpa tanggungan", "Ekstrem", 3.0, 0, 0, 50,
      ["Layak", "Sangat Layak"], None,
      "Penghasilan Rp0 (bobot 60) lebih dominan daripada tanggungan nol."),
   _s(23, "IPK 4,00 tanpa prestasi", "Kombinasi", 4.0, 4_000_000, 4, 0,
      ["Layak", "Sangat Layak"], None,
      "IPK maksimum (bobot 70) ditambah ekonomi rendah-sedang menutup prestasi nol."),
   _s(24, "Ekonomi sulit, akademik lemah, prestasi tinggi", "Konflik", 1.8, 800_000, 7, 90,
      ["Cukup Layak", "Layak"], None, "Prestasi tinggi mengangkat hasil, IPK membatasi."),
]


# Profil tetap untuk analisis sensitivitas
PROFIL_SENSITIVITAS: Dict[str, Dict[str, float]] = {
   "A. Ekonomi sulit": {"ipk": 3.0, "penghasilan": 1_500_000, "tanggungan": 5, "prestasi": 60},
   "B. Menengah": {"ipk": 3.0, "penghasilan": 5_000_000, "tanggungan": 3, "prestasi": 50},
   "C. Ekonomi mampu": {"ipk": 3.0, "penghasilan": 12_000_000, "tanggungan": 1, "prestasi": 60},
   "D. Akademik lemah": {"ipk": 1.5, "penghasilan": 3_000_000, "tanggungan": 4, "prestasi": 30},
   "E. Akademik kuat": {"ipk": 3.7, "penghasilan": 6_000_000, "tanggungan": 2, "prestasi": 85},
}