from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from spk import data as dt
from spk import fuzzy_engine as fe
from spk import validation as val
from spk import visualization as viz
from spk.fuzzy_engine import (
    INPUT_COLUMNS,
    INPUT_VARIABLES,
    OUTPUT_VARIABLE,
    FuzzyInferenceError,
    InferenceResult,
    InputValidationError,
    format_angka,
)
from spk.rules import (
    AMBANG_KONSEKUEN,
    BATAS_KONSEKUEN_IPK_RENDAH,
    BOBOT,
    RULES,
    cek_kelengkapan,
    distribusi_konsekuen,
    rules_to_dataframe,
)

st.set_page_config(
    page_title="SPK Kelayakan Beasiswa - Fuzzy Mamdani",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .kartu {border: 1px solid rgba(128,128,128,0.35); border-radius: 8px;
            padding: 12px 16px; margin-bottom: 8px;}
    .kartu-label {font-size: 0.80rem; opacity: 0.75;}
    .kartu-nilai {font-size: 1.55rem; font-weight: 600; line-height: 1.3;}
    .catatan {border-left: 4px solid #D9972B; padding: 6px 12px; margin: 8px 0;
            background: rgba(217,151,43,0.08);}
    </style>
    """,
    unsafe_allow_html=True,
)

DISCLAIMER = (
    "Skor yang dihasilkan merupakan rekomendasi pendukung keputusan, bukan keputusan "
    "mutlak penerimaan beasiswa. Keputusan akhir tetap berada pada panitia/pengambil keputusan."
)

DEFAULT_INPUT = {"ipk": 3.0, "penghasilan": 3_000_000, "tanggungan": 3, "prestasi": 60.0}


# Helper ui
def rupiah(nilai: float) -> str:
    return "Rp" + f"{nilai:,.0f}".replace(",", ".")


def tampil_fig(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def kartu(label: str, nilai: str, warna: str | None = None) -> None:
    gaya = f"color:{warna};" if warna else ""
    st.markdown(
        f"<div class='kartu'><div class='kartu-label'>{label}</div>"
        f"<div class='kartu-nilai' style='{gaya}'>{nilai}</div></div>",
        unsafe_allow_html=True,
    )


def tampil_tabel(df: pd.DataFrame, desimal: int = 4) -> None:
    st.dataframe(df.round(desimal), hide_index=True)


def init_state() -> None:
    if "df_mahasiswa" not in st.session_state:
        st.session_state["df_mahasiswa"] = dt.buat_dataset_awal()


@st.cache_data(show_spinner=False)
def evaluasi_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluasi seluruh dataset dengan algoritma Mamdani sebenarnya (di-cache per isi data)."""
    return fe.evaluate_dataframe(df, RULES)


def input_form_ringkas(prefix: str, default: dict) -> dict:
    """Empat input crisp tanpa tombol"""
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ipk = st.number_input("IPK (0-4)", 0.0, 4.0, float(default["ipk"]), 0.01,
                            format="%.2f", key=f"{prefix}_ipk")
    with k2:
        peng = st.number_input("Penghasilan orang tua / bulan (Rp)", 0, 15_000_000,
                            int(default["penghasilan"]), 100_000, key=f"{prefix}_peng")
    with k3:
        tang = st.number_input("Jumlah tanggungan (0-8)", 0, 8, int(default["tanggungan"]), 1,
                            key=f"{prefix}_tang")
    with k4:
        pres = st.number_input("Prestasi (0-100)", 0.0, 100.0, float(default["prestasi"]), 0.5,
                            format="%.1f", key=f"{prefix}_pres")
    return {"ipk": ipk, "penghasilan": peng, "tanggungan": tang, "prestasi": pres}


# Komponen tahapan Mamdani
def tampil_input(inputs: dict, nama: str = "", nim: str = "") -> None:
    baris = {
        "Nama": nama or "-",
        "NIM / Kode": nim or "-",
        "IPK": f"{inputs['ipk']:.2f}",
        "Penghasilan Orang Tua": rupiah(inputs["penghasilan"]),
        "Tanggungan": f"{inputs['tanggungan']:g} orang",
        "Prestasi": f"{inputs['prestasi']:g}",
    }
    st.dataframe(pd.DataFrame([baris]), hide_index=True)


def tampil_fuzzifikasi(result: InferenceResult, grafik: bool = True) -> None:
    st.caption("Setiap nilai crisp diubah menjadi derajat keanggotaan μ pada tiap himpunan linguistik.")
    tabel = result.fuzzification_table()
    # nama himpunan berbeda antar variabel -> tampilkan satu tabel per variabel
    kolom = st.columns(len(INPUT_VARIABLES))
    for kol, (key, var) in zip(kolom, INPUT_VARIABLES.items()):
        with kol:
            sub = tabel[tabel["Variabel"] == var.label][["Himpunan", "Derajat Keanggotaan (μ)"]]
            st.markdown(f"**{var.label}** = {result.inputs[key]:g}")
            tampil_tabel(sub)
    if grafik:
        with st.expander("Grafik fuzzifikasi (titik = μ pada nilai crisp)"):
            tampil_fig(viz.plot_fuzzification(result))


def tampil_aturan_aktif(result: InferenceResult, termasuk_tidak_aktif: bool = False) -> None:
    n = len(result.active_activations)
    st.caption(
        f"α-predikat = MIN(μ IPK, μ Penghasilan, μ Tanggungan, μ Prestasi). "
        f"Dari 81 aturan, **{n} aturan aktif** (α > 0)."
    )
    tampil_tabel(result.activation_table(only_active=not termasuk_tidak_aktif))


def tampil_implikasi(result: InferenceResult) -> None:
    st.caption("Implikasi MIN: himpunan output tiap aturan aktif dipotong setinggi α-predikatnya.")
    tampil_tabel(result.implication_table())


def tampil_agregasi(result: InferenceResult) -> None:
    st.caption("Agregasi MAX: kurva implikasi seluruh aturan aktif digabung dengan mengambil nilai maksimum.")
    tampil_tabel(result.aggregation_by_output())
    tampil_fig(viz.plot_aggregation(result))


def tampil_defuzzifikasi(result: InferenceResult) -> None:
    st.caption(
        f"Centroid diskrit pada grid {fe.GRID_MIN:g}-{fe.GRID_MAX:g} dengan interval {fe.GRID_STEP:g} "
        f"({fe.GRID_POINTS} titik):  Z* = Σ(z·μ(z)) / Σμ(z)."
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kartu("Σ z·μ(z)", f"{result.numerator:,.4f}")
    with k2:
        kartu("Σ μ(z)", f"{result.denominator:,.4f}")
    with k3:
        kartu("Skor kelayakan Z*", f"{result.centroid:.2f}", viz.CATEGORY_COLORS[result.category])
    with k4:
        kartu("Kategori", result.category, viz.CATEGORY_COLORS[result.category])
    mu_kat = ", ".join(f"{n}: {m:.3f}" for n, m in result.category_memberships.items())
    st.caption(f"Derajat keanggotaan Z* pada himpunan output → {mu_kat}. Kategori = himpunan dengan μ terbesar.")


def jelaskan_kasus(result: InferenceResult) -> list[str]:
    """Penjelasan singkat hasil satu kasus berdasarkan hasil perhitungan."""
    teks = []
    dominan = []
    for key, var in INPUT_VARIABLES.items():
        nama, mu = max(result.fuzzified[key].items(), key=lambda kv: kv[1])
        dominan.append(f"{var.label}: {nama} (μ={mu:.2f})")
    teks.append("Himpunan input dominan → " + "; ".join(dominan) + ".")
    agr = result.aggregation_by_output()
    agr = agr[agr["Jumlah Aturan Aktif"] > 0].sort_values("α Maksimum (MAX)", ascending=False)
    teks.append(
        "Kontribusi himpunan output (α maksimum) → "
        + "; ".join(f"{r['Himpunan Output']}: {r['α Maksimum (MAX)']:.2f}" for _, r in agr.iterrows()) + "."
    )
    if result.fuzzified["ipk"]["Rendah"] > 0:
        teks.append(
            f"Ada aturan dengan IPK Rendah yang aktif; konsekuennya dibatasi maksimum "
            f"'{BATAS_KONSEKUEN_IPK_RENDAH}' oleh veto akademik."
        )
    teks.append(f"Defuzzifikasi menghasilkan Z* = {result.centroid:.2f} → {result.category}.")
    return teks


# Halaman 1 - Dashboard
def halaman_dashboard() -> None:
    st.title("Sistem Pendukung Keputusan Kelayakan Penerima Beasiswa")
    st.caption("Logika Fuzzy Metode Mamdani")
    st.write(
        "Aplikasi ini menilai kelayakan calon penerima beasiswa berdasarkan empat kriteria: "
        "IPK, penghasilan orang tua, jumlah tanggungan keluarga, dan prestasi mahasiswa. "
        "Seluruh tahapan Mamdani dihitung manual dengan NumPy dan dapat diperiksa pada halaman "
        "*Detail Proses Perhitungan*. Data yang digunakan adalah **data dummy**, bukan data mahasiswa asli."
    )
    st.markdown(f"<div class='catatan'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    df = st.session_state["df_mahasiswa"]
    if df.empty:
        st.info("Dataset kosong. Tambahkan data pada halaman *Dataset Mahasiswa*.")
        return
    hasil = evaluasi_dataset(df)
    valid = hasil[hasil["Kategori"] != "Error"]

    k = st.columns(3)
    with k[0]:
        kartu("Jumlah data mahasiswa yang diuji", str(len(hasil)))
    with k[1]:
        kartu("Rata-rata skor kelayakan", f"{valid['Skor Kelayakan'].mean():.2f}" if len(valid) else "-")
    with k[2]:
        kartu("Data gagal diproses", str(int((hasil["Kategori"] == "Error").sum())))
    k = st.columns(4)
    for kol, nama in zip(k, viz.URUTAN_KATEGORI):
        with kol:
            kartu(nama, str(int((hasil["Kategori"] == nama).sum())), viz.CATEGORY_COLORS[nama])

    c1, c2 = st.columns(2)
    with c1:
        tampil_fig(viz.plot_category_distribution(valid))
    with c2:
        tampil_fig(viz.plot_score_histogram(valid))

    with st.expander("Ringkasan metode Mamdani yang digunakan", expanded=True):
        st.markdown(
            f"""
        1. **Fuzzifikasi** - nilai crisp 4 variabel diubah menjadi derajat keanggotaan (segitiga/trapesium).
        2. **Operasi AND = MIN** - α-predikat tiap aturan = MIN dari 4 derajat keanggotaan antecedent.
        3. **Implikasi = MIN** - himpunan output dipotong setinggi α-predikat.
        4. **Agregasi = MAX** - seluruh kurva implikasi aturan aktif digabung dengan nilai maksimum.
        5. **Defuzzifikasi = Centroid** - Z\\* = Σ(z·μ(z)) / Σμ(z) pada grid {fe.GRID_MIN:g}-{fe.GRID_MAX:g} interval {fe.GRID_STEP:g} ({fe.GRID_POINTS} titik).

        Basis aturan lengkap berisi **{len(RULES)} aturan** (3 × 3 × 3 × 3).
        """
    )


# Halaman 2 - Simulasi Individu
def halaman_simulasi() -> None:
    st.title("Simulasi Individu")
    st.caption("Masukkan data seorang mahasiswa, lalu tekan *Proses Penilaian* untuk melihat seluruh tahapan Mamdani.")

    with st.form("form_simulasi"):
        c1, c2 = st.columns(2)
        with c1:
            nama = st.text_input("Nama mahasiswa", placeholder="Contoh: Ayu Pratiwi (dummy)")
            nim = st.text_input("NIM / kode mahasiswa (opsional)")
            ipk = st.slider("IPK", 0.0, 4.0, 3.0, 0.01)
        with c2:
            peng = st.number_input("Penghasilan orang tua per bulan (Rp)", 0, 15_000_000,
                                3_000_000, 100_000)
            tang = st.number_input("Jumlah tanggungan keluarga (orang)", 0, 8, 3, 1)
            pres = st.slider("Prestasi mahasiswa (0-100)", 0.0, 100.0, 60.0, 0.5)
        kirim = st.form_submit_button("Proses Penilaian")

    if kirim:
        masukan = {"ipk": ipk, "penghasilan": peng, "tanggungan": tang, "prestasi": pres}
        if not nama.strip():
            st.error("Nama mahasiswa wajib diisi.")
        else:
            try:
                hasil = fe.infer(masukan, RULES)
            except (InputValidationError, FuzzyInferenceError) as err:
                st.session_state.pop("hasil_individu", None)
                st.error(f"Proses penilaian gagal: {err}")
            else:
                st.session_state["hasil_individu"] = {
                    "nama": nama.strip(), "nim": nim.strip(), "inputs": masukan, "result": hasil,
                }
                st.session_state["input_terakhir"] = masukan

    paket = st.session_state.get("hasil_individu")
    if not paket:
        st.info("Belum ada hasil. Isi formulir lalu tekan *Proses Penilaian*.")
        return

    result: InferenceResult = paket["result"]
    st.divider()
    warna = viz.CATEGORY_COLORS[result.category]
    k = st.columns(3)
    with k[0]:
        kartu("Skor kelayakan (centroid)", f"{result.centroid:.2f}", warna)
    with k[1]:
        kartu("Kategori kelayakan", result.category, warna)
    with k[2]:
        kartu("Aturan aktif", f"{len(result.active_activations)} dari {len(RULES)}")
    st.markdown(f"<div class='catatan'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    st.subheader("1. Data input")
    tampil_input(paket["inputs"], paket["nama"], paket["nim"])
    st.subheader("2. Hasil fuzzifikasi")
    tampil_fuzzifikasi(result)
    st.subheader("3. Aturan aktif dan α-predikat")
    tampil_aturan_aktif(result)
    st.subheader("4. Hasil implikasi")
    tampil_implikasi(result)
    st.subheader("5. Hasil agregasi")
    tampil_agregasi(result)
    st.subheader("6. Defuzzifikasi centroid dan kategori akhir")
    tampil_defuzzifikasi(result)
    st.subheader("Ringkasan interpretasi")
    for baris in jelaskan_kasus(result):
        st.write("• " + baris)


# Halaman 3 - Visualisasi Fungsi Keanggotaan
def halaman_mf() -> None:
    st.title("Visualisasi Fungsi Keanggotaan")
    st.caption("Pilih variabel untuk melihat grafik, parameter, persamaan, dan interpretasinya.")

    semua = list(INPUT_VARIABLES.values()) + [OUTPUT_VARIABLE]
    pilihan = st.selectbox("Variabel", [v.label for v in semua])
    var = next(v for v in semua if v.label == pilihan)
    lo, hi = var.universe

    with st.expander("Uji nilai crisp pada variabel ini (opsional)"):
        if var.key == "penghasilan":
            x = st.number_input("Nilai crisp (Rp)", int(lo), int(hi), 3_500_000, 100_000, key="mf_x")
        elif var.key == "tanggungan":
            x = st.number_input("Nilai crisp (orang)", int(lo), int(hi), 2, 1, key="mf_x")
        else:
            x = st.number_input("Nilai crisp", float(lo), float(hi), float(lo + (hi - lo) / 2),
                                (hi - lo) / 100, key="mf_x")
        mu = var.fuzzify(x)
        st.write(" · ".join(f"μ {n} = {m:.4f}" for n, m in mu.items()))
    tampil_fig(viz.plot_variable(var, crisp=float(x)))

    st.subheader("Parameter, rentang, dan kategori linguistik")
    baris = []
    for s in var.sets:
        a, b, c, d = s.trapezoid_params()
        inti = f"{format_angka(b)}" if b == c else f"{format_angka(b)} - {format_angka(c)}"
        baris.append(
            {
                "Himpunan": s.name,
                "Jenis": "Segitiga" if s.kind == "triangular" else "Trapesium",
                "Parameter": "(" + ", ".join(format_angka(p) for p in s.params) + ")",
                "Rentang pendukung (μ > 0)": f"{format_angka(a)} - {format_angka(d)}",
                "Inti (μ = 1)": inti,
            }
        )
    st.dataframe(pd.DataFrame(baris), hide_index=True)

    st.subheader("Persamaan matematis")
    kolom = st.columns(len(var.sets))
    for kol, s in zip(kolom, var.sets):
        with kol:
            st.markdown(f"**{s.name}**")
            st.code("\n".join(s.equation_lines()), language=None)

    st.subheader("Alasan pemilihan parameter")
    st.write(var.rationale)
    st.caption("Parameter di atas adalah asumsi perancang, bukan ketentuan dosen.")
    st.subheader("Interpretasi")
    st.write(var.interpretation)
    if var.key == "penghasilan":
        st.write("Semakin rendah penghasilan orang tua, semakin tinggi prioritas bantuan (diatur oleh basis aturan).")
    if var.key == "tanggungan":
        st.write("Semakin banyak tanggungan keluarga, semakin tinggi prioritas bantuan (diatur oleh basis aturan).")


# Halaman 4 - Basis Aturan
def halaman_aturan() -> None:
    st.title("Basis Aturan Fuzzy")
    st.caption("81 aturan = 3 × 3 × 3 × 3 kombinasi kategori IPK, penghasilan, tanggungan, dan prestasi.")

    kel = cek_kelengkapan(RULES)
    k = st.columns(3)
    with k[0]:
        kartu("Jumlah aturan", str(kel["jumlah_aturan"]))
    with k[1]:
        kartu("Kombinasi unik", str(kel["jumlah_kombinasi_unik"]))
    with k[2]:
        kartu("Kelengkapan basis aturan", "Lengkap" if kel["lengkap"] else "TIDAK LENGKAP")

    with st.expander("Cara penentuan konsekuen (pemetaan berbobot)", expanded=False):
        st.markdown(
            f"""
            Konsekuen **tidak ditentukan acak**. Setiap kategori input diberi *tingkat manfaat* 0, 1, atau 2:

            - IPK dan Prestasi: Rendah = 0, Sedang = 1, Tinggi = 2.
            - Penghasilan orang tua: **Tinggi = 0**, Sedang = 1, **Rendah = 2** (penghasilan rendah → prioritas naik).
            - Tanggungan: Sedikit = 0, Sedang = 1, **Banyak = 2** (tanggungan banyak → prioritas naik).

            Skor bobot aturan = Σ (bobot × tingkat manfaat), rentang 0-200, dengan bobot
            IPK = {BOBOT['ipk']}, Penghasilan = {BOBOT['penghasilan']}, Prestasi = {BOBOT['prestasi']},
            Tanggungan = {BOBOT['tanggungan']} (total 100).

            Ambang konsekuen: skor < {AMBANG_KONSEKUEN[0][0]} → Tidak Layak; < {AMBANG_KONSEKUEN[1][0]} → Cukup Layak;
            < {AMBANG_KONSEKUEN[2][0]} → Layak; selebihnya → Sangat Layak.

            **Veto akademik:** aturan dengan IPK Rendah dibatasi maksimum *{BATAS_KONSEKUEN_IPK_RENDAH}*.

            Pemetaan ini membuat keempat variabel selalu berpengaruh, IPK diberi bobot terbesar karena beasiswa
            umumnya mensyaratkan kelayakan akademik. **Bobot, ambang, dan veto adalah asumsi perancang**,
            bukan ketentuan dosen, dan dapat diubah di `rules.py`.
            """
        )
        st.dataframe(distribusi_konsekuen(RULES), hide_index=True)

    df = rules_to_dataframe(RULES)
    st.subheader("Pencarian dan filter")
    c = st.columns(5)
    pil = {}
    for kol, (kolom, label) in zip(c, [("IPK", "IPK"), ("Penghasilan", "Penghasilan"),
                                    ("Tanggungan", "Tanggungan"), ("Prestasi", "Prestasi"),
                                    ("Kesimpulan", "Kesimpulan")]):
        with kol:
            opsi = ["Semua"] + sorted(df[kolom].unique(), key=lambda v: list(df[kolom].unique()).index(v))
            if kolom == "Kesimpulan":
                opsi = ["Semua"] + viz.URUTAN_KATEGORI
            pil[kolom] = st.selectbox(label, opsi, key=f"fil_{kolom}")
    cari = st.text_input("Cari nomor aturan (mis. 12 atau R12)")
    hasil = df
    for kolom, nilai in pil.items():
        if nilai != "Semua":
            hasil = hasil[hasil[kolom] == nilai]
    if cari.strip():
        angka = cari.strip().upper().replace("R", "")
        hasil = hasil[hasil["No"].astype(str) == angka] if angka.isdigit() else hasil.iloc[0:0]
    st.write(f"Menampilkan **{len(hasil)}** dari {len(df)} aturan.")
    st.dataframe(hasil.drop(columns=["No"]), hide_index=True, height=520)


# Halaman 5 - Detail Proses Perhitungan
def halaman_detail() -> None:
    st.title("Detail Proses Perhitungan")
    st.caption("Seluruh tahapan Mamdani ditampilkan transparan, dihitung langsung oleh program.")
    awal = st.session_state.get("input_terakhir", DEFAULT_INPUT)
    masukan = input_form_ringkas("det", awal)
    tampil_tidak_aktif = st.checkbox("Tampilkan juga aturan yang tidak aktif (α = 0)")

    try:
        result = fe.infer(masukan, RULES)
    except (InputValidationError, FuzzyInferenceError) as err:
        st.error(f"Proses perhitungan gagal: {err}")
        return

    st.subheader("Tahap 1 - Input crisp")
    tampil_input(masukan)
    st.subheader("Tahap 2 - Hasil fuzzifikasi")
    tampil_fuzzifikasi(result, grafik=True)
    st.subheader("Tahap 3 - α-predikat setiap aturan (operasi AND = MIN)")
    tampil_aturan_aktif(result, tampil_tidak_aktif)
    st.subheader("Tahap 4 - Hasil implikasi MIN")
    tampil_implikasi(result)
    st.subheader("Tahap 5 - Hasil agregasi MAX")
    st.caption("Agregasi mengambil nilai maksimum dari seluruh kurva implikasi aturan aktif (bukan hanya α terbesar).")
    tampil_tabel(result.aggregation_by_output())
    st.subheader("Tahap 6 - Defuzzifikasi centroid")
    tampil_defuzzifikasi(result)

    st.subheader("Grafik proses")
    t1, t2, t3 = st.tabs(["Output awal", "Kurva implikasi", "Agregasi + centroid"])
    with t1:
        tampil_fig(viz.plot_output_sets())
    with t2:
        tampil_fig(viz.plot_implication(result))
    with t3:
        tampil_fig(viz.plot_aggregation(result))

    st.subheader("Tabel nilai numerik pada grid output")
    langkah = st.selectbox("Tampilkan setiap … satuan z", [10.0, 5.0, 1.0, 0.1], index=1)
    tampil_tabel(result.grid_table(langkah), 4)
    st.caption(
        f"Kolom R# = kurva implikasi tiap aturan aktif; 'Agregasi (MAX)' = max antar kolom R#; "
        f"Z* = Σ(z·μ)/Σμ dihitung dari seluruh {fe.GRID_POINTS} titik grid (bukan hanya baris yang ditampilkan)."
    )


# Halaman 6 - Dataset Mahasiswa
def halaman_dataset() -> None:
    st.title("Dataset Mahasiswa (Data Dummy)")
    st.caption(
        "Seluruh data adalah simulasi, bukan data mahasiswa asli. Skor dan kategori dihitung langsung "
        "oleh algoritma Mamdani, tidak ditulis manual."
    )
    df = st.session_state["df_mahasiswa"]
    tab1, tab2, tab3 = st.tabs(["Tabel dan statistik", "Tambah / hapus data", "Kasus ekstrem dan konflik"])

    with tab1:
        if df.empty:
            st.info("Dataset kosong.")
        else:
            hasil = evaluasi_dataset(df)
            valid = hasil[hasil["Kategori"] != "Error"]
            k = st.columns(4)
            with k[0]:
                kartu("Jumlah data", str(len(hasil)))
            with k[1]:
                kartu("Rata-rata skor", f"{valid['Skor Kelayakan'].mean():.2f}" if len(valid) else "-")
            with k[2]:
                kartu("Skor minimum", f"{valid['Skor Kelayakan'].min():.2f}" if len(valid) else "-")
            with k[3]:
                kartu("Skor maksimum", f"{valid['Skor Kelayakan'].max():.2f}" if len(valid) else "-")
            st.dataframe(hasil.drop(columns=["Catatan Error"]).round(4), hide_index=True, height=420)
            if (hasil["Kategori"] == "Error").any():
                st.error("Beberapa data gagal diproses:")
                st.dataframe(hasil[hasil["Kategori"] == "Error"][["Kode", "Nama", "Catatan Error"]], hide_index=True)
            st.download_button(
                "Unduh hasil (CSV)",
                hasil.to_csv(index=False).encode("utf-8"),
                file_name="hasil_kelayakan_beasiswa.csv",
                mime="text/csv",
            )

            st.subheader("Distribusi hasil")
            ringkas = (
                valid.groupby("Kategori").agg(Jumlah=("Kode", "count"), Rata2_Skor=("Skor Kelayakan", "mean"))
                .reindex(viz.URUTAN_KATEGORI).fillna(0).reset_index()
            )
            ringkas["Persentase"] = (ringkas["Jumlah"] / max(1, len(valid)) * 100).round(1)
            st.dataframe(ringkas.round(2), hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                tampil_fig(viz.plot_category_distribution(valid))
            with c2:
                tampil_fig(viz.plot_score_histogram(valid))
            st.subheader("Distribusi variabel input")
            tampil_fig(viz.plot_input_distributions(df))
            pil = st.selectbox("Skor terhadap variabel", [v.label for v in INPUT_VARIABLES.values()])
            key = next(k_ for k_, v in INPUT_VARIABLES.items() if v.label == pil)
            tampil_fig(viz.plot_scatter_dataset(valid, key))

    with tab2:
        st.subheader("Tambah satu data")
        with st.form("form_tambah", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                kode = st.text_input("Kode (kosong = otomatis)")
                nama = st.text_input("Nama mahasiswa dummy")
                ipk = st.number_input("IPK", 0.0, 4.0, 3.0, 0.01, format="%.2f")
                jenis = st.selectbox("Jenis kasus", dt.JENIS_KASUS)
            with c2:
                peng = st.number_input("Penghasilan orang tua (Rp)", 0, 15_000_000, 3_000_000, 100_000)
                tang = st.number_input("Tanggungan (orang)", 0, 8, 3, 1)
                pres = st.number_input("Prestasi (0-100)", 0.0, 100.0, 50.0, 0.5)
                ket = st.text_input("Keterangan")
            tambah = st.form_submit_button("Tambah data")
        if tambah:
            galat = dt.validasi_record(nama, ipk, peng, tang, pres)
            kode_final = kode.strip() or dt.kode_berikutnya(df)
            if kode_final in set(df["Kode"].astype(str)):
                galat.append(f"Kode '{kode_final}' sudah digunakan.")
            if galat:
                for g in galat:
                    st.error(g)
            else:
                baru = pd.DataFrame([{
                    "Kode": kode_final, "Nama": nama.strip(), "IPK": float(ipk),
                    "Penghasilan": float(peng), "Tanggungan": int(tang), "Prestasi": float(pres),
                    "Jenis Kasus": jenis, "Keterangan": ket.strip(),
                }])
                st.session_state["df_mahasiswa"] = pd.concat([df, baru], ignore_index=True)
                st.success(f"Data {kode_final} ditambahkan.")
                st.rerun()

        st.subheader("Tambah data acak (seed tetap)")
        jumlah = st.number_input("Jumlah data acak", 1, 50, 5, 1)
        if st.button("Tambah data acak"):
            baru = dt.buat_data_acak(int(jumlah), df)
            st.session_state["df_mahasiswa"] = pd.concat([df, baru], ignore_index=True)
            st.rerun()

        st.subheader("Hapus data")
        if df.empty:
            st.info("Tidak ada data untuk dihapus.")
        else:
            label = {r["Kode"]: f"{r['Kode']} - {r['Nama']}" for _, r in df.iterrows()}
            hapus = st.multiselect("Pilih data yang dihapus", list(label), format_func=lambda k_: label[k_])
            if st.button("Hapus data terpilih") and hapus:
                st.session_state["df_mahasiswa"] = df[~df["Kode"].isin(hapus)].reset_index(drop=True)
                st.rerun()
        if st.button("Reset ke dataset awal (CSV)"):
            st.session_state["df_mahasiswa"] = dt.buat_dataset_awal()
            st.rerun()

    with tab3:
        st.write(
            "Dataset sengaja memuat nilai ekstrem, outlier, titik batas fungsi keanggotaan, dan kriteria yang "
            "saling bertentangan. Data ini **tidak dihapus atau dinormalkan** agar sistem diuji pada kondisi sulit."
        )
        if df.empty:
            st.info("Dataset kosong.")
            return
        hasil = evaluasi_dataset(df)
        khusus = hasil[hasil["Jenis Kasus"] != "Normal"]
        st.dataframe(
            khusus[["Kode", "Nama", "Jenis Kasus", "IPK", "Penghasilan", "Tanggungan", "Prestasi",
                    "Skor Kelayakan", "Kategori", "Keterangan"]].round(4),
            hide_index=True, height=420,
        )
        st.subheader("Telaah satu kasus")
        label = {r["Kode"]: f"{r['Kode']} - {r['Nama']} ({r['Jenis Kasus']})" for _, r in khusus.iterrows()}
        if not label:
            st.info("Tidak ada kasus khusus pada dataset saat ini.")
            return
        kode = st.selectbox("Pilih mahasiswa", list(label), format_func=lambda k_: label[k_])
        baris = hasil[hasil["Kode"] == kode].iloc[0]
        st.write(f"**Keterangan:** {baris['Keterangan']}")
        masukan = {k_: baris[kol] for k_, kol in INPUT_COLUMNS.items()}
        try:
            result = fe.infer(masukan, RULES)
        except (InputValidationError, FuzzyInferenceError) as err:
            st.error(str(err))
            return
        for t in jelaskan_kasus(result):
            st.write("• " + t)
        tampil_fig(viz.plot_aggregation(result))


# Halaman 7 - Pengujian Sistem
def halaman_pengujian() -> None:
    st.title("Pengujian Sistem")
    st.caption(
        "Setiap skenario dijalankan melalui fungsi infer() yang sebenarnya. Harapan kategori ditetapkan dari "
        "logika bobot basis aturan, dan harapan μ dihitung dari rumus fungsi keanggotaan; "
        "bukan diambil dari hasil program."
    )
    tabel = val.jalankan_skenario_pengujian(RULES)
    lulus = int((tabel["Status"] == "LULUS").sum())
    k = st.columns(3)
    with k[0]:
        kartu("Jumlah skenario", str(len(tabel)))
    with k[1]:
        kartu("Lulus", str(lulus), "#2E8B6E")
    with k[2]:
        kartu("Gagal", str(len(tabel) - lulus), "#B5483C" if len(tabel) - lulus else None)

    st.markdown(
        "**Kriteria LULUS:** inferensi berhasil tanpa error · semua μ ∈ [0,1] · tiap variabel punya μ > 0 · "
        "≥ 1 aturan aktif · skor ∈ [0,100] · kategori sesuai harapan · μ sesuai nilai teoretis (bila ditentukan)."
    )
    kelompok = st.multiselect("Filter kelompok", sorted(tabel["Kelompok"].unique()),
                            default=sorted(tabel["Kelompok"].unique()))
    tampil = tabel[tabel["Kelompok"].isin(kelompok)]
    st.dataframe(tampil.drop(columns=["Keterangan Gagal"]).round(4), hide_index=True, height=560)
    if (tabel["Status"] == "GAGAL").any():
        st.error("Skenario yang gagal:")
        st.dataframe(tabel[tabel["Status"] == "GAGAL"][["No", "Skenario", "Keterangan Gagal"]], hide_index=True)

    st.subheader("Uji integritas model")
    integritas = val.cek_integritas(RULES)
    lulus_i = int((integritas["Status"] == "LULUS").sum())
    st.write(f"{lulus_i} dari {len(integritas)} pemeriksaan lulus.")
    st.dataframe(integritas, hide_index=True, height=600)


# Halaman 8 - Validasi Perhitungan Manual
def rumus_mu(x: float, himpunan) -> str:
    a, b, c, d = himpunan.trapezoid_params()
    f = format_angka
    if x < a or x > d:
        return "0 (di luar rentang)"
    if b <= x <= c:
        return "1 (daerah plateau/puncak)"
    if x < b:
        return f"({f(x)} − {f(a)}) / ({f(b)} − {f(a)})"
    return f"({f(d)} − {f(x)}) / ({f(d)} − {f(c)})"


def halaman_validasi() -> None:
    st.title("Validasi Perhitungan Manual")
    st.markdown(
        f"""
        Perhitungan manual ditulis terpisah di `validation.py` dengan **Python murni (tanpa NumPy)**: percabangan
        `if/else` untuk fungsi keanggotaan, `min()` untuk α, serta *loop* untuk implikasi, agregasi, dan centroid.
        Program utama memakai operasi vektor NumPy, sehingga selisih yang muncul benar-benar menguji implementasi.

        **Toleransi numerik:**
        - μ dan α-predikat: {val.TOL_MU:g} (selisih hanya galat pembulatan floating point).
        - Centroid pada grid yang sama (0,1): {val.TOL_CENTROID_DISKRIT:g}.
        - Centroid vs pembanding grid 0,01 atau rumus analitik: {val.TOL_CENTROID_KONTINU:g} (galat diskretisasi grid 0,1;
        centroid kontinu adalah batas dari centroid diskrit ketika interval grid → 0).
        """
    )
    ringkasan = []
    rincian = []
    for kasus in val.KASUS_VALIDASI:
        hasil = val.jalankan_validasi_manual(kasus, RULES)
        rincian.append((kasus, hasil))
        ringkasan.append({
            "Kasus": kasus["nama"],
            "Aturan Aktif": len(hasil["manual"]["aktif"]),
            "Z* Manual (grid 0,1)": hasil["manual"]["centroid"],
            "Z* Program": hasil["program"].centroid,
            "Selisih": abs(hasil["manual"]["centroid"] - hasil["program"].centroid),
            "Status": "VALID" if hasil["valid"] else "TIDAK VALID",
        })
    st.subheader("Ringkasan")
    st.dataframe(pd.DataFrame(ringkasan).round(8), hide_index=True)

    for kasus, hasil in rincian:
        with st.expander(f"{kasus['nama']}  —  {'VALID' if hasil['valid'] else 'TIDAK VALID'}"):
            st.write(kasus["deskripsi"])
            m, p = hasil["manual"], hasil["program"]
            x = kasus["inputs"]

            st.markdown("**Langkah 1 - Fuzzifikasi (substitusi rumus)**")
            baris = []
            for key, var in INPUT_VARIABLES.items():
                for s in var.sets:
                    baris.append({
                        "Variabel": var.label, "Nilai": format_angka(x[key]), "Himpunan": s.name,
                        "Parameter": "(" + ", ".join(format_angka(q) for q in s.params) + ")",
                        "Rumus": rumus_mu(x[key], s),
                        "μ Manual": m["fuzzifikasi"][key][s.name],
                        "μ Program": p.fuzzified[key][s.name],
                    })
            st.dataframe(pd.DataFrame(baris).round(6), hide_index=True)

            st.markdown("**Langkah 2 - Aturan aktif dan α-predikat = MIN**")
            baris = []
            for a in m["aktif"]:
                baris.append({
                    "Aturan": f"R{a['id']}",
                    "μ (IPK, Penghasilan, Tanggungan, Prestasi)": ", ".join(f"{v:.4f}" for v in a["mus"]),
                    "α = MIN": a["alpha"],
                    "Konsekuen": a["konsekuen"],
                })
            st.dataframe(pd.DataFrame(baris).round(6), hide_index=True)

            st.markdown("**Langkah 3 - Implikasi MIN**")
            st.write("Untuk setiap aturan aktif: μ_terpotong(z) = MIN(α, μ_konsekuen(z)) untuk seluruh z pada grid.")
            baris = []
            for a in m["aktif"]:
                baris.append({"Aturan": f"R{a['id']}", "Himpunan dipotong": a["konsekuen"],
                            "Tinggi potongan (α)": a["alpha"]})
            st.dataframe(pd.DataFrame(baris).round(6), hide_index=True)

            st.markdown("**Langkah 4 - Agregasi MAX (cuplikan nilai pada grid)**")
            idx = list(range(0, len(m["agregasi"]), 100))
            cuplik = pd.DataFrame({
                "z": [round(i * m["step"], 1) for i in idx],
                "μ agregasi manual": [m["agregasi"][i] for i in idx],
                "μ agregasi program": [float(p.aggregated[i]) for i in idx],
            })
            cuplik["Selisih"] = (cuplik["μ agregasi manual"] - cuplik["μ agregasi program"]).abs()
            st.dataframe(cuplik.round(6), hide_index=True)

            st.markdown("**Langkah 5 - Centroid**")
            st.write(
                f"Σ z·μ = {m['pembilang']:.6f};  Σ μ = {m['penyebut']:.6f};  "
                f"Z* = {m['pembilang']:.6f} / {m['penyebut']:.6f} = **{m['centroid']:.6f}**"
            )
            if hasil["analitik"] is not None:
                st.write(
                    "Karena hanya satu aturan aktif, centroid kontinu dapat dihitung analitik dengan memecah daerah "
                    "terpotong menjadi segitiga kiri + persegi panjang + segitiga kanan: "
                    f"Z*(analitik) = **{hasil['analitik']:.6f}**."
                )

            st.markdown("**Perbandingan hasil manual dan hasil program**")
            st.dataframe(hasil["tabel"], hide_index=True, height=420)


# Halaman 9 - Analisis Sensitivitas
def interpretasi_sensitivitas(tabel: pd.DataFrame, var) -> list[str]:
    x = tabel["Nilai Input"].to_numpy()
    y = tabel["Skor Kelayakan"].to_numpy()
    if len(x) < 3:
        return ["Jumlah titik terlalu sedikit untuk dianalisis."]
    teks = []
    delta = y[-1] - y[0]
    arah = "naik" if delta > 1 else "turun" if delta < -1 else "relatif tidak berubah"
    teks.append(f"Skor {arah} dari {y[0]:.2f} menjadi {y[-1]:.2f} (selisih {delta:+.2f}); "
                f"rentang skor {y.min():.2f}-{y.max():.2f}.")
    dy = np.diff(y)
    if np.all(dy >= -1e-9):
        teks.append("Pola monoton tidak turun: kenaikan input tidak pernah menurunkan skor.")
    elif np.all(dy <= 1e-9):
        teks.append("Pola monoton tidak naik: kenaikan input tidak pernah menaikkan skor.")
    else:
        teks.append("Pola tidak monoton: skor naik pada sebagian rentang dan turun pada rentang lain.")
    lereng = np.abs(dy / np.diff(x))
    if lereng.max() > 0:
        i = int(np.argmax(lereng))
        rata = lereng.mean()
        teks.append(f"Perubahan paling curam terjadi pada interval {format_angka(x[i])} - {format_angka(x[i + 1])}.")
        if lereng.max() > 1.5 * rata:
            teks.append("Kemiringan sangat bervariasi (maks > 1,5 × rata-rata) → hubungan input-skor TIDAK linear.")
    datar = float(np.mean(np.abs(dy) < 0.05))
    if datar >= 0.2:
        teks.append(f"Sekitar {datar * 100:.0f}% interval hampir datar (|Δskor| < 0,05): terdapat daerah jenuh (plateau).")
    kat = tabel["Kategori"].tolist()
    perubahan = [f"{kat[i]} → {kat[i + 1]} di sekitar {format_angka(x[i + 1])}"
                for i in range(len(kat) - 1) if kat[i] != kat[i + 1]]
    teks.append("Perpindahan kategori: " + ("; ".join(perubahan) if perubahan else "tidak ada (kategori tetap)") + ".")
    return teks


def halaman_sensitivitas() -> None:
    st.title("Analisis Sensitivitas")
    st.caption("Satu variabel diubah bertahap, variabel lain tetap. Setiap titik dihitung dengan Mamdani sebenarnya.")

    pil = st.selectbox("Variabel yang diuji", [v.label for v in INPUT_VARIABLES.values()])
    key = next(k_ for k_, v in INPUT_VARIABLES.items() if v.label == pil)
    var = INPUT_VARIABLES[key]
    lo, hi = var.universe
    awal = {"ipk": (2.0, 4.0), "penghasilan": (0.0, 15_000_000.0), "tanggungan": (0.0, 8.0),
            "prestasi": (0.0, 100.0)}[key]
    langkah = {"ipk": 0.05, "penghasilan": 250_000.0, "tanggungan": 0.25, "prestasi": 1.0}[key]
    rentang = st.slider("Rentang nilai yang diuji", float(lo), float(hi), (float(awal[0]), float(awal[1])),
                        float(langkah))
    n = st.slider("Jumlah titik pengujian", 11, 81, 41, 2)
    if rentang[1] <= rentang[0]:
        st.error("Batas atas harus lebih besar daripada batas bawah.")
        return

    profil = dict(dt.PROFIL_SENSITIVITAS)
    with st.expander("Profil kustom (opsional)"):
        c = st.columns(4)
        with c[0]:
            kc_ipk = st.number_input("IPK tetap", 0.0, 4.0, 3.0, 0.01, key="sens_ipk")
        with c[1]:
            kc_peng = st.number_input("Penghasilan tetap (Rp)", 0, 15_000_000, 4_000_000, 100_000, key="sens_peng")
        with c[2]:
            kc_tang = st.number_input("Tanggungan tetap", 0, 8, 3, 1, key="sens_tang")
        with c[3]:
            kc_pres = st.number_input("Prestasi tetap", 0.0, 100.0, 70.0, 0.5, key="sens_pres")
        profil["F. Kustom"] = {"ipk": kc_ipk, "penghasilan": kc_peng, "tanggungan": kc_tang, "prestasi": kc_pres}
    terpilih = st.multiselect("Skenario kondisi tetap", list(profil), default=list(profil)[:3])
    if not terpilih:
        st.info("Pilih minimal satu skenario kondisi tetap.")
        return

    nilai = np.linspace(rentang[0], rentang[1], n)
    hasil = {}
    try:
        for nama in terpilih:
            hasil[nama] = fe.sensitivity_analysis(profil[nama], key, nilai, RULES)
    except (InputValidationError, FuzzyInferenceError) as err:
        st.error(f"Analisis gagal: {err}")
        return

    tampil_fig(viz.plot_sensitivity(hasil, var))

    st.subheader("Kondisi tetap setiap skenario")
    tetap = []
    for nama in terpilih:
        baris = {"Skenario": nama}
        for k_, v in INPUT_VARIABLES.items():
            baris[v.label] = "(diuji)" if k_ == key else format_angka(profil[nama][k_])
        tetap.append(baris)
    st.dataframe(pd.DataFrame(tetap), hide_index=True)

    st.subheader("Tabel hasil")
    gabung = pd.DataFrame({"Nilai Input": nilai})
    for nama, tabel in hasil.items():
        gabung[f"Skor - {nama}"] = tabel["Skor Kelayakan"].to_numpy()
        gabung[f"Kategori - {nama}"] = tabel["Kategori"].to_numpy()
    st.dataframe(gabung.round(4), hide_index=True, height=420)

    st.subheader("Interpretasi pola perubahan")
    for nama, tabel in hasil.items():
        with st.expander(nama, expanded=True):
            for t in interpretasi_sensitivitas(tabel, var):
                st.write("• " + t)

    st.subheader("Mengapa perubahan tidak linear?")
    st.markdown(
        """
        - Fungsi keanggotaan bersifat **linear per bagian**: kemiringan berubah di titik a, b, c, d sehingga pengaruh input
        terhadap μ berubah-ubah.
        - Pada daerah plateau (μ = 1 atau μ = 0) perubahan input **tidak mengubah** derajat keanggotaan → skor datar.
        - Pergantian aturan aktif: ketika input melewati titik batas, aturan baru aktif dan aturan lama padam.
        - **Centroid** adalah rata-rata terbobot luas daerah, sehingga bergerak halus tetapi tidak proporsional
        terhadap α, dan **jenuh** di dekat 13 (plateau Tidak Layak) dan 82 (plateau Sangat Layak).
        - Veto akademik (IPK Rendah) membatasi konsekuen sehingga kenaikan variabel lain tidak menaikkan skor melewati batas tertentu.
        """
    )


# Halaman 10 - Analisis dan Keterbatasan
def halaman_analisis() -> None:
    st.title("Analisis dan Keterbatasan Sistem")
    st.markdown(f"<div class='catatan'>{DISCLAIMER}</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        ### 1. Interpretasi hasil
        Skor Z* (0-100) adalah hasil centroid dari gabungan seluruh aturan aktif. Kategori ditentukan dari himpunan output
        dengan derajat keanggotaan terbesar pada Z* (batas pada Z* = 25, 45, dan 65). Skor mendekati batas kategori
        menandakan kasus **marjinal** yang perlu ditinjau manual. Skor tidak pernah tepat 0 atau 100 karena centroid
        merupakan rata-rata; skor maksimum teoretis sekitar 82 dan minimum sekitar 13 (centroid plateau output).

        ### 2. Pengaruh parameter fungsi keanggotaan
        Menggeser titik a, b, c, d mengubah kapan suatu nilai crisp dianggap "rendah/sedang/tinggi". Overlap yang lebar membuat
        skor berubah halus tetapi mengaburkan perbedaan kategori; overlap yang sempit membuat skor lebih tajam namun peka
        terhadap perubahan kecil. Seluruh variabel input memakai partisi Ruspini (Σμ = 1) sehingga tidak ada celah.

        ### 3. Pengaruh kualitas basis aturan
        Kualitas keputusan sangat bergantung pada konsekuen 81 aturan. Pada sistem ini konsekuen ditentukan dari bobot
        (IPK {BOBOT['ipk']}, Penghasilan {BOBOT['penghasilan']}, Prestasi {BOBOT['prestasi']}, Tanggungan {BOBOT['tanggungan']}),
        ambang skor, dan veto akademik IPK Rendah. Mengubah bobot atau ambang akan mengubah hasil banyak mahasiswa tanpa
        mengubah satu pun fungsi keanggotaan. Veto membuat seluruh aturan ber-IPK Rendah berkonsekuen paling tinggi
        *{BATAS_KONSEKUEN_IPK_RENDAH}*, sehingga beberapa kasus berbeda menghasilkan skor yang sama persis.

        ### 4. Subjektivitas penentuan fungsi keanggotaan
        Parameter fungsi keanggotaan, bobot, ambang, dan veto adalah **asumsi perancang**, tidak diturunkan dari data historis
        atau ketentuan resmi. Pakar atau panitia yang berbeda dapat menetapkan nilai yang berbeda dan memperoleh ranking
        yang berbeda. Validasi dengan pakar atau data keputusan nyata diperlukan sebelum dipakai di dunia nyata.

        ### 5. Keterbatasan data simulasi
        Dataset adalah data dummy yang dirancang manual untuk menguji kondisi ekstrem dan konflik; distribusinya tidak
        mewakili populasi mahasiswa sebenarnya, sehingga statistik pada Dashboard (rata-rata, jumlah per kategori) tidak
        boleh digeneralisasi. Tidak ada variabel lain yang realistis dipertimbangkan (mis. kondisi khusus, asal daerah,
        jumlah kuota beasiswa).

        ### 6. Mengapa hasil bukan keputusan mutlak
        Sistem hanya mengurutkan prioritas dari empat kriteria. Keputusan nyata mempertimbangkan kuota, verifikasi dokumen,
        wawancara, kebijakan lembaga, dan pertimbangan etis. Sistem juga tidak dapat memverifikasi kebenaran data
        yang diinput dan memperlakukan semua kombinasi secara deterministik. Oleh karena itu, keluaran harus dipandang
        sebagai **rekomendasi pendukung keputusan** yang tetap memerlukan penilaian manusia.
        """
    )


# Navigasi
HALAMAN = {
    "Dashboard": halaman_dashboard,
    "Simulasi Individu": halaman_simulasi,
    "Fungsi Keanggotaan": halaman_mf,
    "Basis Aturan": halaman_aturan,
    "Detail Proses Perhitungan": halaman_detail,
    "Dataset Mahasiswa": halaman_dataset,
    "Pengujian Sistem": halaman_pengujian,
    "Validasi Perhitungan Manual": halaman_validasi,
    "Analisis Sensitivitas": halaman_sensitivitas,
    "Analisis dan Keterbatasan": halaman_analisis,
}

def main() -> None:
    init_state()
    st.sidebar.title("SPK Beasiswa")
    st.sidebar.caption("Fuzzy Mamdani")
    pilihan = st.sidebar.radio("Navigasi", list(HALAMAN))
    st.sidebar.divider()
    st.sidebar.caption(
        f"Grid output: {fe.GRID_MIN:g}-{fe.GRID_MAX:g}, interval {fe.GRID_STEP:g}\n\n"
        "AND = MIN · Implikasi = MIN · Agregasi = MAX · Defuzzifikasi = Centroid"
    )
    HALAMAN[pilihan]()


main()