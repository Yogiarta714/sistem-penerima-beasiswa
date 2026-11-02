from __future__ import annotations

from typing import Dict, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from .fuzzy_engine import (
    GRID_STEP,
    INPUT_COLUMNS,
    INPUT_VARIABLES,
    OUTPUT_MU,
    OUTPUT_VARIABLE,
    Z_GRID,
    FuzzyVariable,
    InferenceResult,
    category_boundaries,
)

# Warna konsisten untuk kategori kelayakan
CATEGORY_COLORS: Dict[str, str] = {
    "Tidak Layak": "#B5483C",
    "Cukup Layak": "#D9972B",
    "Layak": "#3F7FBF",
    "Sangat Layak": "#2E8B6E",
    "Error": "#808080",
}
SET_COLORS = ["#4C78A8", "#E0A030", "#54A24B"]  # himpunan input: rendah/sedang/tinggi
URUTAN_KATEGORI = OUTPUT_VARIABLE.set_names

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "figure.dpi": 110,
    }
)


def _formatter_sumbu(var: FuzzyVariable):
    """Penghasilan ditampilkan dalam juta rupiah agar sumbu mudah dibaca."""
    if var.key == "penghasilan":
        return FuncFormatter(lambda v, _: f"{v / 1e6:g}"), "Penghasilan orang tua (Rp juta / bulan)"
    return None, f"{var.label} ({var.unit})"


def _warna_himpunan(var: FuzzyVariable, indeks: int, nama: str) -> str:
    if var.key == "kelayakan":
        return CATEGORY_COLORS[nama]
    return SET_COLORS[indeks % len(SET_COLORS)]


def _gambar_variabel(ax, var: FuzzyVariable, crisp: Optional[float] = None) -> None:
    lo, hi = var.universe
    x = np.linspace(lo, hi, 1001)
    for i, s in enumerate(var.sets):
        warna = _warna_himpunan(var, i, s.name)
        y = s.membership(x)
        ax.plot(x, y, color=warna, lw=2, label=s.name)
        ax.fill_between(x, y, color=warna, alpha=0.10)
    if crisp is not None:
        ax.axvline(crisp, color="#444444", ls="--", lw=1.2)
        for i, s in enumerate(var.sets):
            mu = float(s.membership(crisp))
            if mu > 0:
                ax.scatter([crisp], [mu], color=_warna_himpunan(var, i, s.name), zorder=5, s=36)
                ax.annotate(f"{mu:.2f}", (crisp, mu), textcoords="offset points",
                            xytext=(6, 4), fontsize=8)
    fmt, label_x = _formatter_sumbu(var)
    if fmt is not None:
        ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel(label_x)
    ax.set_ylabel("Derajat keanggotaan μ")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.02, 1.12)
    ax.set_title(f"Fungsi keanggotaan – {var.label}")
    ax.legend(loc="upper center", ncol=len(var.sets), frameon=False, bbox_to_anchor=(0.5, -0.22))


def plot_variable(var: FuzzyVariable, crisp: Optional[float] = None):
    """Grafik fungsi keanggotaan satu variabel (opsional dengan garis nilai crisp)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _gambar_variabel(ax, var, crisp)
    fig.tight_layout()
    return fig


def plot_fuzzification(result: InferenceResult):
    """Empat variabel input beserta titik hasil fuzzifikasi nilai crisp."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, var) in zip(axes.ravel(), INPUT_VARIABLES.items()):
        _gambar_variabel(ax, var, result.inputs[key])
    fig.tight_layout()
    return fig


def plot_output_sets():
    """Fungsi keanggotaan output awal (sebelum implikasi)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _gambar_variabel(ax, OUTPUT_VARIABLE)
    ax.set_title("Fungsi keanggotaan output awal – Kelayakan Beasiswa")
    fig.tight_layout()
    return fig


def plot_implication(result: InferenceResult):
    """Kurva implikasi MIN setiap aturan aktif"""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for s in OUTPUT_VARIABLE.sets:
        ax.plot(Z_GRID, OUTPUT_MU[s.name], color=CATEGORY_COLORS[s.name], ls="--", lw=1, alpha=0.45)
    dipakai = set()
    for act in result.active_activations:
        kurva = result.clipped[act.rule.id]
        warna = CATEGORY_COLORS[act.rule.consequent]
        dipakai.add(act.rule.consequent)
        ax.fill_between(Z_GRID, kurva, color=warna, alpha=0.16)
        ax.plot(Z_GRID, kurva, color=warna, lw=1.1, alpha=0.9)
    handles = [Patch(color=CATEGORY_COLORS[n], alpha=0.5, label=f"Konsekuen {n}")
            for n in URUTAN_KATEGORI if n in dipakai]
    handles.append(Line2D([0], [0], color="#777777", ls="--", lw=1, label="Himpunan output awal"))
    ax.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2))
    ax.set_xlabel("Skor kelayakan z (0-100)")
    ax.set_ylabel("μ setelah implikasi MIN")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.02, 1.1)
    ax.set_title(f"Kurva implikasi MIN - {len(result.active_activations)} aturan aktif")
    fig.tight_layout()
    return fig


def plot_aggregation(result: InferenceResult):
    """Hasil agregasi MAX dan garis vertikal centroid Z*."""
    warna = CATEGORY_COLORS.get(result.category, "#444444")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for s in OUTPUT_VARIABLE.sets:
        ax.plot(Z_GRID, OUTPUT_MU[s.name], color=CATEGORY_COLORS[s.name], ls="--", lw=1, alpha=0.35)
    ax.fill_between(Z_GRID, result.aggregated, color=warna, alpha=0.30, label="Agregasi (MAX)")
    ax.plot(Z_GRID, result.aggregated, color=warna, lw=2)
    ax.axvline(result.centroid, color="#222222", lw=1.8, label=f"Centroid Z* = {result.centroid:.2f}")
    ax.annotate(f"Z* = {result.centroid:.2f}\n{result.category}",
                (result.centroid, max(0.05, float(result.aggregated.max()) * 0.5)),
                textcoords="offset points", xytext=(8, 0), fontsize=9, fontweight="bold")
    ax.set_xlabel("Skor kelayakan z (0-100)")
    ax.set_ylabel("μ hasil agregasi")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.02, 1.1)
    ax.set_title("Agregasi MAX dan hasil defuzzifikasi centroid")
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    return fig


def plot_category_distribution(df: pd.DataFrame):
    """Jumlah mahasiswa per kategori kelayakan."""
    hitung = df["Kategori"].value_counts().reindex(URUTAN_KATEGORI, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    batang = ax.bar(hitung.index, hitung.values, color=[CATEGORY_COLORS[k] for k in hitung.index])
    for b, v in zip(batang, hitung.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, str(int(v)), ha="center", fontsize=9)
    ax.set_ylabel("Jumlah mahasiswa")
    ax.set_xlabel("Kategori kelayakan")
    ax.set_title("Distribusi kategori kelayakan")
    ax.set_ylim(0, max(1, hitung.max()) * 1.2)
    fig.tight_layout()
    return fig


def plot_score_histogram(df: pd.DataFrame):
    """Histogram skor kelayakan dengan garis batas kategori."""
    skor = df["Skor Kelayakan"].dropna()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(skor, bins=np.arange(0, 105, 5), color="#4C78A8", edgecolor="white")
    for z, _, _ in category_boundaries():
        ax.axvline(z, color="#777777", ls="--", lw=1)
    ax.set_xlabel("Skor kelayakan Z*")
    ax.set_ylabel("Jumlah mahasiswa")
    ax.set_title("Histogram skor kelayakan (garis putus = batas kategori)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    return fig


def plot_input_distributions(df: pd.DataFrame):
    """Histogram keempat variabel input pada dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (key, var) in zip(axes.ravel(), INPUT_VARIABLES.items()):
        kolom = INPUT_COLUMNS[key]
        lo, hi = var.universe
        nilai = df[kolom].astype(float)
        ax.hist(nilai, bins=np.linspace(lo, hi, 11), color="#4C78A8", edgecolor="white")
        fmt, label_x = _formatter_sumbu(var)
        if fmt is not None:
            ax.xaxis.set_major_formatter(fmt)
        ax.set_xlabel(label_x)
        ax.set_ylabel("Jumlah")
        ax.set_title(f"Distribusi {var.label}")
        ax.set_xlim(lo, hi)
    fig.tight_layout()
    return fig


def plot_scatter_dataset(df: pd.DataFrame, key: str):
    """Skor kelayakan terhadap salah satu variabel input, diwarnai per kategori."""
    var = INPUT_VARIABLES[key]
    kolom = INPUT_COLUMNS[key]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for kat in URUTAN_KATEGORI:
        sub = df[df["Kategori"] == kat]
        ax.scatter(sub[kolom], sub["Skor Kelayakan"], color=CATEGORY_COLORS[kat], label=kat, s=40, alpha=0.9)
    fmt, label_x = _formatter_sumbu(var)
    if fmt is not None:
        ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel(label_x)
    ax.set_ylabel("Skor kelayakan Z*")
    ax.set_ylim(0, 100)
    ax.set_title(f"Skor kelayakan terhadap {var.label}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_sensitivity(hasil: Mapping[str, pd.DataFrame], var: FuzzyVariable):
    """Grafik perubahan skor terhadap nilai input untuk beberapa profil tetap."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    batas = [0.0] + [z for z, _, _ in category_boundaries()] + [100.0]
    for i, nama in enumerate(URUTAN_KATEGORI):
        ax.axhspan(batas[i], batas[i + 1], color=CATEGORY_COLORS[nama], alpha=0.07)
    palet = ["#1F3A5F", "#C0504D", "#4F9D69", "#8E6BBF", "#E08A1E", "#2A9D8F"]
    for i, (nama, tabel) in enumerate(hasil.items()):
        ax.plot(tabel["Nilai Input"], tabel["Skor Kelayakan"], marker="o", ms=3,
                lw=1.8, color=palet[i % len(palet)], label=nama)
    fmt, label_x = _formatter_sumbu(var)
    if fmt is not None:
        ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel(label_x)
    ax.set_ylabel("Skor kelayakan Z*")
    ax.set_ylim(0, 100)
    ax.set_title(f"Analisis sensitivitas – pengaruh {var.label} (pita warna = kategori)")
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2), fontsize=8)
    fig.tight_layout()
    return fig