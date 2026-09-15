# -*- coding: utf-8 -*-
r"""
Create publication-ready figures from the core and supplement regression outputs.

Expected input directories
--------------------------
Core results:
    E:\PythonProject\GenAI\data_process\reg_results_core
Supplement results:
    E:\PythonProject\GenAI\data_process\reg_results_supplement

Expected output directory
-------------------------
    E:\PythonProject\GenAI\data_process\figures_main
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CORE_DIR = Path(r"E:\PythonProject\GenAI\data_process\reg_results_core")
SUPP_DIR = Path(r"E:\PythonProject\GenAI\data_process\reg_results_supplement")
FIG_DIR = Path(r"E:\PythonProject\GenAI\data_process\figures_main")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)



def load_coef(model_name: str, source: str = "core") -> pd.DataFrame:
    base_dir = CORE_DIR if source == "core" else SUPP_DIR
    path = base_dir / f"{model_name}_coef.csv"
    return read_csv_safe(path)



def get_term_row(df: pd.DataFrame, term: str) -> pd.Series:
    match = df[df["term"] == term]
    if match.empty:
        available = ", ".join(df["term"].astype(str).head(20).tolist())
        raise KeyError(f"Term not found: {term}. Sample available terms: {available}")
    return match.iloc[0]



def save_fig(fig, base_name: str, manifest_rows: list, description: str):
    png_path = FIG_DIR / f"{base_name}.png"
    pdf_path = FIG_DIR / f"{base_name}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    manifest_rows.append({
        "figure_name": base_name,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "description": description,
    })
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")



def make_threshold_plot_logrcr(manifest_rows: list):
    df = load_coef("M2_LogRCR_Threshold_StageFE", source="core")
    bins = ["Pure STEM", "Low HSS", "Medium HSS", "High HSS", "HSS-dominant"]
    term_map = {
        "Low HSS": "C(HSS_Bin, Treatment(reference='Pure STEM'))[T.Low HSS]",
        "Medium HSS": "C(HSS_Bin, Treatment(reference='Pure STEM'))[T.Medium HSS]",
        "High HSS": "C(HSS_Bin, Treatment(reference='Pure STEM'))[T.High HSS]",
        "HSS-dominant": "C(HSS_Bin, Treatment(reference='Pure STEM'))[T.HSS-dominant]",
    }

    rows = [{"bin": "Pure STEM", "coef": 0.0, "ci_low": np.nan, "ci_high": np.nan}]
    for b in bins[1:]:
        r = get_term_row(df, term_map[b])
        rows.append({"bin": b, "coef": r["coef"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(plot_df))
    ax.axhline(0, linewidth=1)
    ax.plot(x, plot_df["coef"], marker="o", linewidth=1.5)

    for i, row in plot_df.iloc[1:].iterrows():
        ax.vlines(i, row["ci_low"], row["ci_high"], linewidth=1)
        ax.plot([i, i], [row["ci_low"], row["ci_high"]], marker="_", markersize=8)
        ax.text(i, row["coef"], f"{row['coef']:.3f}", fontsize=9, ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["bin"], rotation=20)
    ax.set_ylabel("Coefficient on Log_RCR\n(vs Pure STEM)")
    ax.set_title("Threshold effects of HSS integration on impact")

    save_fig(
        fig,
        "fig1_threshold_logrcr",
        manifest_rows,
        "Coefficient plot for M2_LogRCR_Threshold_StageFE. Pure STEM is the omitted reference group.",
    )



def make_threshold_plot_top10(manifest_rows: list):
    df = load_coef("M6_Hit10YearCat_Threshold_LPM", source="core")
    bins = ["Pure STEM", "Low HSS", "Medium HSS", "High HSS", "HSS-dominant"]
    term_map = {
        "Low HSS": "C(HSS_Bin, Treatment(reference='Pure STEM'))[T.Low HSS]",
        "Medium HSS": "C(HSS_Bin, Treatment(reference='Pure STEM'))[T.Medium HSS]",
        "High HSS": "C(HSS_Bin, Treatment(reference='Pure STEM'))[T.High HSS]",
        "HSS-dominant": "C(HSS_Bin, Treatment(reference='Pure STEM'))[T.HSS-dominant]",
    }

    rows = [{"bin": "Pure STEM", "coef": 0.0, "ci_low": np.nan, "ci_high": np.nan}]
    for b in bins[1:]:
        r = get_term_row(df, term_map[b])
        rows.append({"bin": b, "coef": r["coef"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(plot_df))
    ax.axhline(0, linewidth=1)
    ax.plot(x, plot_df["coef"], marker="o", linewidth=1.5)

    for i, row in plot_df.iloc[1:].iterrows():
        ax.vlines(i, row["ci_low"], row["ci_high"], linewidth=1)
        ax.plot([i, i], [row["ci_low"], row["ci_high"]], marker="_", markersize=8)
        ax.text(i, row["coef"], f"{row['coef']:.3f}", fontsize=9, ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["bin"], rotation=20)
    ax.set_ylabel("Coefficient on Top-10% probability\n(vs Pure STEM)")
    ax.set_title("Threshold effects of HSS integration on breakthrough probability")

    save_fig(
        fig,
        "fig2_threshold_top10",
        manifest_rows,
        "Coefficient plot for M6_Hit10YearCat_Threshold_LPM. Pure STEM is the omitted reference group.",
    )



def make_stage_slopes_plot(manifest_rows: list):
    df = read_csv_safe(SUPP_DIR / "stage_slope_tests.csv")
    keep = ["Phase1_total_slope", "Phase2_total_slope", "Phase3_total_slope"]
    plot_df = df[df["label"].isin(keep)].copy()
    order = {"Phase1_total_slope": 0, "Phase2_total_slope": 1, "Phase3_total_slope": 2}
    plot_df["order"] = plot_df["label"].map(order)
    plot_df = plot_df.sort_values("order")
    labels = ["Phase 1", "Phase 2", "Phase 3"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(plot_df))
    ax.axhline(0, linewidth=1)
    ax.errorbar(
        x,
        plot_df["estimate"],
        yerr=[plot_df["estimate"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["estimate"]],
        fmt="o-",
        capsize=4,
    )

    for i, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(i, row.estimate, f"{row.estimate:.3f}", fontsize=9, ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total slope of HSS_Continuous on Log_RCR")
    ax.set_title("Marginal impact of HSS integration across GenAI stages")

    save_fig(
        fig,
        "fig3_stage_total_slopes",
        manifest_rows,
        "Total-slope plot based on stage_slope_tests.csv from the post-estimation interaction tests.",
    )



def make_topic_slopes_plot(manifest_rows: list):
    df = read_csv_safe(SUPP_DIR / "topic_slope_tests.csv")
    keep = ["No_HSS_concepts_total_slope", "Has_HSS_concepts_total_slope"]
    plot_df = df[df["label"].isin(keep)].copy()
    order = {"No_HSS_concepts_total_slope": 0, "Has_HSS_concepts_total_slope": 1}
    plot_df["order"] = plot_df["label"].map(order)
    plot_df = plot_df.sort_values("order")
    labels = ["No HSS concepts", "Has HSS concepts"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(plot_df))
    ax.axhline(0, linewidth=1)
    ax.errorbar(
        x,
        plot_df["estimate"],
        yerr=[plot_df["estimate"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["estimate"]],
        fmt="o-",
        capsize=4,
    )

    for i, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(i, row.estimate, f"{row.estimate:.3f}", fontsize=9, ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Total slope of HSS_Continuous on Log_RCR")
    ax.set_title("Marginal impact of HSS integration by topic type")

    save_fig(
        fig,
        "fig4_topic_total_slopes",
        manifest_rows,
        "Total-slope plot based on topic_slope_tests.csv from the post-estimation interaction tests.",
    )



def make_diffusion_plot(manifest_rows: list):
    model_specs = [
        ("D2_Citation_Field_Diversity_CitedOnly", "Citation field diversity"),
        ("D2_Citation_Entropy_CitedOnly", "Citation entropy"),
        ("D2_NonCS_Citation_Share_CitedOnly", "Non-CS citation share"),
        ("D2_HSS_Citing_Share_CitedOnly", "HSS citing share"),
    ]

    rows = []
    for model_name, label in model_specs:
        df = load_coef(model_name, source="core")
        r = get_term_row(df, "HSS_Continuous")
        rows.append({
            "outcome": label,
            "coef": r["coef"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
        })
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(plot_df))
    ax.axvline(0, linewidth=1)
    ax.errorbar(
        plot_df["coef"],
        y,
        xerr=[plot_df["coef"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["coef"]],
        fmt="o",
        capsize=4,
    )

    for i, row in plot_df.iterrows():
        ax.text(row["coef"], i, f" {row['coef']:.3f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["outcome"])
    ax.set_xlabel("Coefficient on HSS_Continuous (cited-only sample)")
    ax.set_title("Selective boundary-spanning effects of HSS integration")

    save_fig(
        fig,
        "fig5_diffusion_cited_only",
        manifest_rows,
        "Horizontal coefficient plot for cited-only diffusion outcomes using D2 models.",
    )



def make_outcome_compare_plot(manifest_rows: list):
    model_specs = [
        ("M1_LogRCR_Continuous_StageFE", "core", "Log_RCR"),
        ("M5_Hit10YearCat_Continuous_LPM", "core", "Top-10% probability"),
        ("S2_AnyCitation_Continuous_LPM", "supp", "Any citation"),
        ("S4_LogCitations_Continuous_StageFE", "supp", "log(1 + citations)"),
    ]

    rows = []
    for model_name, source, label in model_specs:
        df = load_coef(model_name, source=("core" if source == "core" else "supp"))
        r = get_term_row(df, "HSS_Continuous")
        rows.append({
            "outcome": label,
            "coef": r["coef"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
        })
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(plot_df))
    ax.axvline(0, linewidth=1)
    ax.errorbar(
        plot_df["coef"],
        y,
        xerr=[plot_df["coef"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["coef"]],
        fmt="o",
        capsize=4,
    )

    for i, row in plot_df.iterrows():
        ax.text(row["coef"], i, f" {row['coef']:.3f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["outcome"])
    ax.set_xlabel("Coefficient on HSS_Continuous")
    ax.set_title("Main and supplemental outcomes move in the same direction")

    save_fig(
        fig,
        "fig6_outcome_compare_continuous",
        manifest_rows,
        "Comparison of HSS_Continuous coefficients across core and supplemental outcomes.",
    )



def main():
    manifest_rows = []

    make_threshold_plot_logrcr(manifest_rows)
    make_threshold_plot_top10(manifest_rows)
    make_stage_slopes_plot(manifest_rows)
    make_topic_slopes_plot(manifest_rows)
    make_diffusion_plot(manifest_rows)
    make_outcome_compare_plot(manifest_rows)

    pd.DataFrame(manifest_rows).to_csv(FIG_DIR / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    print("\nAll figures generated.")
    print(f"Output directory: {FIG_DIR}")


if __name__ == "__main__":
    main()
