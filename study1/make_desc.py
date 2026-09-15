from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DEFAULT_INPUT = r"E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v2.csv"
DEFAULT_OUTPUT_DIR = r"E:\PythonProject\GenAI\data_process\reg_results_main"

CANONICAL_COLUMNS: Dict[str, List[str]] = {
    "Log_RCR": ["Log_RCR", "log_rcr", "Log_RCR_winsor", "log_rcr_winsor"],
    "Hit_Rate_10_year_cat": ["Hit_Rate_10_year_cat", "Hit_Rate_10", "Hit_Rate10", "hit_rate_10_year_cat"],
    "Any_Citation": ["Any_Citation", "any_citation"],
    "HSS_Continuous": ["HSS_Continuous", "hss_continuous"],
    "HSS_Diversity": ["HSS_Diversity", "hss_diversity"],
    "has_hss_concepts": ["has_hss_concepts", "Has_HSS_Concepts", "is_hss_label"],
    "Broad_Social_Topic": ["broad_social_topic", "has_broad_social_topic", "broad_hss_topic"],
    "Log_Team_Size": ["Log_Team_Size", "log_team_size", "Log(Team_Size)"],
    "Log_Prior_Knowledge_Stock": [
        "Log_Prior_Knowledge_Stock",
        "log_prior_knowledge_stock",
        "Log(Prior_Knowledge_Stock)",
    ],
    "HSS_Bin": ["HSS_Bin", "hss_bin"],
    "stage": ["stage", "Stage"],
}

DISPLAY_NAMES = {
    "Log_RCR": "Log\\_RCR",
    "Hit_Rate_10_year_cat": "Hit\\_Rate\\_10\\_year\\_cat",
    "Any_Citation": "Any\\_Citation",
    "HSS_Continuous": "HSS\\_Continuous",
    "HSS_Diversity": "HSS\\_Diversity",
    "has_hss_concepts": "has\\_hss\\_concepts",
    "Broad_Social_Topic": "Broad\\_Social\\_Topic",
    "Log_Team_Size": "Log(Team\\_Size)",
    "Log_Prior_Knowledge_Stock": "Log(Prior\\_Knowledge\\_Stock)",
}

PANEL_MAP = {
    "Log_RCR": "Panel A: Outcome Variables",
    "Hit_Rate_10_year_cat": "Panel A: Outcome Variables",
    "Any_Citation": "Panel A: Outcome Variables",
    "HSS_Continuous": "Panel B: Core Explanatory Variables",
    "HSS_Diversity": "Panel B: Core Explanatory Variables",
    "has_hss_concepts": "Panel B: Core Explanatory Variables",
    "Broad_Social_Topic": "Panel B: Core Explanatory Variables",
    "Log_Team_Size": "Panel C: Controls",
    "Log_Prior_Knowledge_Stock": "Panel C: Controls",
}

PREFERRED_ORDER = [
    "Log_RCR",
    "Hit_Rate_10_year_cat",
    "Any_Citation",
    "HSS_Continuous",
    "HSS_Diversity",
    "has_hss_concepts",
    "Broad_Social_Topic",
    "Log_Team_Size",
    "Log_Prior_Knowledge_Stock",
]


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for canon, candidates in CANONICAL_COLUMNS.items():
        col = find_column(df, candidates)
        if col is not None:
            mapping[canon] = col
    return mapping


def safe_float(v) -> float:
    if pd.isna(v):
        return math.nan
    return float(v)


def fmt_num(v: float, digits: int = 4) -> str:
    if pd.isna(v):
        return ""
    return f"{v:.{digits}f}"


def summarize_series(s: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(s, errors="coerce")
    return {
        "N": int(s.notna().sum()),
        "Mean": safe_float(s.mean()),
        "SD": safe_float(s.std()),
        "Min": safe_float(s.min()),
        "Max": safe_float(s.max()),
    }


def build_main_desc(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for canon in PREFERRED_ORDER:
        if canon not in mapping:
            continue
        stats = summarize_series(df[mapping[canon]])
        stats["variable"] = canon
        stats["display_name"] = DISPLAY_NAMES.get(canon, canon)
        stats["panel"] = PANEL_MAP.get(canon, "Other")
        rows.append(stats)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[["panel", "variable", "display_name", "N", "Mean", "SD", "Min", "Max"]]
    return out


def build_share_table(df: pd.DataFrame, col: str, order: Optional[List[str]] = None, label_name: str = "label") -> pd.DataFrame:
    s = df[col].fillna("<NA>").astype(str)
    counts = s.value_counts(dropna=False)
    if order:
        ordered_index = [x for x in order if x in counts.index] + [x for x in counts.index if x not in order]
        counts = counts.reindex(ordered_index)
    total = len(df)
    out = pd.DataFrame({
        label_name: counts.index,
        "count": counts.values,
        "share": counts.values / total,
    })
    return out


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def render_desc_table(main_desc: pd.DataFrame, bin_df: pd.DataFrame, stage_df: pd.DataFrame, total_n: int, cited_n: Optional[int]) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Descriptive Statistics --- Study 1}")
    lines.append(r"\label{tab:desc_study1}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Variable} & \textbf{N} & \textbf{Mean} & \textbf{SD} & \textbf{Min} & \textbf{Max} \\")
    lines.append(r"\midrule")

    if not main_desc.empty:
        current_panel = None
        for _, row in main_desc.iterrows():
            if row["panel"] != current_panel:
                current_panel = row["panel"]
                lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{latex_escape(current_panel)}}}}} \\")
            lines.append(
                f"{row['display_name']} & {int(row['N'])} & {fmt_num(row['Mean'])} & {fmt_num(row['SD'])} & {fmt_num(row['Min'])} & {fmt_num(row['Max'])} \\\\"

            )

        lines.append(r"\midrule")

    if not bin_df.empty:
        lines.append(r"\multicolumn{6}{l}{\textit{Panel D: HSS Composition (HSS\_Bin)}} \\")
        for _, row in bin_df.iterrows():
            label = latex_escape(str(row["label"]))
            share_pct = row["share"] * 100
            lines.append(
                f"\\hspace{{0.4cm}} {label} & {int(row['count'])} & \\multicolumn{{4}}{{l}}{{{share_pct:.1f}\\% of sample}} \\\\"

            )
        lines.append(r"\midrule")

    if not stage_df.empty:
        lines.append(r"\multicolumn{6}{l}{\textit{Panel E: Technology Stage Distribution}} \\")
        for _, row in stage_df.iterrows():
            label = latex_escape(str(row["label"]))
            share_pct = row["share"] * 100
            lines.append(
                f"\hspace{{0.4cm}} {label} & {int(row['count'])} & \multicolumn{{4}}{{l}}{{{share_pct:.1f}\\% of sample}} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{minipage}{\linewidth}")
    lines.append(r"\smallskip")
    note = f"\\footnotesize \\textit{{Notes:}} This table reports descriptive statistics for the Study 1 paper-level sample. The final analysis sample contains {total_n:,} papers."
    if cited_n is not None:
        note += f" Among them, {cited_n:,} papers received at least one citation during the observation window."
    note += " Continuous variables are reported with their mean, standard deviation, minimum, and maximum values; categorical distributions are reported as counts and sample shares."
    lines.append(note)
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create descriptive-statistics files for Study 1.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to genai_regression_enriched_v2.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    mapping = resolve_columns(df)

    if "Any_Citation" not in mapping:
        for c in ["citations", "Citations", "citation_count", "cited_by_count"]:
            if c in df.columns:
                df["Any_Citation"] = (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0).astype(int)
                mapping["Any_Citation"] = "Any_Citation"
                break

    main_desc = build_main_desc(df, mapping)

    bin_df = pd.DataFrame(columns=["label", "count", "share"])
    if "HSS_Bin" in mapping:
        bin_order = ["Pure STEM", "Low HSS", "Medium HSS", "High HSS", "HSS-dominant"]
        bin_df = build_share_table(df, mapping["HSS_Bin"], order=bin_order, label_name="label")

    stage_df = pd.DataFrame(columns=["label", "count", "share"])
    if "stage" in mapping:
        stage_order = ["Phase1", "Phase2", "Phase3", "Phase 1", "Phase 2", "Phase 3"]
        stage_df = build_share_table(df, mapping["stage"], order=stage_order, label_name="label")

    total_n = len(df)
    cited_n = None
    if "Any_Citation" in mapping:
        cited_n = int(pd.to_numeric(df[mapping["Any_Citation"]], errors="coerce").fillna(0).sum())

    main_desc.to_csv(output_dir / "desc_study1_main.csv", index=False)
    if not bin_df.empty:
        bin_df.to_csv(output_dir / "desc_study1_hss_bin.csv", index=False)
    if not stage_df.empty:
        stage_df.to_csv(output_dir / "desc_study1_stage.csv", index=False)

    tex = render_desc_table(main_desc, bin_df, stage_df, total_n, cited_n)
    (output_dir / "desc_study1_table.tex").write_text(tex, encoding="utf-8")

    print("Descriptive-statistics files saved to:", output_dir)
    print("Resolved columns:")
    for k, v in mapping.items():
        print(f"  {k:<28} -> {v}")
    print(f"Total rows: {total_n:,}")
    if cited_n is not None:
        print(f"Any_Citation = 1 rows: {cited_n:,} ({cited_n / total_n:.3%})")


if __name__ == "__main__":
    main()
