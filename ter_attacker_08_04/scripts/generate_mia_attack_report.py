
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from scipy.stats import ks_2samp  # type: ignore
except Exception:  # pragma: no cover
    ks_2samp = None


STYLE = r"""
  :root {
    --bg: #0d1117;
    --panel: #151b23;
    --panel-2: #1b2430;
    --line: #293243;
    --text: #e8ecf3;
    --muted: #98a2b3;
    --accent: #7aa2ff;
    --accent-soft: rgba(122, 162, 255, 0.12);
    --good: #2ecc71;
    --good-soft: rgba(46, 204, 113, 0.14);
    --warn: #f5b942;
    --warn-soft: rgba(245, 185, 66, 0.14);
    --danger: #ff7b7b;
    --danger-soft: rgba(255, 123, 123, 0.14);
    --radius: 18px;
    --shadow: 0 10px 30px rgba(0,0,0,.18);
    --font: Inter, system-ui, sans-serif;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    font-family: var(--font);
    color: var(--text);
    background:
      radial-gradient(circle at top right, rgba(122,162,255,0.10), transparent 24%),
      linear-gradient(180deg, #0b1016 0%, #0d1117 100%);
    line-height: 1.65;
  }
  header {
    padding: 52px 28px 36px;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(135deg, rgba(122,162,255,.08), rgba(122,162,255,0) 45%);
  }
  .wrap { max-width: 1220px; margin: 0 auto; }
  .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid rgba(122,162,255,.22);
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
  }
  h1 {
    margin: 18px 0 12px;
    font-size: clamp(2rem, 4vw, 3rem);
    line-height: 1.08;
    letter-spacing: -0.03em;
  }
  .lead { max-width: 920px; color: var(--muted); font-size: 1rem; }
  nav {
    position: sticky;
    top: 0;
    z-index: 10;
    backdrop-filter: blur(10px);
    background: rgba(13,17,23,.88);
    border-bottom: 1px solid var(--line);
  }
  nav .wrap { display: flex; gap: 18px; flex-wrap: wrap; padding: 14px 28px; }
  nav a { color: var(--muted); text-decoration: none; font-size: 13px; font-weight: 600; }
  nav a:hover { color: var(--accent); }
  main { padding: 28px; }
  section { margin: 0 0 22px; }
  .card {
    background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0));
    background-color: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 24px;
  }
  .section-title { margin: 0 0 16px; font-size: 1.55rem; letter-spacing: -0.02em; }
  .section-subtitle { margin: -4px 0 18px; color: var(--muted); font-size: .96rem; }
  .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
  .metric {
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px 16px 14px;
  }
  .metric .k {
    color: var(--muted);
    font-size: .76rem;
    text-transform: uppercase;
    letter-spacing: .06em;
  }
  .metric .v {
    margin-top: 6px;
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.03em;
  }
  .good { color: var(--good); }
  .warn { color: var(--warn); }
  .danger { color: var(--danger); }
  .accent { color: var(--accent); }
  .two { display: grid; gap: 16px; grid-template-columns: 1fr 1fr; }
  .callout {
    border-left: 4px solid var(--accent);
    background: var(--accent-soft);
    border-radius: 0 16px 16px 0;
    padding: 14px 16px;
    color: #dbe5ff;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: .9rem;
    overflow: hidden;
    border-radius: 16px;
    background: var(--panel-2);
    border: 1px solid var(--line);
  }
  th, td {
    padding: 11px 14px;
    border-bottom: 1px solid var(--line);
    text-align: left;
    vertical-align: top;
  }
  tr:last-child td { border-bottom: none; }
  th {
    color: var(--accent);
    font-size: .75rem;
    letter-spacing: .05em;
    text-transform: uppercase;
    background: rgba(122,162,255,0.06);
  }
  code {
    font-family: var(--mono);
    font-size: .82rem;
    color: var(--accent);
    background: rgba(122,162,255,0.10);
    padding: 2px 6px;
    border-radius: 8px;
  }
  .small { color: var(--muted); font-size: .88rem; }
  .chart-card {
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px;
  }
  .chart-wrap { position: relative; height: 340px; }
  .cm {
    display: grid;
    grid-template-columns: 110px 1fr 1fr;
    gap: 10px;
    align-items: stretch;
    font-size: .92rem;
  }
  .cm .label {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-weight: 600;
  }
  .cm .cell {
    border-radius: 16px;
    border: 1px solid var(--line);
    padding: 16px;
    min-height: 98px;
    background: var(--panel-2);
  }
  .cm .cell strong {
    display: block;
    font-size: 1.6rem;
    line-height: 1.1;
    margin-bottom: 6px;
  }
  .cell.tp { background: rgba(46,204,113,.10); }
  .cell.tn { background: rgba(122,162,255,.10); }
  .cell.fp { background: rgba(255,123,123,.10); }
  .cell.fn { background: rgba(245,185,66,.10); }
  ul.clean { margin: 8px 0 0 18px; padding: 0; }
  @media (max-width: 980px) { .two { grid-template-columns: 1fr; } }
  @media print {
    nav { display: none; }
    body { background: white; color: black; }
    .card, .metric, table, .chart-card { box-shadow: none; }
  }
"""

SUMMARY_KEYS = [
    "attack_id",
    "known_qids",
    "qid_filter_qids",
    "refine_qids",
    "target_id_col",
    "member_col",
    "n_targets",
    "n_members",
    "n_non_members",
    "seed",
    "max_compatible_fraction",
    "use_privjedai_fuzzy",
    "tp",
    "tn",
    "fp",
    "fn",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "member_recall",
    "member_false_negative_rate",
    "non_member_true_negative_rate",
    "non_member_false_positive_rate",
    "member_avg_stage1_equivalence_class_size",
    "non_member_avg_stage1_equivalence_class_size",
    "member_avg_compatible_candidate_count",
    "non_member_avg_compatible_candidate_count",
    "member_avg_equivalence_class_reduction",
    "non_member_avg_equivalence_class_reduction",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Génère un rapport HTML pour une MIA.")
    p.add_argument("--project-root", type=Path, default=Path.cwd(), help="Racine du projet")
    p.add_argument("--attack-dir", type=Path, default=None, help="Dossier de l'attaque MIA")
    p.add_argument("--summary-json", type=Path, default=None, help="Chemin direct vers summary.json")
    p.add_argument("--targets-csv", type=Path, default=None, help="Chemin direct vers targets.csv")
    p.add_argument("--metrics-json", type=Path, default=None, help="Chemin direct vers le JSON de métriques d'anonymisation")
    p.add_argument("--config-json", type=Path, default=None, help="Chemin direct vers le JSON de configuration d'anonymisation")
    p.add_argument("--output", type=Path, default=None, help="Fichier HTML de sortie")
    p.add_argument("--title", type=str, default=None, help="Titre personnalisé")
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{int(round(float(value))):,}".replace(",", " ")
    except Exception:
        return html.escape(str(value))


def fmt_float(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}".replace(".", ",")
    except Exception:
        return html.escape(str(value))


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{100.0 * float(value):.{digits}f}\xa0%".replace(".", ",")
    except Exception:
        return html.escape(str(value))


def fmt_list(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, list):
        return ", ".join(str(x) for x in value) if value else "—"
    return str(value)


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def find_latest_mia_dir(project_root: Path) -> Path:
    base = project_root / "outputs" / "attacks" / "mia"
    candidates = [p.parent for p in base.rglob("summary.json")]
    if not candidates:
        raise FileNotFoundError(f"Aucun summary.json trouvé sous {base}")
    return max(candidates, key=lambda p: (p / "summary.json").stat().st_mtime)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path, Path | None]:
    project_root = args.project_root.resolve()

    if args.summary_json:
        summary_json = args.summary_json.resolve()
        attack_dir = summary_json.parent
    else:
        attack_dir = args.attack_dir.resolve() if args.attack_dir else find_latest_mia_dir(project_root)
        summary_json = attack_dir / "summary.json"

    targets_csv = args.targets_csv.resolve() if args.targets_csv else attack_dir / "targets.csv"
    summary = read_json(summary_json)

    if args.metrics_json:
        metrics_json = args.metrics_json.resolve()
    else:
        attack_id = str(summary.get("attack_id", attack_dir.name))
        experiment_id = attack_id.split("__mia_")[0]
        candidate = project_root / "outputs" / "metrics" / f"{experiment_id}.json"
        metrics_json = candidate if candidate.exists() else None

    if args.config_json:
        config_json = args.config_json.resolve()
    else:
        config_json = None
        config_path_in_summary = summary.get("config_path")
        if config_path_in_summary:
            candidate = Path(str(config_path_in_summary))
            if candidate.exists():
                config_json = candidate
            else:
                basename_candidate = project_root / "outputs" / "configs" / candidate.name
                if basename_candidate.exists():
                    config_json = basename_candidate
        if config_json is None:
            attack_id = str(summary.get("attack_id", attack_dir.name))
            experiment_id = attack_id.split("__mia_")[0]
            fallback = project_root / "outputs" / "configs" / f"{experiment_id}.json"
            if fallback.exists():
                config_json = fallback

    return project_root, summary_json, metrics_json, targets_csv, config_json


def rel_or_abs(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path)


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    thead = "<thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr></thead>"
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"


def grouped_histogram(
    in_values: list[float],
    out_values: list[float],
    n_bins: int,
    value_min: float | None = None,
    value_max: float | None = None,
) -> tuple[list[str], list[int], list[int]]:
    values = [*in_values, *out_values]
    if not values:
        return [], [], []
    lo = min(values) if value_min is None else value_min
    hi = max(values) if value_max is None else value_max
    if math.isclose(lo, hi):
        return [f"{100.0 * lo:.1f}%"], [len(in_values)], [len(out_values)]

    width = (hi - lo) / n_bins
    labels: list[str] = []
    in_counts = [0 for _ in range(n_bins)]
    out_counts = [0 for _ in range(n_bins)]
    for i in range(n_bins):
        a = lo + i * width
        b = lo + (i + 1) * width
        labels.append(f"{100.0 * a:.1f}–{100.0 * b:.1f}")

    for v in in_values:
        idx = int((v - lo) / width)
        if idx == n_bins:
            idx -= 1
        in_counts[idx] += 1
    for v in out_values:
        idx = int((v - lo) / width)
        if idx == n_bins:
            idx -= 1
        out_counts[idx] += 1

    return labels, in_counts, out_counts


def compute_ks_pvalue(a: list[float], b: list[float]) -> float | None:
    if not a or not b or ks_2samp is None:
        return None
    try:
        return float(ks_2samp(a, b).pvalue)
    except Exception:
        return None


def compute_best_fraction_threshold(rows: list[dict[str, str]]) -> tuple[float | None, float | None]:
    candidates = []
    for row in rows:
        frac = to_float(row.get("compatible_candidate_fraction"))
        count = to_int(row.get("compatible_candidate_count")) or 0
        truth = to_int(row.get("ground_truth_member"))
        if frac is None or truth is None:
            continue
        candidates.append((frac, count, truth))
    if not candidates:
        return None, None

    thresholds = sorted({frac for frac, _, _ in candidates})
    thresholds = [min(thresholds) - 1e-12, *thresholds, max(thresholds) + 1e-12]

    best_acc = -1.0
    best_thr: float | None = None
    for thr in thresholds:
        correct = 0
        for frac, count, truth in candidates:
            pred = 1 if (count > 0 and frac <= thr) else 0
            if pred == truth:
                correct += 1
        acc = correct / len(candidates)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc


def build_report(
    project_root: Path,
    summary_json: Path,
    metrics_json: Path | None,
    config_json: Path | None,
    targets_csv: Path,
    output_path: Path,
    title: str | None,
) -> Path:
    summary = read_json(summary_json)
    metrics = read_json(metrics_json) if metrics_json and metrics_json.exists() else {}
    config = read_json(config_json) if config_json and config_json.exists() else {}
    rows = read_csv_rows(targets_csv) if targets_csv.exists() else []

    attack_id = str(summary.get("attack_id", summary_json.parent.name))
    report_title = title or f"Rapport MIA – {attack_id}"

    tp = to_int(summary.get("tp")) or 0
    tn = to_int(summary.get("tn")) or 0
    fp = to_int(summary.get("fp")) or 0
    fn = to_int(summary.get("fn")) or 0

    accuracy = to_float(summary.get("accuracy"))
    precision = to_float(summary.get("precision"))
    recall = to_float(summary.get("recall"))
    f1 = to_float(summary.get("f1"))
    tnr = to_float(summary.get("non_member_true_negative_rate"))
    max_compatible_fraction = to_float(summary.get("max_compatible_fraction"))

    n_targets = to_int(summary.get("n_targets")) or len(rows)
    n_members = to_int(summary.get("n_members")) or sum(1 for r in rows if to_int(r.get("ground_truth_member")) == 1)
    n_non_members = to_int(summary.get("n_non_members")) or max(n_targets - n_members, 0)

    member_frac = [
        frac for row in rows
        if to_int(row.get("ground_truth_member")) == 1
        for frac in [to_float(row.get("compatible_candidate_fraction"))]
        if frac is not None
    ]
    non_member_frac = [
        frac for row in rows
        if to_int(row.get("ground_truth_member")) == 0
        for frac in [to_float(row.get("compatible_candidate_fraction"))]
        if frac is not None
    ]
    member_count = [
        cnt for row in rows
        if to_int(row.get("ground_truth_member")) == 1
        for cnt in [to_float(row.get("compatible_candidate_count"))]
        if cnt is not None
    ]
    non_member_count = [
        cnt for row in rows
        if to_int(row.get("ground_truth_member")) == 0
        for cnt in [to_float(row.get("compatible_candidate_count"))]
        if cnt is not None
    ]

    ks_pvalue = compute_ks_pvalue(member_frac, non_member_frac)
    best_thr, best_acc = compute_best_fraction_threshold(rows)

    hist_labels, hist_in, hist_out = grouped_histogram(member_frac, non_member_frac, 18, 0.0, max([*member_frac, *non_member_frac], default=0.0))
    pred_counts = Counter(to_int(row.get("predicted_member")) for row in rows)
    predicted_in_count = pred_counts.get(1, 0)
    predicted_out_count = pred_counts.get(0, 0)

    synthesis_parts = []
    if accuracy is not None:
        synthesis_parts.append(f"L'accuracy globale de la MIA est de <strong>{fmt_pct(accuracy)}</strong>.")
    if precision is not None and recall is not None:
        synthesis_parts.append(
            f"La détection des membres obtient une précision de <strong>{fmt_pct(precision)}</strong> et un recall de <strong>{fmt_pct(recall)}</strong>."
        )
    if ks_pvalue is not None:
        synthesis_parts.append(
            f"Le test KS sur la fraction de candidats compatibles donne <strong>p = {fmt_float(ks_pvalue, 3)}</strong>."
        )
    if best_acc is not None and best_thr is not None:
        synthesis_parts.append(
            f"En n'utilisant que la variable <code>compatible_candidate_fraction</code>, la meilleure accuracy observée est <strong>{fmt_pct(best_acc)}</strong> avec un seuil autour de <strong>{fmt_pct(best_thr)}</strong>."
        )
    synthesis_html = " ".join(synthesis_parts) if synthesis_parts else "Aucune synthèse disponible."

    metrics_cards = [
        ("Cibles MIA", fmt_int(n_targets), ""),
        ("IN", fmt_int(n_members), ""),
        ("OUT", fmt_int(n_non_members), ""),
        ("Accuracy", fmt_pct(accuracy), "warn"),
        ("Precision", fmt_pct(precision), ""),
        ("Recall membres", fmt_pct(recall), ""),
        ("TNR non-membres", fmt_pct(tnr), ""),
        ("F1-score", fmt_pct(f1), "accent"),
    ]
    metrics_html = "".join(
        f"<div class='metric'><div class='k'>{escape(k)}</div><div class='v {cls}'>{v}</div></div>"
        for k, v, cls in metrics_cards
    )

    config_rows = [
        ["Attack ID", f"<code>{escape(attack_id)}</code>"],
        ["QID connus attaquant", escape(fmt_list(summary.get("known_qids")))],
        ["QID stage 1", escape(fmt_list(summary.get("qid_filter_qids")))],
        ["QID stage 2", escape(fmt_list(summary.get("refine_qids")))],
        ["Cibles", fmt_int(n_targets)],
        ["Membres / non-membres", f"{fmt_int(n_members)} / {fmt_int(n_non_members)}"],
        ["Seuil max compatible fraction", fmt_pct(max_compatible_fraction)],
        ["PrivJedAI fuzzy", escape(str(summary.get("use_privjedai_fuzzy", False)).lower())],
        ["Seed", fmt_int(summary.get("seed"))],
    ]

    if metrics or config:
        transformations = (metrics.get("transformations") if metrics else None) or {}
        config_rows.extend(
            [
                ["QI anonymisation", escape(fmt_list((config.get("quasi_identifiers") if config else None) or (metrics.get("quasi_identifiers") if metrics else None))) if ((config.get("quasi_identifiers") if config else None) or (metrics.get("quasi_identifiers") if metrics else None)) else "—"],
                ["k / l / t", f"<code>k={escape((config.get('k') if config else None) if (config.get('k') if config else None) is not None else (metrics.get('k') if metrics else None))}</code> · <code>l={escape((config.get('l') if config else None) if (config.get('l') if config else None) is not None else (metrics.get('l') if metrics else None))}</code> · <code>t={escape((config.get('t') if config else None) if (config.get('t') if config else None) is not None else (metrics.get('t') if metrics else None))}</code>"],
                ["Suppression limit", f"<code>{escape((config.get('suppression_limit') if config else None) if (config.get('suppression_limit') if config else None) is not None else (metrics.get('suppression_limit') if metrics else None))}</code>" if (((config.get("suppression_limit") if config else None) is not None) or ((metrics.get("suppression_limit") if metrics else None) is not None)) else "—"],
                ["Transformations", "<code>" + escape(json.dumps(transformations, ensure_ascii=False)) + "</code>" if transformations else "—"],
            ]
        )

    files_rows = [
        ["summary.json", f"<code>{escape(rel_or_abs(summary_json, project_root))}</code>"],
        ["targets.csv", f"<code>{escape(rel_or_abs(targets_csv, project_root))}</code>"],
    ]
    if metrics_json and metrics_json.exists():
        files_rows.append(["metrics.json", f"<code>{escape(rel_or_abs(metrics_json, project_root))}</code>"])
    if config_json and config_json.exists():
        files_rows.append(["config.json", f"<code>{escape(rel_or_abs(config_json, project_root))}</code>"])

    summary_table_rows = []
    for key in SUMMARY_KEYS:
        if key in summary:
            value = summary[key]
            if isinstance(value, list):
                value_str = fmt_list(value)
            elif isinstance(value, bool):
                value_str = "true" if value else "false"
            elif isinstance(value, float) and (
                key.endswith("_rate")
                or key in {"accuracy", "precision", "recall", "f1", "member_recall", "non_member_true_negative_rate", "non_member_false_positive_rate", "member_false_negative_rate", "max_compatible_fraction"}
            ):
                value_str = fmt_pct(value)
            elif isinstance(value, (int, float)):
                value_str = fmt_float(value) if isinstance(value, float) and not float(value).is_integer() else fmt_int(value)
            else:
                value_str = escape(value)
            summary_table_rows.append([f"<code>{escape(key)}</code>", value_str])

    op_counter = summary.get("operation_counter") or {}
    op_rows: list[list[str]] = []
    if isinstance(op_counter, dict):
        for k, v in op_counter.items():
            if isinstance(v, float) and not float(v).is_integer():
                val = fmt_float(v, 4)
            elif isinstance(v, (int, float)):
                val = fmt_int(v)
            else:
                val = escape(v)
            op_rows.append([f"<code>{escape(k)}</code>", val])

    op_est = to_float(op_counter.get("estimated_total_operations")) if isinstance(op_counter, dict) else None
    op_per_target = (op_est / n_targets) if op_est and n_targets else None

    anonymization_block = ""
    if metrics:
        anonymization_block = f"""
<section id="anonymization" class="card">
  <h2 class="section-title">3. Résultats d'anonymisation</h2>
  <div class="grid">
    <div class="metric"><div class="k">Classes d'équivalence</div><div class="v">{fmt_int(metrics.get('number_of_equivalence_classes'))}</div></div>
    <div class="metric"><div class="k">Taille moyenne</div><div class="v">{fmt_float(metrics.get('average_equivalence_class_size'))}</div></div>
    <div class="metric"><div class="k">Taille min</div><div class="v">{fmt_int(metrics.get('min_equivalence_class_size'))}</div></div>
    <div class="metric"><div class="k">Taille max</div><div class="v">{fmt_int(metrics.get('max_equivalence_class_size'))}</div></div>
    <div class="metric"><div class="k">Enregistrements supprimés</div><div class="v warn">{fmt_int(metrics.get('number_of_suppressed_records'))}</div></div>
    <div class="metric"><div class="k">Après retrait complet</div><div class="v">{fmt_int(metrics.get('n_rows_after_full_suppression_drop'))}</div></div>
  </div>
</section>
"""

    distribution_table = make_table(
        ["Groupe", "Avg. compatible count", "Median compatible count", "Avg. compatible fraction", "Median compatible fraction"],
        [
            [
                "IN",
                fmt_float(statistics.mean(member_count) if member_count else None),
                fmt_float(statistics.median(member_count) if member_count else None),
                fmt_pct(statistics.mean(member_frac) if member_frac else None),
                fmt_pct(statistics.median(member_frac) if member_frac else None),
            ],
            [
                "OUT",
                fmt_float(statistics.mean(non_member_count) if non_member_count else None),
                fmt_float(statistics.median(non_member_count) if non_member_count else None),
                fmt_pct(statistics.mean(non_member_frac) if non_member_frac else None),
                fmt_pct(statistics.median(non_member_frac) if non_member_frac else None),
            ],
        ],
    )

    op_section = ""
    if op_rows:
        op_section = f"""
<section id="complexity" class="card">
  <h2 class="section-title">6. Compteurs de complexité</h2>
  <div class="grid">
    <div class="metric"><div class="k">Opérations estimées</div><div class="v">{fmt_int(op_est)}</div></div>
    <div class="metric"><div class="k">Opérations par cible</div><div class="v">{fmt_float(op_per_target)}</div></div>
    <div class="metric"><div class="k">Décisions membership</div><div class="v">{fmt_int(op_counter.get('membership_decisions') if isinstance(op_counter, dict) else None)}</div></div>
  </div>
  <div style="margin-top:16px">{make_table(['Compteur', 'Valeur'], op_rows)}</div>
</section>
"""

    hist_section = ""
    if hist_labels:
        hist_section = f"""
<section id="distribution" class="card">
  <h2 class="section-title">5. Analyse IN vs OUT</h2>
  <div class="grid">
    <div class="metric"><div class="k">KS p-value</div><div class="v {'good' if ks_pvalue is not None else ''}">{fmt_float(ks_pvalue, 3)}</div></div>
    <div class="metric"><div class="k">Best accuracy via fraction seule</div><div class="v warn">{fmt_pct(best_acc)}</div></div>
    <div class="metric"><div class="k">Best threshold</div><div class="v">{fmt_pct(best_thr)}</div></div>
    <div class="metric"><div class="k">Seuil utilisé dans l'attaque</div><div class="v">{fmt_pct(max_compatible_fraction)}</div></div>
  </div>
  <div class="two" style="margin-top:16px;">
    <div class="chart-card">
      <h3 style="margin-top:0">Distribution de la fraction de candidats compatibles</h3>
      <div class="chart-wrap"><canvas id="histChart"></canvas></div>
    </div>
    <div>
      {distribution_table}
      <div class="callout" style="margin-top:14px;">
        Cette section regarde uniquement la variable <code>compatible_candidate_fraction</code>. Plus les distributions IN et OUT se recouvrent, plus la MIA a du mal à distinguer les membres des non-membres.
      </div>
    </div>
  </div>
</section>
"""

    chart_js = ""
    if hist_labels:
        chart_js = f"""
new Chart(document.getElementById('histChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(hist_labels, ensure_ascii=False)},
    datasets: [
      {{
        label: 'IN',
        data: {json.dumps(hist_in)},
        backgroundColor: 'rgba(122,162,255,0.58)',
        borderColor: 'rgba(122,162,255,0.95)',
        borderWidth: 1
      }},
      {{
        label: 'OUT',
        data: {json.dumps(hist_out)},
        backgroundColor: 'rgba(255,123,123,0.46)',
        borderColor: 'rgba(255,123,123,0.95)',
        borderWidth: 1
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: textColor }} }} }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Fraction de candidats compatibles (%)', color: textColor, font: {{ size: 12 }} }},
        ticks: {{ color: textColor, maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: {{ size: 11 }} }},
        grid: {{ display: false }}
      }},
      y: {{
        title: {{ display: true, text: 'Nombre de cibles', color: textColor, font: {{ size: 12 }} }},
        ticks: {{ color: textColor, font: {{ size: 11 }} }},
        grid: {{ color: gridColor }}
      }}
    }}
  }}
}});
"""

    html_doc = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{escape(report_title)}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>{STYLE}</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="eyebrow">TER</div>
    <h1>{escape(report_title)}</h1>
    <p class="lead">Rapport généré automatiquement à partir d'un <code>summary.json</code> et du fichier <code>targets.csv</code> de membership inference attack. Il synthétise la configuration de l'attaque, ses performances, la comparaison IN vs OUT et les compteurs de complexité.</p>
  </div>
</header>

<nav>
  <div class="wrap">
    <a href="#summary">Résumé</a>
    <a href="#protocol">Configuration</a>
    {'<a href="#anonymization">Anonymisation</a>' if metrics else ''}
    <a href="#mia">MIA</a>
    <a href="#distribution">Analyse IN vs OUT</a>
    {'<a href="#complexity">Complexité</a>' if op_rows else ''}
    <a href="#details">Détails bruts</a>
  </div>
</nav>

<main class="wrap">
<section id="summary" class="card">
  <h2 class="section-title">1. Résumé exécutif</h2>
  <div class="grid">{metrics_html}</div>
  <div class="two" style="margin-top:16px;">
    <div class="callout"><strong>Synthèse :</strong> {synthesis_html}</div>
    <div class="callout"><strong>Fichiers utilisés :</strong><br>{make_table(['Fichier', 'Chemin'], files_rows)}</div>
  </div>
</section>

<section id="protocol" class="card">
  <h2 class="section-title">2. Configuration</h2>
  {make_table(['Paramètre', 'Valeur'], config_rows)}
</section>

{anonymization_block}

<section id="mia" class="card">
  <h2 class="section-title">4. Analyse de la membership inference attack</h2>
  <div class="grid">
    <div class="metric"><div class="k">TP</div><div class="v good">{fmt_int(tp)}</div></div>
    <div class="metric"><div class="k">TN</div><div class="v accent">{fmt_int(tn)}</div></div>
    <div class="metric"><div class="k">FP</div><div class="v danger">{fmt_int(fp)}</div></div>
    <div class="metric"><div class="k">FN</div><div class="v warn">{fmt_int(fn)}</div></div>
  </div>

  <div class="two" style="margin-top:16px;">
    <div>
      <h3 style="margin-top:0">Matrice de confusion</h3>
      <div class="cm">
        <div></div>
        <div class="label">Prédit IN</div>
        <div class="label">Prédit OUT</div>
        <div class="label">Réel IN</div>
        <div class="cell tp"><strong>{fmt_int(tp)}</strong>TP · membres correctement détectés</div>
        <div class="cell fn"><strong>{fmt_int(fn)}</strong>FN · membres manqués</div>
        <div class="label">Réel OUT</div>
        <div class="cell fp"><strong>{fmt_int(fp)}</strong>FP · faux membres</div>
        <div class="cell tn"><strong>{fmt_int(tn)}</strong>TN · non-membres correctement rejetés</div>
      </div>
    </div>
    <div>
      {make_table(
        ['Indicateur', 'Valeur'],
        [
          ['Accuracy', fmt_pct(accuracy)],
          ['Precision', fmt_pct(precision)],
          ['Recall membres', fmt_pct(recall)],
          ['TNR non-membres', fmt_pct(tnr)],
          ['F1-score', fmt_pct(f1)],
          ['Prédits IN / OUT', f'{fmt_int(predicted_in_count)} / {fmt_int(predicted_out_count)}'],
          ['Avg. compatible count (IN)', fmt_float(statistics.mean(member_count) if member_count else None)],
          ['Avg. compatible count (OUT)', fmt_float(statistics.mean(non_member_count) if non_member_count else None)],
          ['Avg. compatible fraction (IN)', fmt_pct(statistics.mean(member_frac) if member_frac else None)],
          ['Avg. compatible fraction (OUT)', fmt_pct(statistics.mean(non_member_frac) if non_member_frac else None)],
        ],
      )}
    </div>
  </div>
</section>

{hist_section}
{op_section}

<section id="details" class="card">
  <h2 class="section-title">7. Détails bruts du summary.json</h2>
  <p class="section-subtitle">Cette section affiche les champs principaux du résumé d'attaque, sans interprétation détaillée.</p>
  {make_table(['Champ', 'Valeur'], summary_table_rows)}
  <p class="small" style="margin-top:14px;">Rapport généré le {escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
</section>
</main>

<script>
const textColor = '#98a2b3';
const gridColor = 'rgba(255,255,255,0.08)';
{chart_js}
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    project_root, summary_json, metrics_json, targets_csv, config_json = resolve_paths(args)
    summary = read_json(summary_json)
    attack_id = str(summary.get("attack_id", summary_json.parent.name))

    if args.output:
        output_path = args.output.resolve()
    else:
        output_path = summary_json.parent / f"{attack_id}__report.html"

    report_path = build_report(
        project_root=project_root,
        summary_json=summary_json,
        metrics_json=metrics_json,
        config_json=config_json,
        targets_csv=targets_csv,
        output_path=output_path,
        title=args.title,
    )
    print(f"[OK] Rapport généré: {report_path}")


if __name__ == "__main__":
    main()
