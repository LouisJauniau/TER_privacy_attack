#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


STYLE = r'''
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
  .chart-wrap { position: relative; height: 320px; }
  .kv { display: grid; grid-template-columns: 210px 1fr; gap: 10px 16px; }
  .kv div:nth-child(odd) { color: var(--muted); }
  ul.clean { margin: 8px 0 0 18px; padding: 0; }
  @media (max-width: 980px) { .two { grid-template-columns: 1fr; } }
  @media print {
    nav { display: none; }
    body { background: white; color: black; }
    .card, .metric, table, .chart-card { box-shadow: none; }
  }
'''


SUMMARY_KEYS = [
    "attack_id",
    "known_attrs",
    "qid_filter_attrs",
    "refine_attrs",
    "target_id_col",
    "sensitive_attr",
    "n_targets",
    "seed",
    "n_anonymized_rows",
    "use_privjedai_fuzzy",
    "unique_reidentification_rate",
    "false_unique_match_rate",
    "true_record_kept_after_refinement_rate",
    "avg_qid_equivalence_class_size",
    "median_qid_equivalence_class_size",
    "avg_equivalence_class_size",
    "median_equivalence_class_size",
    "max_equivalence_class_size",
    "avg_reduction_rate",
    "certainty_sensitive_inference_rate",
    "avg_true_sensitive_probability",
    "median_true_sensitive_probability",
    "avg_top_sensitive_probability",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Génère un rapport HTML pour une linkage attack.")
    p.add_argument("--project-root", type=Path, default=Path.cwd(), help="Racine du projet")
    p.add_argument("--attack-dir", type=Path, default=None, help="Dossier de l'attaque linkage")
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


def find_latest_linkage_dir(project_root: Path) -> Path:
    base = project_root / "outputs" / "attacks" / "linkage"
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
        attack_dir = args.attack_dir.resolve() if args.attack_dir else find_latest_linkage_dir(project_root)
        summary_json = attack_dir / "summary.json"

    targets_csv = args.targets_csv.resolve() if args.targets_csv else attack_dir / "targets.csv"
    summary = read_json(summary_json)

    if args.metrics_json:
        metrics_json = args.metrics_json.resolve()
    else:
        attack_id = str(summary.get("attack_id", attack_dir.name))
        experiment_id = attack_id.split("__known_")[0]
        candidate = project_root / "outputs" / "metrics" / f"{experiment_id}.json"
        metrics_json = candidate if candidate.exists() else None

    if args.config_json:
        config_json = args.config_json.resolve()
    else:
        config_path_in_summary = summary.get("config_path")
        config_json = None
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
            experiment_id = attack_id.split("__known_")[0]
            fallback = project_root / "outputs" / "configs" / f"{experiment_id}.json"
            if fallback.exists():
                config_json = fallback

    return project_root, summary_json, metrics_json, targets_csv, config_json


def build_sensitive_stats(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_value: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_value[row.get("true_sensitive_value", "?")].append(row)

    total = len(rows) or 1
    out: list[dict[str, Any]] = []
    for value, bucket in sorted(by_value.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        correct = sum(1 for r in bucket if r.get("predicted_sensitive_top_value") == r.get("true_sensitive_value"))
        probs = [to_float(r.get("true_sensitive_probability")) for r in bucket]
        probs = [p for p in probs if p is not None]
        certainty = [1.0 if to_bool(r.get("sensitive_value_certain")) else 0.0 for r in bucket]
        out.append(
            {
                "value": value,
                "count": len(bucket),
                "share": len(bucket) / total,
                "correct": correct,
                "correct_rate": correct / len(bucket) if bucket else None,
                "avg_true_prob": statistics.mean(probs) if probs else None,
                "median_true_prob": statistics.median(probs) if probs else None,
                "certainty_rate": statistics.mean(certainty) if certainty else None,
            }
        )
    return out


def histogram(values: list[float], n_bins: int, value_min: float | None = None, value_max: float | None = None) -> tuple[list[str], list[int]]:
    if not values:
        return [], []
    lo = min(values) if value_min is None else value_min
    hi = max(values) if value_max is None else value_max
    if math.isclose(lo, hi):
        return [f"{lo:.2f}"], [len(values)]

    width = (hi - lo) / n_bins
    labels: list[str] = []
    counts = [0 for _ in range(n_bins)]
    for i in range(n_bins):
        a = lo + i * width
        b = lo + (i + 1) * width
        labels.append(f"{a:.2f}–{b:.2f}")
    for v in values:
        idx = int((v - lo) / width)
        if idx == n_bins:
            idx -= 1
        counts[idx] += 1
    return labels, counts


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    thead = "<thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr></thead>"
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"


def rel_or_abs(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path)


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
    sensitive_attr = str(summary.get("sensitive_attr", "sensitive_attr"))
    report_title = title or f"Rapport linkage – {attack_id}"

    unique_rate = to_float(summary.get("unique_reidentification_rate"))
    certainty_rate = to_float(summary.get("certainty_sensitive_inference_rate"))
    avg_class_size = to_float(summary.get("avg_equivalence_class_size"))
    avg_true_prob = to_float(summary.get("avg_true_sensitive_probability"))
    avg_reduction_rate = to_float(summary.get("avg_reduction_rate"))
    n_targets = int(float(summary.get("n_targets", len(rows) or 0)))
    n_rows = int(float(summary.get("n_anonymized_rows", 0))) if summary.get("n_anonymized_rows") is not None else 0

    exact_unique_count = round((unique_rate or 0.0) * n_targets)
    certainty_count = round((certainty_rate or 0.0) * n_targets)

    eq_sizes = [to_float(r.get("equivalence_class_size")) for r in rows]
    eq_sizes = [x for x in eq_sizes if x is not None]
    true_probs = [to_float(r.get("true_sensitive_probability")) for r in rows]
    true_probs = [x for x in true_probs if x is not None]
    reduction_rates = [to_float(r.get("equivalence_class_reduction_rate")) for r in rows]
    reduction_rates = [x for x in reduction_rates if x is not None]

    predicted_counts = Counter(r.get("predicted_sensitive_top_value", "?") for r in rows)
    predicted_top_value, predicted_top_count = (predicted_counts.most_common(1)[0] if predicted_counts else ("—", 0))

    sensitive_rows = build_sensitive_stats(rows)
    sensitive_table_rows = []
    for row in sensitive_rows:
        sensitive_table_rows.append(
            [
                f"<code>{escape(row['value'])}</code>",
                fmt_int(row["count"]),
                fmt_pct(row["share"]),
                f"{fmt_int(row['correct'])} / {fmt_int(row['count'])}",
                fmt_pct(row["correct_rate"]),
                fmt_pct(row["avg_true_prob"]),
                fmt_pct(row["median_true_prob"]),
                fmt_pct(row["certainty_rate"]),
            ]
        )

    summary_table_rows = []
    for key in SUMMARY_KEYS:
        if key in summary:
            value = summary[key]
            if isinstance(value, list):
                value_str = fmt_list(value)
            elif isinstance(value, bool):
                value_str = "true" if value else "false"
            elif isinstance(value, float) and (key.endswith("_rate") or "probability" in key):
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

    config_rows = [
        ["Attack ID", f"<code>{escape(attack_id)}</code>"],
        ["Attributs connus", escape(fmt_list(summary.get("known_attrs")))],
        ["QI pour filtrage", escape(fmt_list(summary.get("qid_filter_attrs")))],
        ["Attributs de réduction", escape(fmt_list(summary.get("refine_attrs")))],
        ["Attribut sensible", f"<code>{escape(sensitive_attr)}</code>"],
        ["PrivJedAI fuzzy", escape(str(summary.get("use_privjedai_fuzzy", False)).lower())],
        ["Seed", fmt_int(summary.get("seed"))],
        ["Cibles", fmt_int(n_targets)],
        ["Lignes anonymisées", fmt_int(n_rows)],
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

    eq_labels, eq_counts = histogram(eq_sizes, 12) if eq_sizes else ([], [])
    prob_labels, prob_counts = histogram(true_probs, 10, 0.0, 1.0) if true_probs else ([], [])
    red_labels, red_counts = histogram(reduction_rates, 10, 0.0, 1.0) if reduction_rates else ([], [])

    synthesis_parts = []
    if unique_rate is not None:
        synthesis_parts.append(
            f"La ré-identification exacte unique reste à <strong>{fmt_pct(unique_rate)}</strong> ({fmt_int(exact_unique_count)} cibles)."
        )
    if avg_class_size is not None:
        synthesis_parts.append(
            f"La taille moyenne des classes d'équivalence finales est de <strong>{fmt_float(avg_class_size)}</strong>."
        )
    if avg_true_prob is not None:
        synthesis_parts.append(
            f"La probabilité moyenne de la vraie valeur sensible vaut <strong>{fmt_pct(avg_true_prob)}</strong>."
        )
    if predicted_top_count:
        synthesis_parts.append(
            f"La valeur sensible la plus souvent prédite est <code>{escape(predicted_top_value)}</code> sur {fmt_int(predicted_top_count)} cibles."
        )
    if avg_reduction_rate is not None:
        synthesis_parts.append(
            f"La réduction moyenne de classe entre le filtrage QI et la classe finale est de <strong>{fmt_pct(avg_reduction_rate)}</strong>."
        )
    synthesis_html = " ".join(synthesis_parts) if synthesis_parts else "Aucune synthèse disponible."

    op_est = None
    if isinstance(op_counter, dict):
        op_est = to_float(op_counter.get("estimated_total_operations"))
    op_per_target = (op_est / n_targets) if op_est and n_targets else None

    metrics_cards = [
        ("Cibles attaquées", fmt_int(n_targets), ""),
        ("Ré-identification exacte unique", fmt_pct(unique_rate), "good"),
        ("Cibles uniques exactes", fmt_int(exact_unique_count), "good"),
        ("Taille moyenne classe", fmt_float(avg_class_size), ""),
        ("Inférence sensible certaine", fmt_pct(certainty_rate), "warn"),
        ("Cibles certaines", fmt_int(certainty_count), "warn"),
        ("Prob. moyenne vraie valeur", fmt_pct(avg_true_prob), "accent"),
        ("Réduction moyenne", fmt_pct(avg_reduction_rate), "accent"),
    ]
    metrics_html = "".join(
        f"<div class='metric'><div class='k'>{escape(k)}</div><div class='v {cls}'>{v}</div></div>"
        for k, v, cls in metrics_cards
    )

    anonymization_block = ""
    if metrics:
        anonymization_block = f"""
<section id=\"anonymization\" class=\"card\">
  <h2 class=\"section-title\">3. Résultats d'anonymisation</h2>
  <div class=\"grid\">
    <div class=\"metric\"><div class=\"k\">Classes d'équivalence</div><div class=\"v\">{fmt_int(metrics.get('number_of_equivalence_classes'))}</div></div>
    <div class=\"metric\"><div class=\"k\">Taille moyenne</div><div class=\"v\">{fmt_float(metrics.get('average_equivalence_class_size'))}</div></div>
    <div class=\"metric\"><div class=\"k\">Taille min</div><div class=\"v\">{fmt_int(metrics.get('min_equivalence_class_size'))}</div></div>
    <div class=\"metric\"><div class=\"k\">Taille max</div><div class=\"v\">{fmt_int(metrics.get('max_equivalence_class_size'))}</div></div>
    <div class=\"metric\"><div class=\"k\">Enregistrements supprimés</div><div class=\"v warn\">{fmt_int(metrics.get('number_of_suppressed_records'))}</div></div>
    <div class=\"metric\"><div class=\"k\">Après retrait des lignes totalement supprimées</div><div class=\"v\">{fmt_int(metrics.get('n_rows_after_full_suppression_drop'))}</div></div>
  </div>
</section>
"""

    op_section = ""
    if op_rows:
        op_section = f"""
<section id=\"complexity\" class=\"card\">
  <h2 class=\"section-title\">6. Compteurs de complexité</h2>
  <div class=\"grid\">
    <div class=\"metric\"><div class=\"k\">Opérations estimées</div><div class=\"v\">{fmt_int(op_est)}</div></div>
    <div class=\"metric\"><div class=\"k\">Opérations par cible</div><div class=\"v\">{fmt_float(op_per_target)}</div></div>
    <div class=\"metric\"><div class=\"k\">Lignes anonymisées</div><div class=\"v\">{fmt_int(n_rows)}</div></div>
  </div>
  <div style=\"margin-top:16px\">{make_table(['Compteur', 'Valeur'], op_rows)}</div>
</section>
"""

    chart_blocks = []
    chart_init = []
    if eq_labels:
        chart_blocks.append("""
      <div class=\"chart-card\">
        <h3 style=\"margin-top:0\">Distribution des tailles de classes finales</h3>
        <div class=\"chart-wrap\"><canvas id=\"eqChart\"></canvas></div>
      </div>
""")
        chart_init.append(
            "new Chart(document.getElementById('eqChart'), {type:'bar', data:{labels:%s, datasets:[{label:'Taille de classe', data:%s}]}, options:baseChartOptions('Nombre de cibles')});"
            % (json.dumps(eq_labels, ensure_ascii=False), json.dumps(eq_counts))
        )

    chart_section = ""
    if chart_blocks:
        chart_section = f"""
<section id=\"distribution\" class=\"card\">
  <h2 class=\"section-title\">5. Distribution</h2>
  <div class=\"two\">{''.join(chart_blocks)}</div>
</section>
"""

    sensitive_section = ""
    if sensitive_table_rows:
        sensitive_section = f"""
<section id=\"sensitive\" class=\"card\">
  <h2 class=\"section-title\">4. Performance par valeur de l'attribut sensible</h2>
  {make_table(
      [
          f"Valeur réelle de {sensitive_attr}",
          "Nombre de cibles",
          "Part des cibles",
          "Prédictions correctes",
          "Taux de bonne prédiction",
          "Avg. true sensitive prob.",
          "Median true sensitive prob.",
          "Certainty rate",
      ],
      sensitive_table_rows,
  )}
</section>
"""

    html_doc = f"""<!DOCTYPE html>
<html lang=\"fr\">
<head>
<meta charset=\"utf-8\"/>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
<title>{escape(report_title)}</title>
<script src=\"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js\"></script>
<style>{STYLE}</style>
</head>
<body>
<header>
  <div class=\"wrap\">
    <div class=\"eyebrow\">TER</div>
    <h1>{escape(report_title)}</h1>
    <p class=\"lead\">Rapport généré automatiquement à partir d'un <code>summary.json</code> de linkage attack et du fichier <code>targets.csv</code></p>
  </div>
</header>

<nav>
  <div class=\"wrap\">
    <a href=\"#summary\">Résumé</a>
    <a href=\"#protocol\">Configuration</a>
    {'<a href="#anonymization">Anonymisation</a>' if metrics else ''}
    <a href=\"#sensitive\">Attribut sensible</a>
    <a href=\"#distributions\">Distributions</a>
    {'<a href="#complexity">Complexité</a>' if op_rows else ''}
    <a href=\"#details\">Détails bruts</a>
  </div>
</nav>

<main class=\"wrap\">
<section id=\"summary\" class=\"card\">
  <h2 class=\"section-title\">1. Résumé exécutif</h2>
  <div class=\"grid\">{metrics_html}</div>
  <div class=\"two\" style=\"margin-top:16px;\">
    <div class=\"callout\"><strong>Synthèse :</strong> {synthesis_html}</div>
    <div class=\"callout\"><strong>Fichiers utilisés :</strong><br>{make_table(['Fichier', 'Chemin'], files_rows)}</div>
  </div>
</section>

<section id=\"protocol\" class=\"card\">
  <h2 class=\"section-title\">2. Configuration</h2>
  {make_table(['Paramètre', 'Valeur'], config_rows)}
</section>

{anonymization_block}
{sensitive_section}
{chart_section}
{op_section}

<section id=\"details\" class=\"card\">
  <h2 class=\"section-title\">7. Détails bruts du summary.json</h2>
  <p class=\"section-subtitle\">Cette section affiche les champs principaux du résumé d'attaque, sans interprétation détaillée.</p>
  {make_table(['Champ', 'Valeur'], summary_table_rows)}
  <p class=\"small\" style=\"margin-top:14px;\">Rapport généré le {escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
</section>
</main>

<script>
const textColor = '#98a2b3';
const gridColor = 'rgba(255,255,255,0.08)';
function baseChartOptions(yTitle) {{
  return {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: textColor, maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: {{ size: 11 }} }}, grid: {{ display: false }} }},
      y: {{ title: {{ display: true, text: yTitle, color: textColor, font: {{ size: 12 }} }}, ticks: {{ color: textColor, font: {{ size: 11 }} }}, grid: {{ color: gridColor }} }}
    }}
  }};
}}
{''.join(chart_init)}
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
