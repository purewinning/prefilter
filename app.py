# app.py
# DraftKings Results Evaluator (Streamlit)
# Upload Results.csv + Projections.csv -> lineup-level metrics + winner blueprint + downloads

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# ----------------------------
# Helpers: parsing + matching
# ----------------------------

def normalize_name(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u00A0", " ")  # non-breaking space
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b(Jr\.?|III|II|IV)\b", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"[^\w\s]", "", s)
    return s.lower().strip()


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class MatchResult:
    key: Optional[str]
    score: float
    method: str


def safe_int(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip().replace(",", "")
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def split_lineup_cell(cell: str) -> List[str]:
    if pd.isna(cell):
        return []
    s = str(cell)
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def build_player_index(proj_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    raw_players = proj_df["Player"].astype(str).tolist()
    exact_map = {p: p for p in raw_players}
    norm_map = {}
    for p in raw_players:
        n = normalize_name(p)
        norm_map.setdefault(n, p)
    return exact_map, norm_map, raw_players


def match_player(
    name: str,
    exact_map: Dict[str, str],
    norm_map: Dict[str, str],
    raw_players: List[str],
    fuzzy_cutoff: float = 0.90
) -> MatchResult:
    if not isinstance(name, str):
        name = str(name)

    name = name.strip()
    if not name:
        return MatchResult(None, 0.0, "empty")

    if name in exact_map:
        return MatchResult(exact_map[name], 1.0, "exact")

    n = normalize_name(name)
    if n in norm_map:
        return MatchResult(norm_map[n], 0.995, "normalized")

    best_p, best_score = None, 0.0
    for p in raw_players:
        sc = seq_ratio(n, normalize_name(p))
        if sc > best_score:
            best_score = sc
            best_p = p

    if best_score >= fuzzy_cutoff:
        return MatchResult(best_p, best_score, "fuzzy")

    return MatchResult(None, best_score, "unmatched")


def log_own_metric(own_pct: float) -> float:
    if own_pct is None or pd.isna(own_pct):
        return 0.0
    v = max(float(own_pct), 0.01)
    return math.log(v / 100.0)


def compute_lineup_metrics(
    lineup_names: List[str],
    proj_df: pd.DataFrame,
    match_cache: Dict[str, MatchResult],
    exact_map: Dict[str, str],
    norm_map: Dict[str, str],
    raw_players: List[str],
    assume_first_is_cpt: bool,
    fuzzy_cutoff: float
) -> Dict:
    cpt_name = lineup_names[0] if (assume_first_is_cpt and len(lineup_names) >= 1) else None

    matched = []
    unmatched = []
    match_methods = []
    match_scores = []

    for nm in lineup_names:
        if nm in match_cache:
            mr = match_cache[nm]
        else:
            mr = match_player(nm, exact_map, norm_map, raw_players, fuzzy_cutoff=fuzzy_cutoff)
            match_cache[nm] = mr

        if mr.key is None:
            unmatched.append(nm)
        else:
            matched.append(mr.key)

        match_methods.append(mr.method)
        match_scores.append(mr.score)

    rows = proj_df[proj_df["Player"].isin(matched)].copy()
    by_player = {r["Player"]: r for _, r in rows.iterrows()}

    has_cpt_cols = all(col in proj_df.columns for col in ["CPT Ownership %", "CPT Optimal %", "CPT Leverage"])

    sum_salary = 0
    sum_proj = 0.0
    sum_own = 0.0
    sum_opt = 0.0
    sum_lev = 0.0
    sum_logown = 0.0
    punts = 0

    teams = []
    positions = []

    for nm in lineup_names:
        mr = match_cache.get(nm)
        if not mr or not mr.key:
            continue

        r = by_player.get(mr.key)
        if r is None:
            continue

        sal = safe_int(r.get("Salary"))
        prj = float(r.get("Projection", 0.0) if not pd.isna(r.get("Projection")) else 0.0)

        own = float(r.get("Ownership %", 0.0) if "Ownership %" in r and not pd.isna(r.get("Ownership %")) else 0.0)
        opt = float(r.get("Optimal %", 0.0) if "Optimal %" in r and not pd.isna(r.get("Optimal %")) else 0.0)
        lev = float(r.get("Leverage", 0.0) if "Leverage" in r and not pd.isna(r.get("Leverage")) else 0.0)

        is_cpt = (cpt_name is not None and nm == cpt_name)

        if is_cpt:
            if sal is not None:
                sal = int(round(sal * 1.5))
            prj *= 1.5
            if has_cpt_cols:
                own = float(r.get("CPT Ownership %", own))
                opt = float(r.get("CPT Optimal %", opt))
                lev = float(r.get("CPT Leverage", lev))

        if sal is None:
            sal = 0

        sum_salary += sal
        sum_proj += prj
        sum_own += own
        sum_opt += opt
        sum_lev += lev
        sum_logown += log_own_metric(own)

        if sal < 3000:
            punts += 1

        if "Team" in r and not pd.isna(r.get("Team")):
            teams.append(str(r.get("Team")))
        if "Position" in r and not pd.isna(r.get("Position")):
            positions.append(str(r.get("Position")))

    team_counts = pd.Series(teams).value_counts() if teams else pd.Series(dtype=int)
    team_split = "-".join(map(str, sorted(team_counts.tolist(), reverse=True))) if len(team_counts) else ""

    pos_counts = pd.Series(positions).value_counts() if positions else pd.Series(dtype=int)

    avg_match_score = sum(match_scores) / max(len(match_scores), 1)

    return {
        "n_players_listed": len(lineup_names),
        "cpt_assumed": bool(cpt_name),
        "cpt_name_raw": cpt_name or "",
        "salary_calc": int(sum_salary),
        "proj_calc": round(sum_proj, 3),
        "own_sum_calc": round(sum_own, 3),
        "optimal_sum_calc": round(sum_opt, 3),
        "leverage_sum_calc": round(sum_lev, 3),
        "logown_sum_calc": round(sum_logown, 6),
        "punts_lt_3k": int(punts),
        "team_split": team_split,
        "team1_count": int(team_counts.iloc[0]) if len(team_counts) else 0,
        "team2_count": int(team_counts.iloc[1]) if len(team_counts) > 1 else 0,
        "pos_counts": "; ".join([f"{k}:{int(v)}" for k, v in pos_counts.items()]) if len(pos_counts) else "",
        "match_avg_score": round(avg_match_score, 3),
        "unmatched_players": ", ".join(unmatched),
        "match_methods": ", ".join(match_methods),
    }


def infer_finish_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "Actual Finish Position",
        "Finish",
        "Rank",
        "Place",
        "Final Rank",
    ]
    lower_map = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def pct_describe(series: pd.Series) -> pd.DataFrame:
    return series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame().rename(columns={0: "value"})


# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(page_title="DK Results Evaluator", layout="wide")

st.title("DraftKings Results Evaluator")
st.caption("Upload **Results.csv** + **Projections.csv** to reverse-engineer winning roster construction.")

with st.sidebar:
    st.header("Inputs")
    results_file = st.file_uploader("Results.csv", type=["csv"])
    proj_file = st.file_uploader("Projections.csv", type=["csv"])

    st.divider()
    st.header("Settings")
    assume_first_is_cpt = st.checkbox("Showdown: assume first player listed is CPT", value=False)
    fuzzy_cutoff = st.slider("Name match strictness (fuzzy cutoff)", 0.80, 0.98, 0.90, 0.01)
    topn = st.number_input("Show top N lineups", min_value=5, max_value=200, value=30, step=5)
    top_window = st.number_input("Blueprint window (top K finishers)", min_value=50, max_value=2000, value=500, step=50)
    min_match_score = st.slider("Hide rows with match_avg_score < ...", 0.0, 1.0, 0.0, 0.05)

if not results_file or not proj_file:
    st.info("Upload both files to start.")
    st.stop()

# Load
try:
    proj_df = pd.read_csv(proj_file)
    res_df = pd.read_csv(results_file)
except Exception as e:
    st.error(f"Failed to read CSV(s): {e}")
    st.stop()

# Basic validation
if "Player" not in proj_df.columns:
    st.error("Projections.csv must contain a 'Player' column.")
    st.stop()

if "Lineups" not in res_df.columns:
    # Try common alternatives
    alt = None
    for c in res_df.columns:
        if c.lower().strip() in ("lineup", "roster", "players"):
            alt = c
            break
    if alt:
        res_df = res_df.rename(columns={alt: "Lineups"})
    else:
        st.error("Results.csv must contain a 'Lineups' column (comma-separated player names).")
        st.stop()

finish_col = infer_finish_column(res_df)
res_df["_finish_int"] = res_df[finish_col].apply(safe_int) if finish_col else None

# Compute metrics
with st.spinner("Evaluating lineups..."):
    exact_map, norm_map, raw_players = build_player_index(proj_df)
    match_cache: Dict[str, MatchResult] = {}

    computed_rows = []
    for _, row in res_df.iterrows():
        names = split_lineup_cell(row["Lineups"])
        m = compute_lineup_metrics(
            lineup_names=names,
            proj_df=proj_df,
            match_cache=match_cache,
            exact_map=exact_map,
            norm_map=norm_map,
            raw_players=raw_players,
            assume_first_is_cpt=assume_first_is_cpt,
            fuzzy_cutoff=fuzzy_cutoff,
        )

        out = dict(row)
        out.update(m)

        # Compare vs provided salary/ownsum if available
        if "Salary" in res_df.columns:
            rs = safe_int(row.get("Salary"))
            out["salary_diff_vs_results"] = (rs - out["salary_calc"]) if rs is not None else None
        else:
            out["salary_diff_vs_results"] = None

        if "OwnSum" in res_df.columns:
            s = str(row.get("OwnSum", "")).replace("%", "").strip()
            try:
                out["_ownsum_results"] = float(s)
                out["ownsum_diff_vs_results"] = out["_ownsum_results"] - out["own_sum_calc"]
            except Exception:
                out["_ownsum_results"] = None
                out["ownsum_diff_vs_results"] = None
        else:
            out["_ownsum_results"] = None
            out["ownsum_diff_vs_results"] = None

        computed_rows.append(out)

    out_df = pd.DataFrame(computed_rows)

# Filter match score if requested
if min_match_score > 0:
    out_df = out_df[out_df["match_avg_score"] >= float(min_match_score)].copy()

# Health
unmatched_rate = (out_df["unmatched_players"].fillna("").str.len() > 0).mean() if len(out_df) else 0.0

# Layout: high-level KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lineups evaluated", f"{len(out_df):,}")
c2.metric("Projection players", f"{len(proj_df):,}")
c3.metric("Fully-matched rows", f"{(1-unmatched_rate)*100:.1f}%")
c4.metric("CPT logic", "ON" if assume_first_is_cpt else "OFF")

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Winner Blueprint", "Top Lineups", "Distribution Explorer", "Download / Debug"])

with tab1:
    st.subheader("Winner Blueprint")

    if finish_col and out_df["_finish_int"].notna().any():
        field = out_df.copy()
        top = out_df.sort_values("_finish_int").head(int(top_window)).copy()

        left, right = st.columns(2)

        with left:
            st.markdown("**Salary Used (calc)**")
            st.dataframe(pd.concat(
                [pct_describe(top["salary_calc"]).rename(columns={"value": f"Top{top_window}"}),
                 pct_describe(field["salary_calc"]).rename(columns={"value": "Field"})],
                axis=1
            ), use_container_width=True)

            st.markdown("**Ownership Sum (calc)**")
            st.dataframe(pd.concat(
                [pct_describe(top["own_sum_calc"]).rename(columns={"value": f"Top{top_window}"}),
                 pct_describe(field["own_sum_calc"]).rename(columns={"value": "Field"})],
                axis=1
            ), use_container_width=True)

        with right:
            st.markdown("**Punts (<$3k) Frequency**")
            punts_tbl = pd.DataFrame({
                f"Top{top_window}": top["punts_lt_3k"].value_counts().sort_index(),
                "Field": field["punts_lt_3k"].value_counts().sort_index()
            }).fillna(0).astype(int)
            st.dataframe(punts_tbl, use_container_width=True)

            st.markdown("**Team Split Patterns (Top)**")
            split_tbl = pd.DataFrame({
                f"Top{top_window}": top["team_split"].value_counts().head(15),
                "Field": field["team_split"].value_counts().head(15)
            }).fillna(0).astype(int)
            st.dataframe(split_tbl, use_container_width=True)

        st.caption("Blueprint compares the best finishers vs the full field. Adjust Top K in the sidebar.")
    else:
        st.warning("No finish/rank column detected in Results.csv. Blueprint will use projection strength instead.")
        top = out_df.sort_values("proj_calc", ascending=False).head(int(top_window)).copy()
        st.dataframe(top[["salary_calc", "proj_calc", "own_sum_calc", "leverage_sum_calc", "punts_lt_3k", "team_split", "Lineups"]].head(50),
                     use_container_width=True)

with tab2:
    st.subheader("Top Lineups (with computed metrics)")

    if finish_col and out_df["_finish_int"].notna().any():
        view = out_df.sort_values("_finish_int").head(int(topn)).copy()
    else:
        view = out_df.sort_values("proj_calc", ascending=False).head(int(topn)).copy()

    show_cols = [c for c in [
        "_finish_int",
        "Duplicates",
        "Salary",
        "salary_calc",
        "salary_diff_vs_results",
        "proj_calc",
        "own_sum_calc",
        "leverage_sum_calc",
        "punts_lt_3k",
        "team_split",
        "cpt_name_raw",
        "unmatched_players",
        "match_avg_score",
        "Lineups",
    ] if c in view.columns]

    st.dataframe(view[show_cols], use_container_width=True, height=520)

    st.caption("If Salary/OwnSum diffs look large, your Results Lineups format may not match CPT assumptions or naming.")
    st.divider()

    st.markdown("### Quick “rules” from this slate (computed)")
    # Simple auto-rules for learning (data-driven)
    # Keep this short + blunt
    if len(out_df) >= 100:
        if finish_col and out_df["_finish_int"].notna().any():
            top = out_df.sort_values("_finish_int").head(int(top_window)).copy()
        else:
            top = out_df.sort_values("proj_calc", ascending=False).head(int(top_window)).copy()

        med_sal = int(top["salary_calc"].median())
        p90_left = int((50000 - top["salary_calc"]).quantile(0.9))
        avg_punts = float(top["punts_lt_3k"].mean())
        common_split = top["team_split"].value_counts().index[0] if top["team_split"].notna().any() else ""

        bullets = [
            f"Most strong lineups used about **${med_sal:,}** in salary.",
            f"90% of strong lineups left **≤ ${p90_left:,}** on the table.",
            f"Average punts (<$3k) among strong lineups: **{avg_punts:.2f}**.",
        ]
        if common_split:
            bullets.append(f"Most common team split among strong lineups: **{common_split}**.")

        for b in bullets:
            st.write("• " + b)

with tab3:
    st.subheader("Distribution Explorer")

    colA, colB = st.columns(2)
    metric = colA.selectbox("Metric", ["salary_calc", "proj_calc", "own_sum_calc", "leverage_sum_calc", "punts_lt_3k", "match_avg_score"])
    group = colB.selectbox("Group", ["Field", f"Top{top_window} (by finish/proj)"])

    if finish_col and out_df["_finish_int"].notna().any():
        top = out_df.sort_values("_finish_int").head(int(top_window)).copy()
    else:
        top = out_df.sort_values("proj_calc", ascending=False).head(int(top_window)).copy()

    series = out_df[metric] if group == "Field" else top[metric]

    st.write(f"**{group}** — {metric}")
    st.dataframe(pct_describe(series), use_container_width=True)

    # quick histogram without custom colors
    st.bar_chart(series.value_counts().sort_index() if metric == "punts_lt_3k" else series.round(0).value_counts().sort_index())

with tab4:
    st.subheader("Download & Debug")

    st.write("**Download evaluated_lineups.csv**")
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download evaluated_lineups.csv", data=csv_bytes, file_name="evaluated_lineups.csv", mime="text/csv")

    st.divider()
    st.markdown("### Debug: unmatched players")
    unmatched = out_df["unmatched_players"].fillna("")
    bad_rows = out_df[unmatched.str.len() > 0][["match_avg_score", "unmatched_players", "Lineups"]].head(50)
    st.dataframe(bad_rows, use_container_width=True, height=380)

    st.caption("If many are unmatched, lower fuzzy cutoff (0.88) or normalize naming in Results.csv (suffixes, punctuation).")
