import pandas as pd
import argparse
from pathlib import Path
import re

# -----------------------------
# Utilities
# -----------------------------
def norm(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Missing expected column. Tried: {candidates}. Found: {list(df.columns)}"
        )
    return None

def to_num(series: pd.Series) -> pd.Series:
    # handles "42.3%" strings too
    s = series.astype(str).str.replace("%", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

# -----------------------------
# Slate type detection
# -----------------------------
def classify_slate(x: pd.DataFrame) -> tuple[str, dict]:
    """
    Returns (slate_type, diagnostics)
    slate_type ∈ {"RAW", "LEVERAGE"}

    RAW DOMINANCE heuristic (needs optimal_pct + ceiling + own + pos):
      1) top_ceiling >= 1.25 * median_ceiling
      2) top_optimal >= 30
      3) top_own >= 35
      4) max QB optimal >= 25
    """
    diagnostics = {}

    required_cols = {"ceiling", "own", "pos", "optimal_pct"}
    if not required_cols.issubset(set(x.columns)):
        diagnostics["reason"] = (
            f"Missing columns for slate classification: "
            f"{sorted(list(required_cols - set(x.columns)))}. Defaulting to LEVERAGE."
        )
        return "LEVERAGE", diagnostics

    tmp = x.dropna(subset=["ceiling", "own", "optimal_pct"]).copy()
    if tmp.empty:
        diagnostics["reason"] = "No usable rows for slate classification. Defaulting to LEVERAGE."
        return "LEVERAGE", diagnostics

    median_ceiling = tmp["ceiling"].median()
    top = tmp.sort_values("ceiling", ascending=False).iloc[0]
    qb_max_opt = tmp[tmp["pos"].eq("QB")]["optimal_pct"].max() if (tmp["pos"].eq("QB").any()) else 0.0

    cond1 = top["ceiling"] >= 1.25 * median_ceiling
    cond2 = top["optimal_pct"] >= 30
    cond3 = top["own"] >= 35
    cond4 = qb_max_opt >= 25

    diagnostics.update({
        "median_ceiling": float(median_ceiling),
        "top_player": str(top.get("player", "")),
        "top_ceiling": float(top["ceiling"]),
        "top_own": float(top["own"]),
        "top_optimal_pct": float(top["optimal_pct"]),
        "qb_max_optimal_pct": float(qb_max_opt),
        "conditions": {"c1_top_ceiling_vs_median": bool(cond1),
                       "c2_top_optimal_ge_30": bool(cond2),
                       "c3_top_own_ge_35": bool(cond3),
                       "c4_qb_opt_ge_25": bool(cond4)}
    })

    is_raw = cond1 and cond2 and cond3 and cond4
    diagnostics["slate_type"] = "RAW" if is_raw else "LEVERAGE"
    return ("RAW" if is_raw else "LEVERAGE"), diagnostics

# -----------------------------
# Core logic
# -----------------------------
def build_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ingest a projections-style file. Supports typical Stokastic/SaberSim exports.

    Required:
      Player/Name, Team, Position, Salary, Proj, Ceiling, Own

    Optional:
      Optimal %, CPT Own, CPT Optimal %
    """
    # Required mappings
    name_c = pick_col(df, ["Player", "Name", "Player Name"])
    team_c = pick_col(df, ["Team", "Tm"])
    pos_c  = pick_col(df, ["Position", "Pos"])
    sal_c  = pick_col(df, ["Salary"])
    proj_c = pick_col(df, ["Proj", "Projection", "FPTS", "Fpts", "Fpts.", "Pts"])
    ceil_c = pick_col(df, ["Ceiling", "Ceil"])
    own_c  = pick_col(df, ["Own", "Ownership", "Proj Own", "Ownership %", "Ownership%"])

    # Optional mappings
    opt_c = pick_col(df, ["Optimal %", "Optimal%", "Optimal", "Opt%", "Optimal Pct", "OptimalPct"], required=False)
    cpt_own_c = pick_col(df, ["CPT Own", "Captain Own", "CPT%", "Captain Ownership", "Captain Ownership %"], required=False)
    cpt_opt_c = pick_col(df, ["CPT Optimal %", "Captain Optimal %", "CPT Opt%", "Captain Optimal%", "CPT Optimal"], required=False)

    keep = [c for c in [name_c, team_c, pos_c, sal_c, proj_c, ceil_c, own_c, opt_c, cpt_own_c, cpt_opt_c] if c]
    x = df[keep].copy()

    rename = {
        name_c: "player",
        team_c: "team",
        pos_c: "pos",
        sal_c: "salary",
        proj_c: "proj",
        ceil_c: "ceiling",
        own_c: "own",
    }
    if opt_c: rename[opt_c] = "optimal_pct"
    if cpt_own_c: rename[cpt_own_c] = "cpt_own"
    if cpt_opt_c: rename[cpt_opt_c] = "cpt_optimal_pct"

    x = x.rename(columns=rename)

    # Normalize
    x["player"] = x["player"].map(norm)
    x["pos"] = x["pos"].astype(str).str.upper().str.strip()

    # Numerics
    for c in ["salary", "proj", "ceiling", "own", "optimal_pct", "cpt_own", "cpt_optimal_pct"]:
        if c in x.columns:
            x[c] = to_num(x[c])

    # Drop unusable rows
    x = x.dropna(subset=["player", "team", "pos", "salary", "proj", "ceiling", "own"]).copy()

    # Derived metrics (common to both modes)
    x["ceil_per_1k"] = x["ceiling"] / (x["salary"] / 1000.0)
    x["ceiling_leverage"] = x["ceiling"] / x["own"].clip(lower=1.0)

    # If Optimal % exists, also compute true leverage = Optimal - Own
    if "optimal_pct" in x.columns:
        x["opt_leverage"] = x["optimal_pct"] - x["own"]

    return x

def score_and_bucket(x: pd.DataFrame) -> tuple[pd.DataFrame, str, dict]:
    """
    Adds:
      - slate_type
      - core_score / attachment_score
      - cpt_score (branched by slate_type)
      - bucket labels
    """
    slate_type, diag = classify_slate(x)
    x = x.copy()
    x["slate_type"] = slate_type

    # --- Core/attachment scoring (stable across slate types) ---
    x["core_score"] = (
        0.55 * x["ceiling"] +
        0.25 * x["proj"] +
        0.20 * x["ceiling_leverage"]
    )

    x["attachment_score"] = (
        0.60 * x["ceil_per_1k"] +
        0.25 * x["ceiling_leverage"] +
        0.15 * (x["salary"] <= 4400).astype(int) * 10
    )

    # --- Captain scoring: RAW vs LEVERAGE mode ---
    if slate_type == "RAW":
        # In RAW mode, do not get cute: prioritize raw ceiling + optimal frequency.
        # If optimal_pct isn't available, it’ll just be treated as 0 (but classification would have defaulted to LEVERAGE).
        opt = x["optimal_pct"] if "optimal_pct" in x.columns else 0.0
        x["cpt_score"] = (0.70 * x["ceiling"] + 0.20 * x["proj"] + 0.10 * opt.fillna(0.0))
    else:
        # LEVERAGE mode: ceiling + ceiling/own + CPT leverage if present.
        if "cpt_own" in x.columns:
            # Prefer CPT in a reasonable band (10–30), plus favor when CPT own < FLEX own (positive "gap")
            x["cpt_band_bonus"] = x["cpt_own"].between(10, 30).astype(int) * 5
            gap = (x["own"] - x["cpt_own"]).fillna(0.0)
            x["cpt_score"] = (
                0.55 * x["ceiling"] +
                0.25 * x["ceiling_leverage"] +
                0.15 * gap +
                0.05 * x["cpt_band_bonus"]
            )
        else:
            x["cpt_score"] = (0.65 * x["ceiling"] + 0.35 * x["ceiling_leverage"])

    # --- Bucket labels (cheat sheet) ---
    x["bucket"] = "POOL"

    # CORE: chalk + ceiling gravity
    own_hi = x["own"].quantile(0.75)
    ceil_hi = x["ceiling"].quantile(0.75)
    x.loc[(x["own"] >= own_hi) & (x["ceiling"] >= ceil_hi), "bucket"] = "CORE"

    # LEVERAGE: either opt_leverage or ceiling_leverage
    if "opt_leverage" in x.columns:
        lev_hi = x["opt_leverage"].quantile(0.80)
        x.loc[x["opt_leverage"] >= lev_hi, "bucket"] = "LEVERAGE"
    else:
        lev_hi = x["ceiling_leverage"].quantile(0.80)
        x.loc[x["ceiling_leverage"] >= lev_hi, "bucket"] = "LEVERAGE"

    # ATTACHMENT: cheap, efficient
    x.loc[(x["salary"] <= 4400) & (x["ceil_per_1k"] >= x["ceil_per_1k"].quantile(0.70)), "bucket"] = "ATTACHMENT"

    # FADE/TRAP: popular but poor leverage
    if "opt_leverage" in x.columns:
        trap_lo = x["opt_leverage"].quantile(0.25)
        x.loc[(x["own"] >= own_hi) & (x["opt_leverage"] <= trap_lo), "bucket"] = "FADE"
    else:
        trap_lo = x["ceiling_leverage"].quantile(0.25)
        x.loc[(x["own"] >= own_hi) & (x["ceiling_leverage"] <= trap_lo), "bucket"] = "FADE"

    return x, slate_type, diag

def build_pools(x: pd.DataFrame, pool_size: int, cpt_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (player_pool_df, captain_pool_df)

    RAW mode:
      - force the "hammer" (top ceiling) into captain pool
      - slightly larger emphasis on CORE
    LEVERAGE mode:
      - CPT pool comes from cpt_score as usual
    """
    slate_type = x["slate_type"].iloc[0] if "slate_type" in x.columns and len(x) else "LEVERAGE"

    core = x[x["bucket"].eq("CORE")].sort_values("core_score", ascending=False)
    lev  = x[x["bucket"].eq("LEVERAGE")].sort_values("core_score", ascending=False)
    att  = x[x["bucket"].eq("ATTACHMENT")].sort_values("attachment_score", ascending=False)

    if slate_type == "RAW":
        pool = pd.concat([
            core.head(max(8, pool_size // 2)),
            lev.head(max(4, pool_size // 4)),
            att.head(max(4, pool_size // 4)),
        ]).drop_duplicates(subset=["player"])
    else:
        pool = pd.concat([
            core.head(max(6, pool_size // 3)),
            lev.head(max(6, pool_size // 3)),
            att.head(max(4, pool_size // 4)),
        ]).drop_duplicates(subset=["player"])

    # top up
    if len(pool) < pool_size:
        topup = x.sort_values("core_score", ascending=False)
        pool = pd.concat([pool, topup]).drop_duplicates(subset=["player"]).head(pool_size)
    else:
        pool = pool.head(pool_size)

    # Captain pool: avoid pure punts (optional rule: salary >= 6000)
    cpt_candidates = x.copy()
    cpt_candidates = cpt_candidates[cpt_candidates["salary"] >= 6000] if len(cpt_candidates) else cpt_candidates
    cpt_pool = cpt_candidates.sort_values("cpt_score", ascending=False).head(cpt_size)

    # Force hammer in RAW mode (top ceiling player)
    if slate_type == "RAW" and len(x):
        hammer = x.sort_values("ceiling", ascending=False).head(1)
        cpt_pool = pd.concat([hammer, cpt_pool]).drop_duplicates(subset=["player"]).head(cpt_size)

    return pool, cpt_pool

# -----------------------------
# Entry point
# -----------------------------
def main(csv_path: str, out_dir: str, pool_size: int, cpt_size: int):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    x = build_frame(df)
    x, slate_type, diag = score_and_bucket(x)
    pool, cpt_pool = build_pools(x, pool_size=pool_size, cpt_size=cpt_size)

    stem = csv_path.stem
    cheat_path = out_dir / f"{stem}_cheatsheet.csv"
    pool_path  = out_dir / f"{stem}_player_pool.csv"
    cpt_path   = out_dir / f"{stem}_captain_pool.csv"
    diag_path  = out_dir / f"{stem}_slate_diagnostics.txt"

    x.sort_values(["bucket", "core_score"], ascending=[True, False]).to_csv(cheat_path, index=False)
    pool.to_csv(pool_path, index=False)
    cpt_pool.to_csv(cpt_path, index=False)

    # diagnostics for transparency
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write(f"SLATE TYPE: {slate_type}\n")
        for k, v in diag.items():
            f.write(f"{k}: {v}\n")

    print(f"SLATE TYPE: {slate_type}")
    if "reason" in diag:
        print(f"Note: {diag['reason']}")
    print("Wrote:")
    print(" ", cheat_path)
    print(" ", pool_path)
    print(" ", cpt_path)
    print(" ", diag_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Projections CSV (imported every slate)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--pool_size", type=int, default=16, help="Player pool size (jbc-style 12–18)")
    ap.add_argument("--cpt_size", type=int, default=6, help="Captain pool size (4–7)")
    args = ap.parse_args()
    main(args.csv, args.out, args.pool_size, args.cpt_size)
