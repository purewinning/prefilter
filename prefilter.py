import pandas as pd
import argparse
from pathlib import Path

# ---- column mapping (edit these if your file uses different headers) ----
COLS = {
    "name": ["Name", "Player", "Player Name"],
    "team": ["Team", "Tm"],
    "pos":  ["Pos", "Position"],
    "salary": ["Salary"],
    "proj": ["Proj", "Projection", "Fpts", "FPTS"],
    "ceiling": ["Ceiling", "Ceil"],
    "own": ["Own", "Ownership", "Proj Own", "Ownership %"],
    "cpt_own": ["CPT Own", "Captain Own", "CPT%", "Captain Ownership"],
}

def pick_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    raise KeyError(f"Missing expected columns. Looked for: {options}. Found: {list(df.columns)}")

def main(csv_path: str, out_dir: str, pool_size: int, cpt_size: int):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    name_c = pick_col(df, COLS["name"])
    team_c = pick_col(df, COLS["team"])
    pos_c = pick_col(df, COLS["pos"])
    sal_c = pick_col(df, COLS["salary"])
    proj_c = pick_col(df, COLS["proj"])
    ceil_c = pick_col(df, COLS["ceiling"])
    own_c = pick_col(df, COLS["own"])

    # CPT ownership is optional
    cpt_col = None
    try:
        cpt_col = pick_col(df, COLS["cpt_own"])
    except Exception:
        pass

    work = df[[name_c, team_c, pos_c, sal_c, proj_c, ceil_c, own_c] + ([cpt_col] if cpt_col else [])].copy()
    work.columns = ["name","team","pos","salary","proj","ceiling","own"] + (["cpt_own"] if cpt_col else [])

    # Clean types
    for c in ["salary","proj","ceiling","own"] + (["cpt_own"] if "cpt_own" in work.columns else []):
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=["name","team","pos","salary","proj","ceiling","own"])

    # --- JBC-style scoring ---
    # Core score: ceiling with mild ownership tolerance (he doesn't hard-fade chalk)
    work["ceiling_leverage"] = work["ceiling"] / (work["own"].clip(lower=1.0))  # avoid divide by 0
    work["core_score"] = (
        0.60 * work["ceiling"] +
        0.25 * work["proj"] +
        0.15 * work["ceiling_leverage"]
    )

    # Attachment score: cheap TD pieces with ceiling per dollar
    work["ceiling_per_$1k"] = work["ceiling"] / (work["salary"] / 1000.0)
    work["attachment_score"] = (
        0.55 * work["ceiling_per_$1k"] +
        0.25 * work["ceiling_leverage"] +
        0.20 * (work["salary"] <= 4400).astype(int) * 10
    )

    # Captain score: ceiling + (optional) CPT leverage
    if "cpt_own" in work.columns:
        # prefer CPT ownership in the ~10-30% band (single-entry sweet spot), but not required
        work["cpt_band_bonus"] = ((work["cpt_own"].between(10, 30)).astype(int) * 5)
        work["cpt_score"] = 0.70 * work["ceiling"] + 0.20 * work["proj"] + 0.10 * (work["own"] - work["cpt_own"]) + work["cpt_band_bonus"]
    else:
        work["cpt_score"] = 0.75 * work["ceiling"] + 0.25 * work["proj"]

    # Build pool:
    # - Take top "core" across all
    core = work.sort_values("core_score", ascending=False).head(max(8, pool_size // 2))
    # - Take attachments (cheap/efficient) and union
    attach = work.sort_values("attachment_score", ascending=False).head(max(6, pool_size - len(core)))

    pool = pd.concat([core, attach]).drop_duplicates(subset=["name"]).sort_values("core_score", ascending=False)
    pool = pool.head(pool_size)

    # Captain pool:
    cpt_pool = work.sort_values("cpt_score", ascending=False).head(cpt_size)

    # Cheat sheet buckets
    core_names = set(core["name"].head(8))
    attach_names = set(attach["name"].head(8))
    pool["bucket"] = pool["name"].apply(lambda n: "CORE" if n in core_names else ("ATTACHMENT" if n in attach_names else "POOL"))

    # Outputs
    pool_out = out_dir / f"{csv_path.stem}_player_pool.csv"
    cpt_out  = out_dir / f"{csv_path.stem}_captain_pool.csv"
    cheat_out= out_dir / f"{csv_path.stem}_cheatsheet.csv"

    pool.to_csv(pool_out, index=False)
    cpt_pool.to_csv(cpt_out, index=False)
    pool.sort_values(["bucket","core_score"], ascending=[True,False]).to_csv(cheat_out, index=False)

    print("Wrote:")
    print(pool_out)
    print(cpt_out)
    print(cheat_out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to projections CSV")
    ap.add_argument("--out", default="out", help="Output folder")
    ap.add_argument("--pool_size", type=int, default=16, help="Total players in pool")
    ap.add_argument("--cpt_size", type=int, default=6, help="Captain pool size")
    args = ap.parse_args()

    main(args.csv, args.out, args.pool_size, args.cpt_size)
