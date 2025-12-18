import pandas as pd
import argparse
from pathlib import Path
import re

def norm(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing expected column. Tried: {candidates}. Found: {list(df.columns)}")

def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def build_cheatsheet(df: pd.DataFrame) -> pd.DataFrame:
    name_c = pick_col(df, ["Player", "Name", "Player Name"])
    team_c = pick_col(df, ["Team", "Tm"])
    pos_c  = pick_col(df, ["Position", "Pos"])
    sal_c  = pick_col(df, ["Salary"])
    proj_c = pick_col(df, ["Proj", "Projection", "FPTS", "Fpts"])
    ceil_c = pick_col(df, ["Ceiling", "Ceil"])
    own_c  = pick_col(df, ["Own", "Ownership", "Proj Own", "Ownership %"])

    cpt_c = None
    for opt in ["CPT Own", "Captain Own", "CPT%", "Captain Ownership"]:
        if opt in df.columns:
            cpt_c = opt
            break

    keep = [name_c, team_c, pos_c, sal_c, proj_c, ceil_c, own_c] + ([cpt_c] if cpt_c else [])
    x = df[keep].copy()

    x = x.rename(columns={
        name_c: "player",
        team_c: "team",
        pos_c: "pos",
        sal_c: "salary",
        proj_c: "proj",
        ceil_c: "ceiling",
        own_c: "own",
        **({cpt_c: "cpt_own"} if cpt_c else {})
    })

    x["player"] = x["player"].map(norm)
    x["pos"] = x["pos"].astype(str).str.upper().str.strip()

    for c in ["salary", "proj", "ceiling", "own"] + (["cpt_own"] if "cpt_own" in x.columns else []):
        x[c] = to_num(x[c])

    x = x.dropna(subset=["player", "team", "pos", "salary", "proj", "ceiling", "own"])

    x["ceil_per_1k"] = x["ceiling"] / (x["salary"] / 1000.0)
    x["ceiling_leverage"] = x["ceiling"] / x["own"].clip(lower=1.0)

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

    if "cpt_own" in x.columns:
        x["cpt_band_bonus"] = x["cpt_own"].between(10, 30).astype(int) * 5
        x["cpt_score"] = (
            0.65 * x["ceiling"] +
            0.20 * x["proj"] +
            0.10 * (x["own"] - x["cpt_own"]) +
            0.05 * x["cpt_band_bonus"]
        )
    else:
        x["cpt_score"] = 0.70 * x["ceiling"] + 0.30 * x["proj"]

    x["bucket"] = "POOL"
    own_hi = x["own"].quantile(0.75)
    ceil_hi = x["ceiling"].quantile(0.75)
    x.loc[(x["own"] >= own_hi) & (x["ceiling"] >= ceil_hi), "bucket"] = "CORE"

    lev_hi = x["ceiling_leverage"].quantile(0.80)
    x.loc[x["ceiling_leverage"] >= lev_hi, "bucket"] = "LEVERAGE"

    x.loc[(x["salary"] <= 4400) & (x["ceil_per_1k"] >= x["ceil_per_1k"].quantile(0.70)), "bucket"] = "ATTACHMENT"

    lev_lo = x["ceiling_leverage"].quantile(0.25)
    x.loc[(x["own"] >= own_hi) & (x["ceiling_leverage"] <= lev_lo), "bucket"] = "FADE"

    return x

def main(csv_path: str, out_dir: str, pool_size: int, cpt_size: int):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    cheat = build_cheatsheet(df)

    core = cheat[cheat["bucket"].eq("CORE")].sort_values("core_score", ascending=False)
    lev  = cheat[cheat["bucket"].eq("LEVERAGE")].sort_values("core_score", ascending=False)
    att  = cheat[cheat["bucket"].eq("ATTACHMENT")].sort_values("attachment_score", ascending=False)

    pool = pd.concat([
        core.head(max(6, pool_size // 3)),
        lev.head(max(6, pool_size // 3)),
        att.head(max(4, pool_size // 4)),
    ]).drop_duplicates(subset=["player"])

    if len(pool) < pool_size:
        topup = cheat.sort_values("core_score", ascending=False)
        pool = pd.concat([pool, topup]).drop_duplicates(subset=["player"]).head(pool_size)
    else:
        pool = pool.head(pool_size)

    cpt_candidates = cheat.copy()
    cpt_candidates = cpt_candidates[cpt_candidates["salary"] >= 6000] if len(cpt_candidates) else cpt_candidates
    cpt_pool = cpt_candidates.sort_values("cpt_score", ascending=False).head(cpt_size)

    stem = csv_path.stem
    cheat_path = out_dir / f"{stem}_cheatsheet.csv"
    pool_path  = out_dir / f"{stem}_player_pool.csv"
    cpt_path   = out_dir / f"{stem}_captain_pool.csv"

    cheat.sort_values(["bucket", "core_score"], ascending=[True, False]).to_csv(cheat_path, index=False)
    pool.to_csv(pool_path, index=False)
    cpt_pool.to_csv(cpt_path, index=False)

    print("Wrote:")
    print(" ", cheat_path)
    print(" ", pool_path)
    print(" ", cpt_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Projections CSV (imported every slate)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--pool_size", type=int, default=16, help="Player pool size (jbc-style 12–18)")
    ap.add_argument("--cpt_size", type=int, default=6, help="Captain pool size (4–7)")
    args = ap.parse_args()
    main(args.csv, args.out, args.pool_size, args.cpt_size)
