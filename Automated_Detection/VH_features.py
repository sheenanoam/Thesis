import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction import FeatureHasher


# utils

def rect(b) -> Tuple[float, float, float, float]:
    """Normalize bounds to (x0, y0, x1, y1) for both [x0,y0,x1,y1] and [[x0,y0],[x1,y1]]."""
    if isinstance(b[0], list):
        (x0, y0), (x1, y1) = b
    else:
        x0, y0, x1, y1 = b
    return float(x0), float(y0), float(x1), float(y1)


def flatten_rico(node: Dict, acc: List[Dict]) -> None:
    """DFS-flatten a RICO tree into a MobileViews-like list of nodes."""
    d = {k: node.get(k) for k in (
        "class", "resource-id", "resource_id", "text", "content_description", "bounds"
    )}
    rid = d.pop("resource-id", None)
    if rid is not None and d.get("resource_id") is None:
        d["resource_id"] = rid
    acc.append(d)
    for ch in node.get("children", []) or []:
        flatten_rico(ch, acc)


# Heuristics for pop-ups / interstitials and tiny close icons
_AD_ACTIVITY = re.compile(r"(?:adactivity|reward(?:ed)?video|unityads)", re.I)
_AD_VIEW = re.compile(r"(interstitial|unityads|reward)", re.I)

def detect_small_close(v: Dict) -> bool:
    """A small close button (≤60 px in either dimension) by id/text/description."""
    b = v.get("bounds")
    if not b:
        return False
    x0, y0, x1, y1 = rect(b)
    w, h = abs(x1 - x0), abs(y1 - y0)
    if w > 60 or h > 60:
        return False
    rid = (v.get("resource_id") or "").lower()
    if rid in {"cbb", "close", "btn_close", "icon_smallclose"}:
        return True
    cd = (v.get("content_description") or "").lower()
    txt = (v.get("text") or "").lower()
    return ("close" in cd) or ("close" in txt)


def detect_popup_ad(state: Dict, views: List[Dict], W: float, H: float) -> int:
    """Large interstitial/reward view based on activity/view ids and area threshold."""
    if _AD_ACTIVITY.search(state.get("foreground_activity", "") or ""):
        return 1
    for v in views:
        cand = f"{v.get('resource_id','')} {v.get('class','')}"
        if "popup" in cand.lower() and not _AD_VIEW.search(cand):
            continue
        if _AD_VIEW.search(cand) and v.get("bounds"):
            x0, y0, x1, y1 = rect(v["bounds"])
            if abs(x1 - x0) * abs(y1 - y0) >= 0.30 * W * H:
                return 1
    return 0


def bucket_ratio(v: float, edges=(0.01, 0.025, 0.05, 0.10, 0.20, 0.40)) -> str:
    for e in edges:
        if v <= e:
            return f"<={e}"
    return f">{edges[-1]}"


def feature_dict(state: Dict) -> Dict[str, int]:
    """
    Build a sparse feature dict for a single screen:
      - widget class
      - area buckets and coarse x/y quadrants
      - tokenized text/content_description/resource_id
      - two heuristic flags (added as separate features)
    """
    if isinstance(state.get("views"), list):
        # MobileViews
        views = state["views"]
        W = int(state.get("width", 1080) or 1080)
        H = int(state.get("height", 1920) or 1920)
    else:
        # RICO tree
        views = []
        flatten_rico(state, views)
        try:
            x0, y0, x1, y1 = rect(state["bounds"])
            W = int(abs(x1 - x0)) or 1080
            H = int(abs(y1 - y0)) or 1920
        except Exception:
            W, H = 1080, 1920

    if W <= 0 or H <= 0:
        W, H = 1080, 1920

    feats: Dict[str, int] = {}
    tokens: List[str] = []
    small_flag = 0

    for v in views:
        b = v.get("bounds")
        if not b:
            continue

        if detect_small_close(v):
            small_flag = 1

        cls = v.get("class") or "NONE"
        feats[f"class={cls}"] = 1

        x0, y0, x1, y1 = rect(b)
        w, h = abs(x1 - x0), abs(y1 - y0)
        area_ratio = (w * h) / max(W * H, 1.0)

        feats[f"area_{bucket_ratio(area_ratio)}"] = 1
        xq = max(0, min(3, int(min(x0, x1) / W * 4)))
        yq = max(0, min(3, int(min(y0, y1) / H * 4)))
        feats[f"xquad_{xq}"] = 1
        feats[f"yquad_{yq}"] = 1

        for k in ("text", "content_description", "resource_id"):
            val = (v.get(k) or "").lower()
            tokens.extend(re.findall(r"[a-zA-Z]+", val))

    for tok in set(tokens):
        feats[f"tok={tok}"] = 1

    # include heuristics as features (also returned separately downstream)
    feats["small_close"] = small_flag
    feats["popup_ad"] = detect_popup_ad(state, views, W, H)
    return feats


# ----------------------------- discovery ---------------------------------- #

def iter_mobile_jsons(root: Path) -> Iterator[Tuple[str, Path]]:
    """
    Yield (screen_id, path) for MobileViews:
      <root>/<app>/states/state_<N>.json  →  screen_id = "<app>_<N>"
    """
    if not root.exists():
        return
    for app_dir in root.iterdir():
        states = app_dir / "states"
        if not states.is_dir():
            continue
        for p in states.glob("state_*.json"):
            num = p.stem.split("_", 1)[-1]
            screen = f"{app_dir.name}_{num}"
            yield screen, p


def iter_rico_jsons(root: Path) -> Iterator[Tuple[str, Path]]:
    """
    Yield (screen_id, path) for RICO:
      <root>/<screen>.json
    """
    if not root.exists():
        return
    for p in root.glob("*.json"):
        yield p.stem, p


def resolve_from_list(screens_df: pd.DataFrame, mobile_root: Path, rico_root: Path) -> List[Tuple[str, Path]]:
    """
    Given a list of screen ids (and optional 'dataset' column: 'mobile'|'rico'),
    resolve JSON paths from the provided roots.
    """
    jobs: List[Tuple[str, Path]] = []

    def mobile_path(screen: str) -> Path:
        if "_" not in screen:
            return Path()
        app, num = screen.rsplit("_", 1)
        return mobile_root / app / "states" / f"state_{num}.json"

    def rico_path(screen: str) -> Path:
        return rico_root / f"{screen}.json"

    for _, row in screens_df.iterrows():
        scr = str(row["screen"])
        ds = str(row["dataset"]).lower() if "dataset" in row and pd.notna(row["dataset"]) else ""
        cand = None
        if ds == "mobile":
            cand = mobile_path(scr)
        elif ds == "rico":
            cand = rico_path(scr)
        else:
            # try both
            mp, rp = mobile_path(scr), rico_path(scr)
            cand = mp if mp.exists() else rp
        if cand and cand.exists():
            jobs.append((scr, cand))
    return jobs


# -------------------------------- main ------------------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Build hashed view-hierarchy features.")
    ap.add_argument("--mobile-root", type=Path, default=Path("data/MobileViews/cleaned_dataset"),
                    help="MobileViews root directory.")
    ap.add_argument("--rico-root", type=Path, default=Path("data/rico/semantic_annotations"),
                    help="RICO semantic annotations directory.")
    ap.add_argument("--screens-csv", type=Path, default=None,
                    help="Optional CSV with columns: screen[,dataset]. If given, only these screens are processed.")
    ap.add_argument("--out", type=Path, default=Path("outputs/vh_features_50k.csv"),
                    help="Output file path (.csv or .parquet).")
    ap.add_argument("--hash-dim", type=int, default=50_000, help="FeatureHasher dimensionality.")
    args = ap.parse_args()

    # Collect (screen, path) pairs
    if args.screens_csv:
        if not args.screens_csv.exists():
            raise FileNotFoundError(args.screens_csv)
        screens_df = pd.read_csv(args.screens_csv)
        if "screen" not in screens_df.columns:
            raise ValueError("screens-csv must include a 'screen' column (and optional 'dataset').")
        jobs = resolve_from_list(screens_df, args.mobile_root, args.rico_root)
    else:
        # Autodiscovery
        jobs = list(iter_mobile_jsons(args.mobile_root)) + list(iter_rico_jsons(args.rico_root))

    if not jobs:
        raise SystemExit("No JSON files found. Check --mobile-root/--rico-root or provide --screens-csv.")

    # Feature extraction
    hasher = FeatureHasher(n_features=args.hash_dim, input_type="dict", alternate_sign=False)
    bags: List[Dict[str, int]] = []
    small_flags: List[int] = []
    popup_flags: List[int] = []
    screens: List[str] = []

    for scr, path in tqdm(jobs, desc="VH parse"):
        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
            fd = feature_dict(state)
            small_flags.append(int(fd.pop("small_close", 0)))
            popup_flags.append(int(fd.pop("popup_ad", 0)))
            bags.append(fd)
            screens.append(scr)
        except Exception as e:
            # keep alignment with empty dict and zeros
            bags.append({})
            small_flags.append(0)
            popup_flags.append(0)
            screens.append(scr)

    X = hasher.transform(bags).astype(np.float32).toarray()

    # Build output frame
    cols_json = [f"h{i}" for i in range(args.hash_dim)]
    out_df = pd.DataFrame(X, columns=cols_json)
    out_df.insert(0, "screen", screens)
    out_df["small_close"] = np.array(small_flags, dtype=np.float32)
    out_df["popup_ad"] = np.array(popup_flags, dtype=np.float32)

    # Sort by screen for determinism
    out_df = out_df.sort_values("screen").reset_index(drop=True)

    # Write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".parquet":
        # Requires pyarrow or fastparquet if installed
        out_df.to_parquet(args.out, index=False)
    else:
        out_df.to_csv(args.out, index=False)

    print(f"Wrote {len(out_df)} rows to {args.out} (D={args.hash_dim} + 2 extras)")


if __name__ == "__main__":
    main()
