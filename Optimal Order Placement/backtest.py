import pandas as pd
import numpy as np
import json
from datetime import datetime
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

## hold quote information from single venue
@dataclass(slots=True)
class VenueLevel:
    venue_id: int # id of venue
    px: float     # price offered
    sz: int       # size offered at price

## snapshot holds venue quotes at a particular timestamp
@dataclass(slots=True)
class Snapshot:
    ts: str # timestamp of the snapshot
    venues: List[VenueLevel] # list of VenueLevel objects

    @staticmethod
    def from_group(ts: str, group: pd.DataFrame) -> "Snapshot":
        return Snapshot(
            ts,
            [
                VenueLevel(int(v), float(p), int(s))
                for v, p, s in zip(
                    group.publisher_id.values,
                    group.ask_px_00.values,
                    group.ask_sz_00.values,
                )
            ],
        )

## load the CSV and parse into snapshots sorted by timestamp 
def load_snapshots(csv_path: str | Path) -> List[Snapshot]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = (
        pd.read_csv(csv_path)
        .sort_values("ts_event")
        .drop_duplicates(subset=["ts_event", "publisher_id"], keep="first")
        .reset_index(drop=True)
    )
    return [Snapshot.from_group(ts, grp) for ts, grp in df.groupby("ts_event", sort=True)]

## Generate n tuple (in multiples of chunk) summing to total
def enumerate_splits(total: int, n: int, *, chunk: int = 100) -> np.ndarray:
    """All non-negative n-tuples (multiples of *chunk*) summing to *total*."""
    levels = total // chunk + 1
    grids = [np.arange(levels) * chunk for _ in range(n)]
    mesh = np.array(np.meshgrid(*grids, indexing="ij"))
    combos = mesh.reshape(n, -1).T
    return combos[combos.sum(axis=1) == total]

## generate n-tiples for each split allocaiton across venues
def compute_cost_matrix(
    splits: np.ndarray,
    venues: List[VenueLevel],
    lambda_over: float,
    lambda_under: float,
    theta_queue: float,
) -> np.ndarray:
    if len(splits) == 0:
        return np.empty(0)

    px = np.fromiter((v.px for v in venues), float)
    q = np.fromiter((v.sz for v in venues), float)

    x = splits
    exec_pay = (x * px).sum(axis=1)
    over_pen = lambda_over * np.maximum(0, x - q).sum(axis=1)
    under_pen = lambda_under * np.maximum(0, q - x).sum(axis=1)
    queue_pen = theta_queue * (1 / np.where(q > 0, q, 1)).sum()
    return exec_pay + over_pen + under_pen + queue_pen

## main allocator: determine how much to send to each venue for a snapshot
def allocate(
    remaining: int,
    venues: List[VenueLevel],
    lambda_over: float,
    lambda_under: float,
    theta_queue: float,
    *,
    chunk: int = 100,
) -> Dict[int, int]:
    """Return {venue_id: shares_to_send} for the current snapshot."""
    if remaining == 0 or len(venues) == 0:
        return {}

    splits = enumerate_splits(remaining, len(venues), chunk=chunk)

    # retry with finest granularity if coarse grid produced nothing
    if len(splits) == 0 and chunk != 1:
        splits = enumerate_splits(remaining, len(venues), chunk=1)

    # ultimate fall-back: hit the cheapest venue with everything
    if len(splits) == 0:
        best = min(venues, key=lambda v: v.px)
        return {best.venue_id: remaining}

    costs = compute_cost_matrix(splits, venues, lambda_over, lambda_under, theta_queue)
    best_split = splits[costs.argmin()]
    return {v.venue_id: int(x) for v, x in zip(venues, best_split)}

# run the backtest with given parameters across all snapshots
def run_backtest(
    snapshots: List[Snapshot],
    lambda_over: float,
    lambda_under: float,
    theta_queue: float,
    *,
    order_size: int = 5_000,
    side: str = "buy",
    chunk: int = 100,
) -> Tuple[float, float, List[Tuple[str, float]]]:
    sign = 1 if side == "buy" else -1
    remaining, cash, filled = order_size, 0.0, 0
    cumulative: List[Tuple[str, float]] = []

    for snap in snapshots:
        if remaining <= 0:
            break
        alloc = allocate(
            remaining, snap.venues,
            lambda_over, lambda_under, theta_queue,
            chunk=chunk,
        )
        for v in snap.venues:
            send = alloc.get(v.venue_id, 0)
            fill = min(send, v.sz)
            cash += sign * fill * v.px
            filled += fill
        remaining = order_size - filled
        cumulative.append((snap.ts, abs(cash)))
    avg = abs(cash) / filled if filled else float("inf")
    return abs(cash), avg, cumulative

# baseline strategy: always hit the best ask (lowest price)
def best_ask_baseline(snaps: List[Snapshot], *, size=5_000):
    remain, cash = size, 0.0
    for s in snaps:
        if remain <= 0:
            break
        v = min(s.venues, key=lambda v: v.px)
        take = min(remain, v.sz)
        cash += take * v.px
        remain -= take
    return cash, cash / size, None

# vwap
def vwap_baseline(snaps: List[Snapshot], *, size=5_000):
    num = sum(v.px * v.sz for s in snaps for v in s.venues)
    den = sum(v.sz for s in snaps for v in s.venues)
    vwap = num / den
    return vwap * size, vwap, None

# twap
def twap_baseline(snaps: List[Snapshot], *, size=5_000, bucket=60):
    times = [datetime.fromisoformat(s.ts[:-1]) for s in snaps]
    start = times[0]
    buckets: Dict[int, List[Snapshot]] = {}
    for t, s in zip(times, snaps):
        idx = math.floor((t - start).total_seconds() / bucket)
        buckets.setdefault(idx, []).append(s)
    target = size / len(buckets)
    cash = filled = 0.0
    for grp in buckets.values():
        pxs = [v.px for s in grp for v in s.venues]
        cash += min(target, size - filled) * (sum(pxs) / len(pxs))
        filled += target
    return cash, cash / size, None

# draws a random float uniformly
def _logu(rng, lo, hi):
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi))

# search for the best parameter using randomized trials
def search_parameters(snaps: List[Snapshot], *, trials=120, seed=42):
    rng = np.random.default_rng(seed)
    best = {"cost": float("inf")}
    for _ in range(trials):
        lo = _logu(rng, 1e-3, 10)
        lu = _logu(rng, 1e-3, 10)
        th = _logu(rng, 1e-2, 1e2)
        cost, avg, _ = run_backtest(snaps, lo, lu, th)
        if cost < best["cost"]:
            best = {
                "cost": cost,
                "avg": avg,
                "lambda_over": lo,
                "lambda_under": lu,
                "theta_queue": th,
            }
    return best

# Calculate basis points improvement
def _bps(router_cost, base_cost):
    return (base_cost - router_cost) / base_cost * 1e4

# Full evaluation of router vs. baselines
def run_full_backtest(csv: str | Path = "l1_day.csv", *, trials: int = 120):
    """High-level helper: returns the full result dict (no printing)."""
    snaps = load_snapshots(csv)
    best = search_parameters(snaps, trials=trials)

    rc, ra, _ = run_backtest(
        snaps, best["lambda_over"], best["lambda_under"], best["theta_queue"]
    )
    bc, ba, _ = best_ask_baseline(snaps)
    tc, ta, _ = twap_baseline(snaps)
    vc, va, _ = vwap_baseline(snaps)

    return {
        "best_params": {
            k: best[k] for k in ("lambda_over", "lambda_under", "theta_queue")
        },
        "router": {"total_spent": rc, "avg_price": ra},
        "best_ask": {
            "total_spent": bc,
            "avg_price": ba,
            "savings_bps": _bps(rc, bc),
        },
        "twap": {
            "total_spent": tc,
            "avg_price": ta,
            "savings_bps": _bps(rc, tc),
        },
        "vwap": {
            "total_spent": vc,
            "avg_price": va,
            "savings_bps": _bps(rc, vc),
        },
    }

if __name__ == "__main__":
    res = run_full_backtest()
    print(json.dumps(res, indent=2))
