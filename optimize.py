"""
Parameter grid-search optimizer for the HeadShot strategy.
Sweeps indicator & risk params, scores by composite metric, outputs HTML report.
"""
import pandas as pd
import numpy as np
import itertools
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

DATA_DIR        = Path("ohlcv_data")
INITIAL_CAPITAL = 10_000.0
TOP_N           = 20          # top combos to show per timeframe
TARGET_TFS      = ["15m", "30m", "1h", "4h", "1d"]  # skip 1m/3m/5m for speed

# ─── Parameter grid ───────────────────────────────────────────────────────────
GRID = {
    "rsi_period":  [6,  9, 12, 14],
    "wma_period":  [14, 21, 28],
    "ema_period":  [2,  3,  5],
    "rsi_bull":    [52, 55, 58, 61],
    "rsi_bear":    [39, 42, 45, 48],
    "sl_perc":     [1.5, 2.0, 2.5],
    "tp_perc":     [4.5, 6.0, 7.5, 9.0],
    "vwap_lb":     [50, 100, 200],
}

# ─── Indicator helpers ────────────────────────────────────────────────────────

def calc_rsi(series, period):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_wma(series, period):
    w = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_vwap_norm(df, lookback):
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    dates      = pd.to_datetime(df["time"], unit="s", utc=True).dt.date
    cum_tp_vol = (hlc3 * df["volume"]).groupby(dates).cumsum()
    cum_vol    = df["volume"].groupby(dates).cumsum()
    vwap       = cum_tp_vol / cum_vol.replace(0, np.nan)
    ph = df["close"].rolling(lookback).max()
    pl = df["close"].rolling(lookback).min()
    rng = ph - pl
    return np.where(rng > 0, (vwap - pl) / rng * 100, 50)

# ─── Vectorised signal builder ────────────────────────────────────────────────

def build_signals(df, p):
    rsi      = calc_rsi(df["close"], p["rsi_period"])
    wma_rsi  = calc_wma(rsi, p["wma_period"])
    ema_rsi  = calc_ema(rsi, p["ema_period"])
    trend    = calc_ema(df["close"], 200)
    vwap     = calc_vwap_norm(df, p["vwap_lb"])

    prev_ema = ema_rsi.shift(1)
    prev_wma = wma_rsi.shift(1)
    cross_up   = (prev_ema <= prev_wma) & (ema_rsi > wma_rsi)
    cross_down = (prev_ema >= prev_wma) & (ema_rsi < wma_rsi)

    bull = cross_up  & (rsi > p["rsi_bull"]) & (pd.Series(vwap, index=df.index) > 50) & (df["close"] > trend)
    bear = cross_down & (rsi < p["rsi_bear"]) & (pd.Series(vwap, index=df.index) < 50) & (df["close"] < trend)
    return bull.values, bear.values, df["close"].values, df["high"].values, df["low"].values

# ─── Backtest (numba-free but tight Python loop) ──────────────────────────────

def run_backtest(bull, bear, close, high, low, p):
    sl_f = p["sl_perc"] / 100
    tp_f = p["tp_perc"] / 100
    equity  = INITIAL_CAPITAL
    pos     = None
    wins    = losses = 0
    equity_arr = np.empty(len(close))
    max_consec_loss = 3
    cooldown_bars   = 10
    consec = 0
    cooldown_until = -1

    for i in range(len(close)):
        in_cd = i < cooldown_until

        if pos is not None:
            closed = False
            pnl    = 0.0
            if pos[0] == 1:  # long
                if low[i] <= pos[2]:
                    pnl, closed = (pos[2] - pos[1]) * pos[3], True
                elif high[i] >= pos[4]:
                    pnl, closed = (pos[4] - pos[1]) * pos[3], True
                elif bear[i]:
                    pnl, closed = (close[i] - pos[1]) * pos[3], True
            else:             # short
                if high[i] >= pos[2]:
                    pnl, closed = (pos[1] - pos[2]) * pos[3], True
                elif low[i] <= pos[4]:
                    pnl, closed = (pos[1] - pos[4]) * pos[3], True
                elif bull[i]:
                    pnl, closed = (pos[1] - close[i]) * pos[3], True

            if closed:
                equity += pnl
                if pnl >= 0:
                    wins += 1; consec = 0
                else:
                    losses += 1; consec += 1
                    if consec >= max_consec_loss:
                        cooldown_until = i + cooldown_bars; consec = 0
                pos = None

        if pos is None and not in_cd:
            risk_amt = min(equity * 1.0 / 100, 30.0)
            qty = risk_amt / (close[i] * sl_f) if close[i] * sl_f > 0 else 0
            if bull[i]:
                pos = (1, close[i], close[i]*(1-sl_f), qty, close[i]*(1+tp_f))
            elif bear[i]:
                pos = (-1, close[i], close[i]*(1+sl_f), qty, close[i]*(1-tp_f))

        equity_arr[i] = equity

    total = wins + losses
    if total == 0:
        return None

    peak   = np.maximum.accumulate(equity_arr)
    dd_arr = (equity_arr - peak) / peak * 100
    max_dd = dd_arr.min()
    ret    = np.diff(equity_arr) / equity_arr[:-1]
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0

    gross_profit = sum(
        (pos[4] - pos[1]) * pos[3]
        for pos in []  # placeholder — use equity delta
    )
    final_eq  = equity_arr[-1]
    total_pnl = final_eq - INITIAL_CAPITAL
    win_rate  = wins / total

    # Composite score: penalise deep drawdown, reward Sharpe and win rate
    # Also require minimum trades (avoid curve-fit combos with 2 trades)
    if total < 10:
        score = -999.0
    else:
        profit_factor = (win_rate * p["tp_perc"]) / ((1 - win_rate) * p["sl_perc"] + 1e-9)
        score = sharpe * profit_factor * (1 + max_dd / 100)  # max_dd is negative

    return {
        "score":      round(score, 4),
        "trades":     total,
        "win_rate":   round(win_rate * 100, 1),
        "total_pnl":  round(total_pnl, 2),
        "max_dd":     round(max_dd, 2),
        "sharpe":     round(sharpe, 4),
        "final_eq":   round(final_eq, 2),
    }

# ─── Worker: evaluate one parameter dict on one DataFrame ─────────────────────

def _worker(args):
    df, params = args
    bull, bear, close, high, low = build_signals(df, params)
    result = run_backtest(bull, bear, close, high, low, params)
    return params, result

# ─── Per-timeframe optimisation ───────────────────────────────────────────────

def optimise_tf(tf, df):
    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    param_list = [{keys[i]: v for i, v in enumerate(c)} for c in combos]

    # Filter obviously bad combos (wma must be > rsi period to be meaningful)
    param_list = [p for p in param_list if p["wma_period"] > p["rsi_period"]]

    args = [(df, p) for p in param_list]
    print(f"  [{tf}] {len(param_list)} combos...")

    results = []
    with Pool(max(1, cpu_count() - 1)) as pool:
        for params, res in pool.imap_unordered(_worker, args, chunksize=32):
            if res is not None:
                results.append({**params, **res})

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results).sort_values("score", ascending=False)
    return df_res.head(TOP_N).reset_index(drop=True)

# ─── HTML report ──────────────────────────────────────────────────────────────

def color(val, positive_good=True):
    if val > 0:
        return "#26a69a" if positive_good else "#ef5350"
    elif val < 0:
        return "#ef5350" if positive_good else "#26a69a"
    return "#aaaaaa"

def table_html(df_top):
    if df_top.empty:
        return "<p style='color:#888'>No results.</p>"
    cols_show = ["score","trades","win_rate","total_pnl","max_dd","sharpe","final_eq",
                 "rsi_period","wma_period","ema_period","rsi_bull","rsi_bear",
                 "sl_perc","tp_perc","vwap_lb"]
    rows = ""
    for _, row in df_top.iterrows():
        pnl_c = color(row["total_pnl"])
        dd_c  = color(row["max_dd"], positive_good=False)
        wr_c  = "#26a69a" if row["win_rate"] >= 50 else "#ef5350"
        rows += f"""<tr>
            <td>{row['score']:.3f}</td>
            <td>{int(row['trades'])}</td>
            <td style='color:{wr_c}'>{row['win_rate']}%</td>
            <td style='color:{pnl_c}'>{row['total_pnl']:+.2f}</td>
            <td style='color:{dd_c}'>{row['max_dd']:.2f}%</td>
            <td>{row['sharpe']:.3f}</td>
            <td>{row['final_eq']:,.2f}</td>
            <td>{int(row['rsi_period'])}</td>
            <td>{int(row['wma_period'])}</td>
            <td>{int(row['ema_period'])}</td>
            <td>{int(row['rsi_bull'])}</td>
            <td>{int(row['rsi_bear'])}</td>
            <td>{row['sl_perc']}</td>
            <td>{row['tp_perc']}</td>
            <td>{int(row['vwap_lb'])}</td>
        </tr>"""
    header = "".join(f"<th>{c}</th>" for c in cols_show)
    return f"<table class='opt-table'><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"

def build_html(all_top):
    nav = "".join(f"<a href='#{tf}'>{tf}</a>" for tf in all_top)
    sections = ""
    for tf, df_top in all_top.items():
        best = df_top.iloc[0] if not df_top.empty else None
        badge = ""
        if best is not None:
            badge = f"""
            <div class='best-badge'>
              Best: RSI={int(best.rsi_period)}, WMA={int(best.wma_period)}, EMA={int(best.ema_period)},
              Bull={int(best.rsi_bull)}, Bear={int(best.rsi_bear)},
              SL={best.sl_perc}%, TP={best.tp_perc}%, VWAP_LB={int(best.vwap_lb)}
              &nbsp;→&nbsp; PnL <b>{best.total_pnl:+.2f}</b> USDT &nbsp;|&nbsp;
              WR {best.win_rate}% &nbsp;|&nbsp; DD {best.max_dd:.2f}% &nbsp;|&nbsp;
              Sharpe {best.sharpe:.3f}
            </div>"""
        sections += f"""
        <section id='{tf}'>
            <h2>{tf} — Top {TOP_N} Parameter Combinations</h2>
            {badge}
            {table_html(df_top)}
        </section><hr>"""

    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<title>HeadShot — Optimization Report</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body        {{ font-family: 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; padding: 24px; }}
h1          {{ font-size: 1.5rem; margin-bottom: 6px; }}
h2          {{ font-size: 1.1rem; color: #90caf9; margin: 24px 0 10px; }}
p.sub       {{ color: #888; font-size: 0.85rem; margin-bottom: 20px; }}
hr          {{ border: none; border-top: 1px solid #333; margin: 28px 0; }}
a           {{ color: #90caf9; text-decoration: none; }}
nav         {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }}
nav a       {{ background: #1e1e1e; border: 1px solid #333; border-radius: 4px; padding: 4px 12px; font-size: 0.85rem; }}
.best-badge {{ background: #1e1e1e; border-left: 3px solid #26a69a; padding: 10px 14px;
               font-size: 0.88rem; margin-bottom: 12px; border-radius: 4px; }}
.opt-table  {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; overflow-x: auto; display: block; }}
.opt-table th {{ background: #1e1e1e; color: #90caf9; padding: 6px 10px; text-align: left;
                  border-bottom: 1px solid #333; white-space: nowrap; }}
.opt-table td {{ padding: 5px 10px; border-bottom: 1px solid #1e1e1e; white-space: nowrap; }}
.opt-table tr:hover {{ background: #1a1a2e; }}
.opt-table tr:first-child td {{ background: #1a2a1a; }}
</style>
</head>
<body>
<h1>HeadShot Strategy — Parameter Optimization Report</h1>
<p class='sub'>Grid search over {sum(len(v) for v in GRID.values())} param axes &nbsp;|&nbsp;
Score = Sharpe × Profit Factor × (1 + MaxDD/100) &nbsp;|&nbsp;
Min 10 trades filter &nbsp;|&nbsp; Capital $10,000</p>
<nav>{nav}</nav>
{sections}
</body>
</html>"""

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_top = {}
    t0 = time.time()
    for tf in TARGET_TFS:
        path = DATA_DIR / f"{tf}.csv"
        if not path.exists():
            print(f"  skip {tf}")
            continue
        df = pd.read_csv(path).sort_values("time").reset_index(drop=True)
        top = optimise_tf(tf, df)
        all_top[tf] = top
        if not top.empty:
            best = top.iloc[0]
            print(f"  [{tf}] best score={best['score']:.3f}  pnl={best['total_pnl']:+.2f}"
                  f"  wr={best['win_rate']}%  dd={best['max_dd']:.2f}%"
                  f"  → RSI={int(best['rsi_period'])} WMA={int(best['wma_period'])}"
                  f" EMA={int(best['ema_period'])} bull={int(best['rsi_bull'])}"
                  f" bear={int(best['rsi_bear'])} sl={best['sl_perc']} tp={best['tp_perc']}"
                  f" vwap={int(best['vwap_lb'])}")

    out = Path("optimization_report.html")
    out.write_text(build_html(all_top), encoding="utf-8")
    print(f"\nDone in {time.time()-t0:.1f}s → {out.resolve()}")

    # Also dump best params per TF as JSON for easy copy-paste
    best_params = {}
    for tf, top in all_top.items():
        if not top.empty:
            b = top.iloc[0]
            best_params[tf] = {k: (int(b[k]) if k in ("rsi_period","wma_period","ema_period","rsi_bull","rsi_bear","vwap_lb") else float(b[k]))
                                for k in ("rsi_period","wma_period","ema_period","rsi_bull","rsi_bear","sl_perc","tp_perc","vwap_lb")}
    Path("best_params.json").write_text(json.dumps(best_params, indent=2))
    print("Best params → best_params.json")

if __name__ == "__main__":
    main()
