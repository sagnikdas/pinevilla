import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path

# ─── Strategy Parameters (mirrors headshot_strategy.pine defaults) ────────────
RSI_PERIOD      = 12   # optimised: was 9
WMA_PERIOD      = 28   # optimised: was 21
EMA_PERIOD      = 3
TREND_EMA       = 200
VWAP_LOOKBACK   = 50   # optimised: was 100
RSI_BULL_THRESH = 58   # optimised: was 55
RSI_BEAR_THRESH = 45
SL_PERC         = 1.5  # optimised: was 2.0
TP_PERC         = 6.0
RISK_PERC       = 1.0
MAX_LOSS_USDT   = 30.0
MAX_CONSEC_LOSS = 3
COOLDOWN_BARS   = 10
INITIAL_CAPITAL = 10_000.0

DATA_DIR   = Path("ohlcv_data")
TIMEFRAMES = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","1w"]

# ─── Indicator Helpers ────────────────────────────────────────────────────────

def calc_rsi(series, period):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    # Wilder smoothing (matches Pine Script ta.rsi)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_wma(series, period):
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_vwap_normalized(df, lookback):
    hlc3     = (df["high"] + df["low"] + df["close"]) / 3
    # Daily-anchored VWAP (resets each UTC day)
    df2      = df.copy()
    df2["date"] = pd.to_datetime(df2["time"], unit="s", utc=True).dt.date
    cum_tp_vol = (hlc3 * df2["volume"]).groupby(df2["date"]).cumsum()
    cum_vol    = df2["volume"].groupby(df2["date"]).cumsum()
    vwap       = cum_tp_vol / cum_vol.replace(0, np.nan)

    price_high = df["close"].rolling(lookback).max()
    price_low  = df["close"].rolling(lookback).min()
    rng        = price_high - price_low
    vwap_norm  = np.where(rng > 0, (vwap - price_low) / rng * 100, 50)
    return pd.Series(vwap_norm, index=df.index)

# ─── Signal Generation ────────────────────────────────────────────────────────

def build_signals(df):
    df = df.copy()
    df["rsi"]       = calc_rsi(df["close"], RSI_PERIOD)
    df["wma_rsi"]   = calc_wma(df["rsi"],   WMA_PERIOD)
    df["ema_rsi"]   = calc_ema(df["rsi"],   EMA_PERIOD)
    df["trend_ema"] = calc_ema(df["close"], TREND_EMA)
    df["vwap_norm"] = calc_vwap_normalized(df, VWAP_LOOKBACK)

    prev_ema = df["ema_rsi"].shift(1)
    prev_wma = df["wma_rsi"].shift(1)

    cross_up   = (prev_ema <= prev_wma) & (df["ema_rsi"] > df["wma_rsi"])
    cross_down = (prev_ema >= prev_wma) & (df["ema_rsi"] < df["wma_rsi"])

    df["bull"] = (
        cross_up &
        (df["rsi"] > RSI_BULL_THRESH) &
        (df["vwap_norm"] > 50) &
        (df["close"] > df["trend_ema"])
    )
    df["bear"] = (
        cross_down &
        (df["rsi"] < RSI_BEAR_THRESH) &
        (df["vwap_norm"] < 50) &
        (df["close"] < df["trend_ema"])
    )
    return df

# ─── Backtester ───────────────────────────────────────────────────────────────

def run_backtest(df):
    equity         = INITIAL_CAPITAL
    position       = None   # dict with keys: side, entry, qty, sl, tp, entry_idx
    trades         = []
    equity_curve   = []
    consec_losses  = 0
    cooldown_until = -1

    for i, row in df.iterrows():
        bar_idx = df.index.get_loc(i)
        in_cooldown = bar_idx < cooldown_until

        # ── Manage open position ──
        if position is not None:
            closed  = False
            exit_px = None
            reason  = None

            if position["side"] == "long":
                if row["low"] <= position["sl"]:
                    exit_px, reason = position["sl"], "SL"
                    closed = True
                elif row["high"] >= position["tp"]:
                    exit_px, reason = position["tp"], "TP"
                    closed = True
                elif row["bear"]:
                    exit_px, reason = row["close"], "Signal"
                    closed = True
            else:  # short
                if row["high"] >= position["sl"]:
                    exit_px, reason = position["sl"], "SL"
                    closed = True
                elif row["low"] <= position["tp"]:
                    exit_px, reason = position["tp"], "TP"
                    closed = True
                elif row["bull"]:
                    exit_px, reason = row["close"], "Signal"
                    closed = True

            if closed:
                if position["side"] == "long":
                    pnl = (exit_px - position["entry"]) * position["qty"]
                else:
                    pnl = (position["entry"] - exit_px) * position["qty"]

                equity += pnl
                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time":  row["time"],
                    "side":       position["side"],
                    "entry":      position["entry"],
                    "exit":       exit_px,
                    "qty":        position["qty"],
                    "pnl":        round(pnl, 4),
                    "reason":     reason,
                    "equity":     round(equity, 4),
                })
                if pnl < 0:
                    consec_losses += 1
                    if consec_losses >= MAX_CONSEC_LOSS:
                        cooldown_until = bar_idx + COOLDOWN_BARS
                        consec_losses  = 0
                else:
                    consec_losses = 0
                position = None

        # ── Check for new entry ──
        if position is None and not in_cooldown:
            risk_amt = min(equity * RISK_PERC / 100, MAX_LOSS_USDT)
            qty      = risk_amt / (row["close"] * SL_PERC / 100)

            if row["bull"]:
                position = {
                    "side":       "long",
                    "entry":      row["close"],
                    "entry_time": row["time"],
                    "qty":        qty,
                    "sl":         row["close"] * (1 - SL_PERC / 100),
                    "tp":         row["close"] * (1 + TP_PERC / 100),
                }
            elif row["bear"]:
                position = {
                    "side":       "short",
                    "entry":      row["close"],
                    "entry_time": row["time"],
                    "qty":        qty,
                    "sl":         row["close"] * (1 + SL_PERC / 100),
                    "tp":         row["close"] * (1 - TP_PERC / 100),
                }

        equity_curve.append(equity)

    return pd.DataFrame(trades), pd.Series(equity_curve, index=df.index)

# ─── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(trades, equity_curve):
    if trades.empty:
        return {}

    longs  = trades[trades["side"] == "long"]
    shorts = trades[trades["side"] == "short"]

    def side_stats(t):
        if t.empty:
            return {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "win_rate": 0.0}
        wins = (t["pnl"] > 0).sum()
        return {
            "trades":   len(t),
            "wins":     int(wins),
            "losses":   int(len(t) - wins),
            "pnl":      round(t["pnl"].sum(), 2),
            "win_rate": round(wins / len(t) * 100, 1),
        }

    # Max drawdown
    peak = equity_curve.cummax()
    dd   = (equity_curve - peak) / peak * 100
    max_dd = round(dd.min(), 2)

    # Sharpe (annualised, assuming daily returns approximation)
    ret = equity_curve.pct_change().dropna()
    sharpe = round(ret.mean() / ret.std() * np.sqrt(252), 2) if ret.std() > 0 else 0.0

    total_pnl = round(trades["pnl"].sum(), 2)
    wins_total = (trades["pnl"] > 0).sum()

    return {
        "total_trades": len(trades),
        "win_rate":     round(wins_total / len(trades) * 100, 1),
        "total_pnl":    total_pnl,
        "max_drawdown": max_dd,
        "sharpe":       sharpe,
        "final_equity": round(equity_curve.iloc[-1], 2),
        "long":         side_stats(longs),
        "short":        side_stats(shorts),
    }

# ─── HTML Report ──────────────────────────────────────────────────────────────

def color(val, positive_good=True):
    if val > 0:
        return "#26a69a" if positive_good else "#ef5350"
    elif val < 0:
        return "#ef5350" if positive_good else "#26a69a"
    return "#aaaaaa"

def build_report(all_results):
    # ── Summary table ──
    rows = ""
    for tf, res in all_results.items():
        m = res["metrics"]
        if not m:
            rows += f"<tr><td>{tf}</td><td colspan='7' style='color:#888'>No trades</td></tr>"
            continue
        pnl_c  = color(m["total_pnl"])
        dd_c   = color(m["max_drawdown"], positive_good=False)
        eq_c   = color(m["final_equity"] - INITIAL_CAPITAL)
        rows += f"""
        <tr>
            <td><a href='#{tf}'>{tf}</a></td>
            <td>{m['total_trades']}</td>
            <td>{m['win_rate']}%</td>
            <td style='color:{pnl_c}'>{m['total_pnl']:+.2f}</td>
            <td style='color:{dd_c}'>{m['max_drawdown']:.2f}%</td>
            <td>{m['sharpe']}</td>
            <td style='color:{eq_c}'>{m['final_equity']:,.2f}</td>
        </tr>"""

    # ── Per-timeframe sections ──
    sections = ""
    for tf, res in all_results.items():
        m      = res["metrics"]
        trades = res["trades"]
        eq     = res["equity"]

        # Equity chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            subplot_titles=("Equity Curve", "Trade PnL"))

        fig.add_trace(go.Scatter(
            x=list(range(len(eq))), y=eq.values,
            mode="lines", name="Equity",
            line=dict(color="#26a69a", width=1.5)
        ), row=1, col=1)

        if not trades.empty:
            bar_colors = ["#26a69a" if p > 0 else "#ef5350" for p in trades["pnl"]]
            fig.add_trace(go.Bar(
                x=list(range(len(trades))), y=trades["pnl"],
                name="Trade PnL", marker_color=bar_colors
            ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark", height=480,
            margin=dict(l=40, r=20, t=40, b=20),
            showlegend=False,
            paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e"
        )
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)

        # Side-by-side Buy/Sell stats
        def side_card(label, s, col):
            if not s or s["trades"] == 0:
                return f"<div class='card'><h4 style='color:{col}'>{label}</h4><p>No trades</p></div>"
            wr_c  = "#26a69a" if s["win_rate"] >= 50 else "#ef5350"
            pnl_c = color(s["pnl"])
            return f"""
            <div class='card'>
                <h4 style='color:{col}'>{label}</h4>
                <table class='mini'>
                    <tr><td>Trades</td><td>{s['trades']}</td></tr>
                    <tr><td>W / L</td><td>{s['wins']} / {s['losses']}</td></tr>
                    <tr><td>Win Rate</td><td style='color:{wr_c}'>{s['win_rate']}%</td></tr>
                    <tr><td>PnL</td><td style='color:{pnl_c}'>{s['pnl']:+.2f} USDT</td></tr>
                </table>
            </div>"""

        if m:
            long_card  = side_card("Buy (Long)",  m.get("long"),  "#26a69a")
            short_card = side_card("Sell (Short)", m.get("short"), "#ef5350")
            pnl_c = color(m["total_pnl"])
            dd_c  = color(m["max_drawdown"], positive_good=False)
            summary_cards = f"""
            <div class='cards-row'>
                {long_card}
                {short_card}
                <div class='card'>
                    <h4>Overall</h4>
                    <table class='mini'>
                        <tr><td>Total Trades</td><td>{m['total_trades']}</td></tr>
                        <tr><td>Win Rate</td><td>{m['win_rate']}%</td></tr>
                        <tr><td>Total PnL</td><td style='color:{pnl_c}'>{m['total_pnl']:+.2f} USDT</td></tr>
                        <tr><td>Max Drawdown</td><td style='color:{dd_c}'>{m['max_drawdown']:.2f}%</td></tr>
                        <tr><td>Sharpe Ratio</td><td>{m['sharpe']}</td></tr>
                        <tr><td>Final Equity</td><td>{m['final_equity']:,.2f} USDT</td></tr>
                    </table>
                </div>
            </div>"""
        else:
            summary_cards = "<p style='color:#888'>No trades generated for this timeframe.</p>"

        sections += f"""
        <section id='{tf}'>
            <h2>{tf} Timeframe</h2>
            {summary_cards}
            {chart_html}
        </section>
        <hr>"""

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>HeadShot Strategy — Backtest Report</title>
<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body     {{ font-family: 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; padding: 24px; }}
  h1       {{ font-size: 1.6rem; margin-bottom: 4px; }}
  h2       {{ font-size: 1.2rem; color: #90caf9; margin: 24px 0 12px; }}
  p.sub    {{ color: #888; font-size: 0.85rem; margin-bottom: 24px; }}
  hr       {{ border: none; border-top: 1px solid #333; margin: 32px 0; }}
  a        {{ color: #90caf9; text-decoration: none; }}
  a:hover  {{ text-decoration: underline; }}

  /* Summary table */
  .summary-table           {{ width: 100%; border-collapse: collapse; margin-bottom: 32px; font-size: 0.9rem; }}
  .summary-table th        {{ background: #1e1e1e; color: #90caf9; padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
  .summary-table td        {{ padding: 7px 12px; border-bottom: 1px solid #222; }}
  .summary-table tr:hover  {{ background: #1a1a2e; }}

  /* Cards */
  .cards-row  {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }}
  .card       {{ background: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 16px; min-width: 200px; flex: 1; }}
  .card h4    {{ margin-bottom: 10px; font-size: 0.95rem; }}
  .mini       {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .mini td    {{ padding: 4px 0; }}
  .mini td:last-child {{ text-align: right; font-weight: 600; }}

  /* Nav */
  nav         {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 32px; }}
  nav a       {{ background: #1e1e1e; border: 1px solid #333; border-radius: 4px; padding: 4px 12px; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>HeadShot Strategy — Backtest Report</h1>
<p class='sub'>Initial Capital: $10,000 &nbsp;|&nbsp; Risk: 1% equity / trade (max 30 USDT) &nbsp;|&nbsp; SL: 2% &nbsp;|&nbsp; TP: 6% &nbsp;|&nbsp; All filters active</p>

<nav>{"".join(f"<a href='#{tf}'>{tf}</a>" for tf in all_results)}</nav>

<table class='summary-table'>
  <thead>
    <tr>
      <th>Timeframe</th><th>Trades</th><th>Win Rate</th>
      <th>Total PnL (USDT)</th><th>Max Drawdown</th><th>Sharpe</th><th>Final Equity</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

{sections}
</body>
</html>"""
    return html

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_results = {}
    for tf in TIMEFRAMES:
        path = DATA_DIR / f"{tf}.csv"
        if not path.exists():
            print(f"  skip {tf} (file not found)")
            continue

        print(f"  processing {tf} ({sum(1 for _ in open(path)) - 1} bars)...")
        df = pd.read_csv(path)
        df = df.sort_values("time").reset_index(drop=True)

        df = build_signals(df)
        trades, equity = run_backtest(df)
        metrics = calc_metrics(trades, equity)

        all_results[tf] = {"trades": trades, "equity": equity, "metrics": metrics}
        if metrics:
            print(f"    trades={metrics['total_trades']}  win={metrics['win_rate']}%  "
                  f"pnl={metrics['total_pnl']:+.2f}  dd={metrics['max_drawdown']:.2f}%")
        else:
            print(f"    no trades")

    report_path = Path("backtest_report.html")
    report_path.write_text(build_report(all_results), encoding="utf-8")
    print(f"\nReport saved → {report_path.resolve()}")

if __name__ == "__main__":
    main()
