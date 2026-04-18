# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a single-file TradingView indicator written in **Pine Script v6**. The script (`headshot.pine`) is deployed directly to TradingView — there is no build system, package manager, or local execution environment.

## Development Workflow

- Edit `headshot.pine` locally, then paste the contents into the TradingView Pine Script editor (Pine Editor tab) and click **Add to chart**.
- TradingView's built-in editor provides syntax checking and runtime errors in the chart pane.
- There is no linter, test runner, or CI pipeline.

## Architecture

The indicator runs in a separate oscillator pane (`overlay=false`). All signal logic flows through a layered gate pipeline evaluated on every bar.

### Core Calculations

| Variable | Formula | Notes |
|----------|---------|-------|
| `rsiVal` | `ta.rsi(close, rsiPeriod)` | — |
| `wmaOnRsi` | `ta.wma(rsiVal, wmaPeriod)` | Slow signal line |
| `emaOnRsi` | `ta.ema(rsiVal, emaPeriod)` | Fast signal line |
| `trendEma` | `ta.ema(close, trendEmaLength)` | Chart-timeframe trend filter |
| `vwapVal` | `ta.vwap(hlc3)` | Session VWAP; compared directly against `close` (not normalized) |

The raw signal trigger is `ta.crossover(emaOnRsi, wmaOnRsi)` (bull) / `ta.crossunder` (bear) combined with an RSI threshold gate (`rsiVal > rsiBull` / `rsiVal < rsiBear`).

### Signal Gate Pipeline

Each gate is independently togglable via inputs. A signal (`bullSignal` / `bearSignal`) fires only when **all enabled gates pass**:

1. **RSI threshold** — always on; `rsiBull`/`rsiBear` thresholds vary by timeframe in auto mode.
2. **VWAP confirmation** (`useVwap`) — long only when `close > vwapVal`; short only when `close < vwapVal`.
3. **Trend EMA filter** (`useTrendFilter`) — long only when `close > trendEma`; short only when below.
4. **Momentum confirmation** (`requireMomentumConfirm`) — requires rising RSI and rising `emaOnRsi` for longs (falling for shorts).
5. **HTF trend filter** (`htfTf`) — uses `request.security` with `lookahead_off` and `close[1]`/`ema[1]` (prior bar) to avoid repainting; disabled when `htfTf` is empty.
6. **RSI exhaustion filter** (`useExhaustFilter`) — blocks longs above `rsiCapLong`, shorts below `rsiFloorShort`.
7. **Cooldown** (`cooldownBars`) — enforces a minimum bar gap between any signal using a `var int lastSignalBar` counter.
8. **Warmup guard** (`isWarmedUp`) — suppresses signals until `bar_index >= max(rsiPeriod, wmaPeriod, emaPeriod, trendEmaLength) + 2`.
9. **Bar-close confirmation** (`confirmOnClose`) — gates on `barstate.isconfirmed` to reduce intrabar repaints.

### Auto-Timeframe Parameter Selection

When `autoParams=true`, `rsiPeriod`, `wmaPeriod`, `emaPeriod`, `rsiBull`, and `rsiBear` are selected from a hardcoded lookup table keyed on `timeframe.period`. The lookup was derived from a grid search over 19,008 parameter combinations. Manual inputs are ignored in this mode.

### Dashboard Table

Rendered once at `barstate.islast` using `table.new`. Displays all active parameters and a `lastConfSnapshot` string (e.g. `"3/4 · Long"`) showing how many confluence layers aligned on the last signal.

## Pine Script Conventions

- Version declaration must be the first line: `//@version=6`
- All user-facing settings use `input.*` functions grouped with `group=` for UI organization.
- `max_bars_back=500` is set on the `indicator()` call to support lookback calculations.
- Use `ta.*` namespace for all built-in technical analysis functions.
- HTF `request.security` calls must use `barmerge.lookahead_off` and index with `[1]` on both the source series and the derived indicator to remain non-repainting.
