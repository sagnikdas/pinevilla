# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a single-file TradingView indicator written in **Pine Script v5**. The script (`headshot.pine`) is deployed directly to TradingView — there is no build system, package manager, or local execution environment.

## Development Workflow

- Edit `headshot.pine` locally, then paste the contents into the TradingView Pine Script editor (Pine Editor tab) and click **Add to chart**.
- TradingView's built-in editor provides syntax checking and runtime errors in the chart pane.
- There is no linter, test runner, or CI pipeline.

## Architecture

The indicator runs in a separate oscillator pane (`overlay=false`) and combines three signals:

| Signal | Calculation | Default Period |
|--------|-------------|----------------|
| RSI | `ta.rsi(close, rsiPeriod)` | 9 |
| WMA on RSI | `ta.wma(rsiVal, wmaPeriod)` | 21 |
| EMA on RSI | `ta.ema(rsiVal, emaPeriod)` | 3 |
| VWAP (normalized) | `ta.vwap(hlc3)` rescaled to 0–100 using 100-bar high/low of `close` | — |

VWAP normalization formula: `(vwapVal - priceLow) / (priceHigh - priceLow) * 100`, clamped to 50 when the range is flat.

## Pine Script Conventions

- Version declaration must be the first line: `//@version=5`
- All user-facing settings use `input.*` functions grouped with `group=` for UI organization.
- `max_bars_back=500` is set on the `indicator()` call to support lookback calculations.
- Use `ta.*` namespace for all built-in technical analysis functions.
