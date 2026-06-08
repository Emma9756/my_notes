#!/usr/bin/env python3
"""Local portfolio allocation, risk, and rule-based action helper."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - local environment has PyYAML
    yaml = None


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text == "":
        return default
    return float(text)


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "是"}


def read_csv(path: Path | str | None) -> list[dict[str, str]]:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_config(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("YAML config requires PyYAML; use JSON or install pyyaml.")
    data = yaml.safe_load(text)
    return data or {}


def normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(config or {})
    config.setdefault("base_currency", "CNY")
    config.setdefault("fx_rates", {"CNY": 1.0})
    config.setdefault("targets", {})
    config.setdefault("rules", {})
    config["fx_rates"] = {k: as_float(v, 1.0) for k, v in config.get("fx_rates", {}).items()}
    config["fx_rates"].setdefault(config["base_currency"], 1.0)
    return config


def load_price_history(prices_path: Path | str | None) -> dict[str, list[dict[str, Any]]]:
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(prices_path):
        symbol = row.get("symbol", "").strip()
        if not symbol:
            continue
        history[symbol].append(
            {
                "date": row.get("date", "").strip(),
                "close": as_float(row.get("close")),
                "currency": row.get("currency", "").strip() or "CNY",
            }
        )
    for rows in history.values():
        rows.sort(key=lambda item: item["date"])
    return dict(history)


def normalize_date(value: str) -> str:
    text = str(value).strip()
    if "-" in text:
        return text
    if len(text) == 8 and text.isdigit():
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    return text


def compact_date(value: str) -> str:
    return normalize_date(value).replace("-", "")


def load_instruments(path: Path | str) -> list[dict[str, Any]]:
    instruments = []
    for row in read_csv(path):
        if not as_bool(row.get("enabled", "1"), True):
            continue
        symbol = row.get("symbol", "").strip()
        if not symbol:
            continue
        instruments.append(
            {
                "symbol": symbol,
                "name": row.get("name", "").strip() or symbol,
                "source": row.get("source", "").strip() or "akshare",
                "kind": row.get("kind", "").strip() or "etf",
                "currency": row.get("currency", "").strip() or "CNY",
            }
        )
    return instruments


class AksharePriceProvider:
    def __init__(self):
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError(
                "update-prices requires akshare. Install it with: "
                "python3 -m pip install akshare"
            ) from exc
        self.ak = ak

    def fetch_history(
        self,
        instrument: dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        symbol = instrument["symbol"]
        kind = instrument["kind"]
        currency = instrument.get("currency") or "CNY"
        start = compact_date(start_date)
        end = compact_date(end_date)

        if kind in {"etf", "lof"}:
            df = self.ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="")
            date_col, close_col = "日期", "收盘"
        elif kind == "a_stock":
            df = self.ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="")
            date_col, close_col = "日期", "收盘"
        elif kind == "hk_stock":
            df = self.ak.stock_hk_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="")
            date_col, close_col = "日期", "收盘"
        else:
            raise ValueError(f"Unsupported instrument kind: {kind}")

        rows = []
        for _, item in df.iterrows():
            rows.append(
                {
                    "date": normalize_date(str(item[date_col])),
                    "symbol": symbol,
                    "close": as_float(item[close_col]),
                    "currency": currency,
                }
            )
        return rows


def merge_price_rows(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    symbol_order: dict[str, int] = {}
    for row in existing_rows + new_rows:
        symbol = str(row.get("symbol", "")).strip()
        date = normalize_date(str(row.get("date", "")))
        if not symbol or not date:
            continue
        if symbol not in symbol_order:
            symbol_order[symbol] = len(symbol_order)
        merged[(date, symbol)] = {
            "date": date,
            "symbol": symbol,
            "close": str(row.get("close", "")).strip(),
            "currency": str(row.get("currency", "")).strip() or "CNY",
        }
    return [merged[key] for key in sorted(merged.keys(), key=lambda item: (symbol_order[item[1]], item[0]))]


def write_price_rows(path: Path | str, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["date", "symbol", "close", "currency"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def update_prices(
    instruments_path: Path | str,
    prices_path: Path | str,
    start_date: str,
    end_date: str,
    provider: Any | None = None,
) -> Path:
    provider = provider or AksharePriceProvider()
    instruments = load_instruments(instruments_path)
    existing_rows = read_csv(prices_path) if Path(prices_path).exists() else []
    fetched_rows: list[dict[str, Any]] = []
    errors = []

    for instrument in instruments:
        try:
            rows = provider.fetch_history(instrument, start_date, end_date)
        except Exception as exc:  # keep other symbols updateable
            errors.append(f"{instrument['symbol']} {instrument['name']}: {exc}")
            continue
        fetched_rows.extend(rows)

    merged_rows = merge_price_rows(existing_rows, fetched_rows)
    written = write_price_rows(prices_path, merged_rows)
    if errors:
        message = "\n".join(errors)
        raise RuntimeError(f"Some instruments failed while updating prices:\n{message}")
    return written


def latest_price_map(history: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    result = {}
    for symbol, rows in history.items():
        if rows:
            result[symbol] = rows[-1]["close"]
    return result


def load_positions(
    holdings_path: Path | str,
    history: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    fx = config["fx_rates"]
    latest = latest_price_map(history)
    positions: list[dict[str, Any]] = []
    for row in read_csv(holdings_path):
        symbol = row.get("symbol", "").strip()
        currency = row.get("currency", "").strip() or config["base_currency"]
        fx_rate = as_float(fx.get(currency), 1.0)
        quantity = as_float(row.get("quantity"))
        price = as_float(row.get("price"), latest.get(symbol, 0.0))
        market_value = as_float(row.get("market_value"), quantity * price)
        cost_price = as_float(row.get("cost_price"))
        cost_amount = as_float(row.get("cost_amount"), quantity * cost_price)
        value_base = market_value * fx_rate
        cost_base = cost_amount * fx_rate
        pnl_base = value_base - cost_base if cost_base else 0.0
        pnl_pct = pnl_base / cost_base if cost_base else 0.0
        positions.append(
            {
                "account": row.get("account", "").strip(),
                "platform": row.get("platform", "").strip(),
                "symbol": symbol,
                "name": row.get("name", "").strip() or symbol,
                "market": row.get("market", "").strip(),
                "asset_class": row.get("asset_class", "").strip() or "未分类",
                "security_type": row.get("security_type", "").strip(),
                "currency": currency,
                "quantity": quantity,
                "price": price,
                "market_value": market_value,
                "cost_amount": cost_amount,
                "value_base": value_base,
                "cost_base": cost_base,
                "pnl_base": pnl_base,
                "pnl_pct": pnl_pct,
                "as_of": row.get("as_of", "").strip(),
            }
        )
    total = sum(item["value_base"] for item in positions)
    for item in positions:
        item["weight"] = item["value_base"] / total if total else 0.0
    return positions


def group_positions(positions: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    total = sum(item["value_base"] for item in positions)
    for item in positions:
        key = item.get(field) or "未分类"
        bucket = grouped.setdefault(key, {"value": 0.0, "cost": 0.0, "weight": 0.0, "count": 0})
        bucket["value"] += item["value_base"]
        bucket["cost"] += item["cost_base"]
        bucket["count"] += 1
    for bucket in grouped.values():
        bucket["weight"] = bucket["value"] / total if total else 0.0
        bucket["pnl"] = bucket["value"] - bucket["cost"] if bucket["cost"] else 0.0
        bucket["pnl_pct"] = bucket["pnl"] / bucket["cost"] if bucket["cost"] else 0.0
    return dict(sorted(grouped.items()))


def max_drawdown(values: list[float]) -> float:
    peak = None
    worst = 0.0
    for value in values:
        if value <= 0:
            continue
        peak = value if peak is None else max(peak, value)
        if peak:
            worst = min(worst, value / peak - 1.0)
    return worst


MIN_ANNUALIZATION_POINTS = 30


def annualized_return(values: list[float], periods_per_year: int = 252) -> float | None:
    if len(values) < MIN_ANNUALIZATION_POINTS:
        return None
    if len(values) < 2 or values[0] <= 0:
        return 0.0
    total = values[-1] / values[0] - 1.0
    years = (len(values) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return (1.0 + total) ** (1.0 / years) - 1.0


def annualized_volatility(values: list[float], periods_per_year: int = 252) -> float | None:
    if len(values) < MIN_ANNUALIZATION_POINTS:
        return None
    returns = []
    for prev, curr in zip(values, values[1:]):
        if prev > 0:
            returns.append(curr / prev - 1.0)
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns) * math.sqrt(periods_per_year)


def symbol_metrics(history: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    metrics = {}
    for symbol, rows in history.items():
        values = [item["close"] for item in rows if item["close"] > 0]
        if len(values) < 2:
            continue
        metrics[symbol] = {
            "total_return": values[-1] / values[0] - 1.0,
            "annualized_return": annualized_return(values),
            "annualized_volatility": annualized_volatility(values),
            "max_drawdown": max_drawdown(values),
            "current_drawdown": values[-1] / max(values) - 1.0 if values else 0.0,
        }
    return metrics


def target_map(config: dict[str, Any], name: str) -> dict[str, float]:
    raw = config.get("targets", {}).get(name, {}) or {}
    return {str(k): as_float(v) for k, v in raw.items()}


def grouped_current_weights(
    positions: list[dict[str, Any]],
    group_field: str,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = defaultdict(lambda: {"value": 0.0, "weight": 0.0})
    total = sum(item["value_base"] for item in positions)
    for item in positions:
        key = item.get(group_field) or "未分类"
        result[key]["value"] += item["value_base"]
    for value in result.values():
        value["weight"] = value["value"] / total if total else 0.0
    return dict(result)


def rebalance_recommendations(
    positions: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rules = config.get("rules", {}).get("rebalance", {}) or {}
    threshold = as_float(rules.get("drift_threshold"), 0.03)
    min_trade = as_float(rules.get("min_trade_amount"), 0.0)
    total = sum(item["value_base"] for item in positions)
    recommendations: list[dict[str, Any]] = []

    dimensions = [
        ("symbol", target_map(config, "symbols")),
        ("asset_class", target_map(config, "asset_classes")),
    ]
    for dimension, targets in dimensions:
        if not targets:
            continue
        current = grouped_current_weights(positions, dimension)
        for target, target_weight in targets.items():
            current_value = current.get(target, {}).get("value", 0.0)
            current_weight = current.get(target, {}).get("weight", 0.0)
            target_value = total * target_weight
            diff = target_value - current_value
            drift = target_weight - current_weight
            if abs(drift) < threshold or abs(diff) < min_trade:
                continue
            recommendations.append(
                {
                    "kind": "rebalance",
                    "dimension": dimension,
                    "target": target,
                    "action": "BUY" if diff > 0 else "SELL",
                    "amount": abs(diff),
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "drift": drift,
                    "reason": f"{dimension} weight drift {drift:.2%}",
                }
            )
    return recommendations


def rule_recommendations(
    positions: list[dict[str, Any]],
    config: dict[str, Any],
    metrics: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rules = config.get("rules", {}) or {}
    symbol_targets = target_map(config, "symbols")
    total = sum(item["value_base"] for item in positions)
    cash = sum(
        item["value_base"]
        for item in positions
        if item["symbol"].upper() == "CASH" or item["security_type"] in {"现金", "cash", "Cash"}
    )
    values = grouped_current_weights(positions, "symbol")
    recommendations = []

    drawdown_rule = rules.get("drawdown_buy", {}) or {}
    if as_bool(drawdown_rule.get("enabled"), False):
        threshold = as_float(drawdown_rule.get("threshold"), -0.08)
        cash_pct = as_float(drawdown_rule.get("cash_pct"), 0.25)
        min_trade = as_float(drawdown_rule.get("min_trade_amount"), 0.0)
        for symbol, metric in metrics.items():
            current_drawdown = metric.get("current_drawdown", 0.0)
            if current_drawdown > threshold:
                continue
            target_weight = symbol_targets.get(symbol)
            current_value = values.get(symbol, {}).get("value", 0.0)
            if target_weight is not None:
                target_gap = max(total * target_weight - current_value, 0.0)
                amount = min(cash * cash_pct, target_gap) if target_gap else cash * cash_pct
            else:
                amount = cash * cash_pct
            if amount >= min_trade and amount > 0:
                recommendations.append(
                    {
                        "kind": "drawdown_buy",
                        "dimension": "symbol",
                        "target": symbol,
                        "action": "BUY",
                        "amount": amount,
                        "current_weight": values.get(symbol, {}).get("weight", 0.0),
                        "target_weight": target_weight,
                        "drift": None,
                        "reason": f"current drawdown {current_drawdown:.2%} <= {threshold:.2%}",
                    }
                )

    gain_rule = rules.get("gain_sell", {}) or {}
    if as_bool(gain_rule.get("enabled"), False):
        threshold = as_float(gain_rule.get("threshold"), 0.15)
        sell_excess_pct = as_float(gain_rule.get("sell_excess_pct"), 0.5)
        min_trade = as_float(gain_rule.get("min_trade_amount"), 0.0)
        for symbol, metric in metrics.items():
            total_return = metric.get("total_return", 0.0)
            if total_return < threshold:
                continue
            target_weight = symbol_targets.get(symbol)
            current_value = values.get(symbol, {}).get("value", 0.0)
            target_value = total * target_weight if target_weight is not None else 0.0
            excess = max(current_value - target_value, 0.0)
            amount = excess * sell_excess_pct if excess else current_value * sell_excess_pct
            if amount >= min_trade and amount > 0:
                recommendations.append(
                    {
                        "kind": "gain_sell",
                        "dimension": "symbol",
                        "target": symbol,
                        "action": "SELL",
                        "amount": amount,
                        "current_weight": values.get(symbol, {}).get("weight", 0.0),
                        "target_weight": target_weight,
                        "drift": None,
                        "reason": f"history return {total_return:.2%} >= {threshold:.2%}",
                    }
                )
    return recommendations


def analyze_portfolio(
    holdings_path: Path | str,
    prices_path: Path | str | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = normalize_config(config)
    history = load_price_history(prices_path)
    positions = load_positions(holdings_path, history, config)
    metrics = {"symbols": symbol_metrics(history)}
    recommendations = rebalance_recommendations(positions, config)
    recommendations.extend(rule_recommendations(positions, config, metrics["symbols"]))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_currency": config["base_currency"],
        "total_value": sum(item["value_base"] for item in positions),
        "positions": positions,
        "groups": {
            "platform": group_positions(positions, "platform"),
            "asset_class": group_positions(positions, "asset_class"),
            "market": group_positions(positions, "market"),
            "security_type": group_positions(positions, "security_type"),
        },
        "metrics": metrics,
        "recommendations": recommendations,
        "config": config,
    }


def fmt_money(value: float, currency: str = "CNY") -> str:
    return f"{value:,.2f} {currency}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2%}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        output.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(output)


def render_markdown(analysis: dict[str, Any]) -> str:
    currency = analysis["base_currency"]
    lines = [
        "# Portfolio Guard Report",
        "",
        f"- Generated: `{analysis['generated_at']}`",
        f"- Total value: **{fmt_money(analysis['total_value'], currency)}**",
        "",
        "> 这是基于本地配置规则的机械提示，不构成投资建议。实际交易前需要自行核对行情、费用、税费、申赎状态和风险承受能力。",
        "",
    ]

    lines.append("## Asset Allocation")
    rows = []
    for name, item in analysis["groups"]["asset_class"].items():
        rows.append([name, fmt_money(item["value"], currency), fmt_pct(item["weight"]), fmt_pct(item["pnl_pct"])])
    lines.append(markdown_table(["资产类别", "市值", "占比", "持仓收益率"], rows))
    lines.append("")

    lines.append("## Platform Allocation")
    rows = []
    for name, item in analysis["groups"]["platform"].items():
        rows.append([name, fmt_money(item["value"], currency), fmt_pct(item["weight"]), item["count"]])
    lines.append(markdown_table(["平台", "市值", "占比", "持仓数"], rows))
    lines.append("")

    lines.append("## Positions")
    rows = []
    for item in sorted(analysis["positions"], key=lambda x: x["value_base"], reverse=True):
        rows.append(
            [
                item["symbol"],
                item["name"],
                item["platform"],
                item["asset_class"],
                fmt_money(item["value_base"], currency),
                fmt_pct(item["weight"]),
                fmt_pct(item["pnl_pct"]),
            ]
        )
    lines.append(markdown_table(["代码", "名称", "平台", "资产类别", "市值", "占比", "持仓收益率"], rows))
    lines.append("")

    lines.append("## Historical Metrics")
    metric_rows = []
    for symbol, item in sorted(analysis["metrics"]["symbols"].items()):
        metric_rows.append(
            [
                symbol,
                fmt_pct(item["total_return"]),
                fmt_pct(item["annualized_return"]),
                fmt_pct(item["annualized_volatility"]),
                fmt_pct(item["max_drawdown"]),
                fmt_pct(item["current_drawdown"]),
            ]
        )
    lines.append(
        markdown_table(
            ["代码", "区间收益", "年化收益", "年化波动", "最大回撤", "当前回撤"],
            metric_rows,
        )
        if metric_rows
        else "No price history available."
    )
    lines.append("")

    lines.append("## Rule-Based Actions")
    action_rows = []
    for item in analysis["recommendations"]:
        action_rows.append(
            [
                item["kind"],
                item["dimension"],
                item["target"],
                item["action"],
                fmt_money(item["amount"], currency),
                fmt_pct(item["current_weight"]),
                fmt_pct(item["target_weight"]),
                item["reason"],
            ]
        )
    lines.append(
        markdown_table(
            ["类型", "维度", "对象", "动作", "金额", "当前占比", "目标占比", "原因"],
            action_rows,
        )
        if action_rows
        else "No action triggered."
    )
    lines.append("")
    return "\n".join(lines)


def render_html(markdown: str) -> str:
    body_lines = []
    in_table = False
    for line in markdown.splitlines():
        if line.startswith("# "):
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("| "):
            cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
            if set(cell.strip("- ") for cell in cells) == {""}:
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                body_lines.append("<table>")
                in_table = True
            body_lines.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>")
        else:
            if in_table:
                body_lines.append("</table>")
                in_table = False
            if line.strip():
                body_lines.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body_lines.append("</table>")
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Portfolio Guard Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2937; }
    h1, h2 { color: #111827; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }
    th, td { border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; }
    th { background: #f3f4f6; }
    p { line-height: 1.6; }
  </style>
</head>
<body>
""" + "\n".join(body_lines) + "\n</body>\n</html>\n"


def write_reports(analysis: dict[str, Any], out_dir: Path | str) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(analysis)
    md_path = out_dir / "latest.md"
    html_path = out_dir / "latest.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(render_html(markdown), encoding="utf-8")
    return md_path, html_path


def command_analyze(args: argparse.Namespace) -> int:
    config = read_config(args.config)
    analysis = analyze_portfolio(args.holdings, args.prices, config)
    md_path, html_path = write_reports(analysis, args.out_dir)
    print(f"Wrote {md_path}")
    print(f"Wrote {html_path}")
    return 0


def command_update_prices(args: argparse.Namespace) -> int:
    written = update_prices(args.instruments, args.prices, args.start, args.end)
    print(f"Updated {written}")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    try:
        from flask import Flask
    except ImportError as exc:  # pragma: no cover - local environment has Flask
        raise RuntimeError("serve requires Flask. Install flask or use analyze command.") from exc

    app = Flask(__name__)

    @app.route("/")
    def index():
        config = read_config(args.config)
        analysis = analyze_portfolio(args.holdings, args.prices, config)
        return render_html(render_markdown(analysis))

    app.run(host=args.host, port=args.port, debug=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local portfolio allocation and rule helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Generate Markdown and HTML reports.")
    analyze.add_argument("--holdings", required=True, help="Path to holdings CSV.")
    analyze.add_argument("--prices", help="Path to historical prices CSV.")
    analyze.add_argument("--config", required=True, help="Path to YAML or JSON config.")
    analyze.add_argument("--out-dir", default="reports", help="Output report directory.")
    analyze.set_defaults(func=command_analyze)

    update = sub.add_parser("update-prices", help="Fetch and merge historical prices.")
    update.add_argument("--instruments", required=True, help="Path to instruments CSV.")
    update.add_argument("--prices", required=True, help="Path to prices CSV to update.")
    update.add_argument("--start", required=True, help="Start date, e.g. 20240101.")
    update.add_argument("--end", required=True, help="End date, e.g. 20260608.")
    update.set_defaults(func=command_update_prices)

    serve = sub.add_parser("serve", help="Start a tiny local Flask report server.")
    serve.add_argument("--holdings", required=True, help="Path to holdings CSV.")
    serve.add_argument("--prices", help="Path to historical prices CSV.")
    serve.add_argument("--config", required=True, help="Path to YAML or JSON config.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8765, type=int)
    serve.set_defaults(func=command_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
