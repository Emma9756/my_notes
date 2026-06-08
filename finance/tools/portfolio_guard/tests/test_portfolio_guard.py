import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "portfolio_guard.py"


def load_module():
    spec = importlib.util.spec_from_file_location("portfolio_guard", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PortfolioGuardTest(unittest.TestCase):
    def setUp(self):
        self.pg = load_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_csv(self, name, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_positions_convert_currency_and_group_weights(self):
        holdings = self.write_csv(
            "holdings.csv",
            [
                {
                    "account": "银河",
                    "platform": "银河证券场内",
                    "symbol": "510300.SH",
                    "name": "沪深300ETF",
                    "market": "A股",
                    "asset_class": "A股",
                    "security_type": "ETF",
                    "currency": "CNY",
                    "quantity": "1000",
                    "price": "4",
                    "market_value": "",
                    "cost_price": "3",
                    "cost_amount": "",
                    "as_of": "2026-06-08",
                },
                {
                    "account": "银河",
                    "platform": "银河证券场内",
                    "symbol": "AAPL",
                    "name": "Apple",
                    "market": "美股",
                    "asset_class": "美股",
                    "security_type": "个股",
                    "currency": "USD",
                    "quantity": "10",
                    "price": "200",
                    "market_value": "",
                    "cost_price": "150",
                    "cost_amount": "",
                    "as_of": "2026-06-08",
                },
            ],
        )
        config = {"base_currency": "CNY", "fx_rates": {"CNY": 1, "USD": 7.2}}

        analysis = self.pg.analyze_portfolio(holdings, None, config)

        self.assertAlmostEqual(analysis["total_value"], 18400.0)
        self.assertAlmostEqual(analysis["groups"]["asset_class"]["A股"]["weight"], 4000 / 18400)
        self.assertAlmostEqual(analysis["groups"]["asset_class"]["美股"]["value"], 14400.0)

    def test_rebalance_recommendations_use_symbol_targets(self):
        holdings = self.write_csv(
            "holdings.csv",
            [
                {
                    "account": "银河",
                    "platform": "银河证券场内",
                    "symbol": "A",
                    "name": "A",
                    "market": "A股",
                    "asset_class": "A股",
                    "security_type": "ETF",
                    "currency": "CNY",
                    "quantity": "1",
                    "price": "700",
                    "market_value": "",
                    "cost_price": "700",
                    "cost_amount": "",
                    "as_of": "2026-06-08",
                },
                {
                    "account": "银河",
                    "platform": "银河证券场内",
                    "symbol": "B",
                    "name": "B",
                    "market": "黄金",
                    "asset_class": "黄金",
                    "security_type": "ETF",
                    "currency": "CNY",
                    "quantity": "1",
                    "price": "300",
                    "market_value": "",
                    "cost_price": "300",
                    "cost_amount": "",
                    "as_of": "2026-06-08",
                },
            ],
        )
        config = {
            "base_currency": "CNY",
            "fx_rates": {"CNY": 1},
            "targets": {"symbols": {"A": 0.5, "B": 0.5}},
            "rules": {"rebalance": {"drift_threshold": 0.05, "min_trade_amount": 10}},
        }

        analysis = self.pg.analyze_portfolio(holdings, None, config)
        actions = {(item["target"], item["action"]): item for item in analysis["recommendations"]}

        self.assertAlmostEqual(actions[("A", "SELL")]["amount"], 200.0)
        self.assertAlmostEqual(actions[("B", "BUY")]["amount"], 200.0)

    def test_price_history_produces_return_and_drawdown_metrics(self):
        holdings = self.write_csv(
            "holdings.csv",
            [
                {
                    "account": "支付宝",
                    "platform": "支付宝场外",
                    "symbol": "FUND",
                    "name": "基金",
                    "market": "A股",
                    "asset_class": "A股",
                    "security_type": "场外基金",
                    "currency": "CNY",
                    "quantity": "100",
                    "price": "8",
                    "market_value": "",
                    "cost_price": "10",
                    "cost_amount": "",
                    "as_of": "2026-06-08",
                }
            ],
        )
        prices = self.write_csv(
            "prices.csv",
            [
                {"date": "2026-06-01", "symbol": "FUND", "close": "10", "currency": "CNY"},
                {"date": "2026-06-02", "symbol": "FUND", "close": "12", "currency": "CNY"},
                {"date": "2026-06-03", "symbol": "FUND", "close": "8", "currency": "CNY"},
            ],
        )
        config = {
            "base_currency": "CNY",
            "fx_rates": {"CNY": 1},
            "targets": {"symbols": {"FUND": 1.0}},
            "rules": {
                "drawdown_buy": {
                    "enabled": True,
                    "threshold": -0.3,
                    "cash_pct": 0.5,
                    "min_trade_amount": 1,
                }
            },
        }

        analysis = self.pg.analyze_portfolio(holdings, prices, config)
        metric = analysis["metrics"]["symbols"]["FUND"]

        self.assertAlmostEqual(metric["total_return"], -0.2)
        self.assertAlmostEqual(metric["max_drawdown"], -1 / 3)
        self.assertIsNone(metric["annualized_return"])
        self.assertIsNone(metric["annualized_volatility"])

    def test_update_prices_merges_and_deduplicates_rows(self):
        instruments = self.write_csv(
            "instruments.csv",
            [
                {
                    "symbol": "510300",
                    "name": "300ETF",
                    "source": "akshare",
                    "kind": "etf",
                    "currency": "CNY",
                    "enabled": "1",
                },
                {
                    "symbol": "161226",
                    "name": "白银基金",
                    "source": "akshare",
                    "kind": "lof",
                    "currency": "CNY",
                    "enabled": "1",
                },
            ],
        )
        prices = self.write_csv(
            "prices.csv",
            [
                {"date": "2026-06-01", "symbol": "510300", "close": "4.0", "currency": "CNY"},
                {"date": "2026-06-02", "symbol": "510300", "close": "4.1", "currency": "CNY"},
            ],
        )

        class FakeProvider:
            def fetch_history(self, instrument, start_date, end_date):
                self.last = (instrument["symbol"], instrument["kind"], start_date, end_date)
                return [
                    {"date": "2026-06-02", "symbol": instrument["symbol"], "close": 4.2, "currency": "CNY"},
                    {"date": "2026-06-03", "symbol": instrument["symbol"], "close": 4.3, "currency": "CNY"},
                ]

        written = self.pg.update_prices(instruments, prices, "20260601", "20260603", FakeProvider())
        with written.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        self.assertEqual(
            [(row["date"], row["symbol"], row["close"]) for row in rows],
            [
                ("2026-06-01", "510300", "4.0"),
                ("2026-06-02", "510300", "4.2"),
                ("2026-06-03", "510300", "4.3"),
                ("2026-06-02", "161226", "4.2"),
                ("2026-06-03", "161226", "4.3"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
