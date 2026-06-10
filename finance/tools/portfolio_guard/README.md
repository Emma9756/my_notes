# Portfolio Guard

`portfolio_guard` 是一个本地轻量投资组合工具，用于维护银河证券场内持仓，并按自定义目标配比生成规则型操作提示。

它不是券商交易工具，也不会自动下单。输出内容是基于本地配置的机械提示，不构成投资建议。

## 功能

1. 汇总当前市值和配比
   - 按平台、资产类别、市场、证券类型聚合。
   - 支持人民币、美元、港币等通过配置汇率换算到统一基准币种。
   - 当前只保留银河证券场内口径，支持场内 ETF、LOF、个股、黄金、债券和现金。

2. 计算历史参考指标
   - 读取本地 `prices.csv` 历史价格。
   - 计算区间收益、年化收益、年化波动、最大回撤、当前回撤。
   - 可通过 `update-prices` 命令联网更新公开行情。

3. 生成规则型操作提示
   - 目标配比偏离超过阈值时，提示买入或卖出到目标仓位。
   - 标的从阶段高点回撤超过阈值时，按现金仓比例提示加仓金额。
   - 标的区间收益超过阈值且超过目标仓位时，提示分批减仓金额。

4. 输出报告
   - `reports/latest.md`
   - `reports/latest.html`
   - 可选 `serve` 命令启动本地 Flask 页面。

## 快速运行

在仓库根目录执行：

```bash
python3 my_notes/finance/tools/portfolio_guard/portfolio_guard.py analyze \
  --holdings my_notes/finance/tools/portfolio_guard/sample/holdings.csv \
  --prices my_notes/finance/tools/portfolio_guard/sample/prices.csv \
  --config my_notes/finance/tools/portfolio_guard/config.example.yaml \
  --out-dir my_notes/finance/tools/portfolio_guard/reports
```

联网更新价格：

```bash
python3 my_notes/finance/tools/portfolio_guard/portfolio_guard.py update-prices \
  --instruments my_notes/finance/tools/portfolio_guard/instruments.csv \
  --prices my_notes/finance/tools/portfolio_guard/sample/prices.csv \
  --start 20240101 \
  --end 20260608
```

联网更新最近行情，适合定时任务反复执行：

```bash
python3 my_notes/finance/tools/portfolio_guard/portfolio_guard.py update-latest \
  --instruments my_notes/finance/tools/portfolio_guard/instruments.csv \
  --prices my_notes/finance/tools/portfolio_guard/sample/prices.csv \
  --lookback-days 10
```

查看输出：

```text
my_notes/finance/tools/portfolio_guard/reports/latest.md
my_notes/finance/tools/portfolio_guard/reports/latest.html
```

启动本地页面：

```bash
python3 my_notes/finance/tools/portfolio_guard/portfolio_guard.py serve \
  --holdings my_notes/finance/tools/portfolio_guard/sample/holdings.csv \
  --prices my_notes/finance/tools/portfolio_guard/sample/prices.csv \
  --config my_notes/finance/tools/portfolio_guard/config.example.yaml \
  --host 127.0.0.1 \
  --port 8765
```

然后打开：

```text
http://127.0.0.1:8765
```

## 输入文件

### instruments.csv

字段：

| 字段 | 含义 |
|---|---|
| `symbol` | 标的代码，例如 `510300`、`09988` |
| `name` | 标的名称 |
| `source` | 当前支持 `akshare` |
| `kind` | `etf`、`lof`、`a_stock`、`hk_stock` |
| `currency` | CNY、HKD、USD |
| `enabled` | `1` 表示启用，`0` 表示跳过 |

当前内置清单覆盖：

```text
510300  300ETF        etf
159361  A500E         etf
513120  HK创新药      etf
159995  芯片ETF       etf
513050  中概互联      etf
09988   阿里巴巴      hk_stock
002594  比亚迪        a_stock
00700   腾讯控股      hk_stock
161226  白银基金      lof
603993  洛阳钼业      a_stock
000426  兴业银锡      a_stock
601899  紫金矿业      a_stock
513500  标普500       etf
159941  纳指ETF       etf
159509  纳指科技      etf
601288  农业银行      a_stock
000858  五 粮 液      a_stock
601318  中国平安      a_stock
518880  黄金ETF       etf
511090  30年国债      etf
511010  国债ETF       etf
511260  十年国债      etf
```

`161226 白银基金` 按用户确认的“场内 LOF 交易价格”处理，使用 `kind=lof`，不走场外开放式基金净值。

### holdings.csv

字段：

| 字段 | 含义 |
|---|---|
| `account` | 账户名，例如银河 |
| `platform` | 平台，当前固定使用银河证券场内 |
| `symbol` | 标的代码；现金用 `CASH` |
| `name` | 标的名称 |
| `market` | A股、美股、港股、黄金、现金等 |
| `asset_class` | 用于目标配比的资产类别 |
| `security_type` | ETF、LOF、个股、现金等 |
| `currency` | CNY、USD、HKD |
| `quantity` | 份额或股数 |
| `price` | 当前价格或净值 |
| `market_value` | 可选，若填了则直接使用该市值 |
| `cost_price` | 成本价 |
| `cost_amount` | 可选，若填了则直接使用总成本 |
| `as_of` | 估值日期 |

### prices.csv

字段：

| 字段 | 含义 |
|---|---|
| `date` | 价格日期 |
| `symbol` | 标的代码 |
| `close` | 收盘价或基金净值 |
| `currency` | 币种 |

### config.yaml

核心结构：

```yaml
base_currency: CNY
fx_rates:
  CNY: 1
  USD: 7.2
  HKD: 0.92

targets:
  asset_classes:
    A股: 0.30
    港股: 0.20
    黄金: 0.15
    贵金属: 0.10
    债券: 0.10
    现金: 0.15
  symbols:
    510300: 0.18
    09988: 0.12
    CASH: 0.15

rules:
  rebalance:
    drift_threshold: 0.03
    min_trade_amount: 500
  drawdown_buy:
    enabled: true
    threshold: -0.08
    cash_pct: 0.25
    min_trade_amount: 500
  gain_sell:
    enabled: true
    threshold: 0.15
    sell_excess_pct: 0.5
    min_trade_amount: 500
```

## 实现原理

### 1. 市值统一

`portfolio_guard.py` 读取 `holdings.csv` 后，对每条持仓计算：

```text
market_value = market_value 或 quantity * price
value_base = market_value * fx_rates[currency]
weight = value_base / portfolio_total_value
```

如果是现金，可以直接填 `market_value`，`quantity` 和 `price` 留空。

### 2. 分组聚合

工具对以下维度分别聚合：

```text
platform
asset_class
market
security_type
```

每个聚合项输出：

```text
value
cost
weight
pnl
pnl_pct
count
```

### 3. 历史风险指标

对 `prices.csv` 中每个 `symbol` 的价格序列计算：

```text
total_return = last / first - 1
annualized_return = (1 + total_return) ** (1 / years) - 1
annualized_volatility = std(daily_returns) * sqrt(252)
max_drawdown = min(price / historical_peak - 1)
current_drawdown = latest / historical_peak - 1
```

如果历史价格少于 30 个点，年化收益和年化波动会显示为 `-`，避免短周期数据被年化后严重放大。实际使用时应导入更长周期的数据。

### 4. 操作提示

再平衡提示按目标市值和当前市值的差值计算：

```text
target_value = total_value * target_weight
diff = target_value - current_value
drift = target_weight - current_weight
```

当 `abs(drift) >= drift_threshold` 且 `abs(diff) >= min_trade_amount`：

```text
diff > 0 -> BUY abs(diff)
diff < 0 -> SELL abs(diff)
```

回撤加仓提示：

```text
current_drawdown <= threshold
amount = min(cash * cash_pct, target_gap)
```

收益减仓提示：

```text
total_return >= threshold
amount = excess_above_target * sell_excess_pct
```

### 5. 报告生成

`analyze` 命令会生成 Markdown，再由内置简单渲染器转成 HTML。HTML 不依赖前端框架，方便直接打开或长期归档。

### 6. 联网更新价格

`update-prices` 会读取 `instruments.csv`，调用 AKShare 获取历史行情，再和已有 `prices.csv` 去重合并。相同 `date + symbol` 的记录以新拉取数据为准。

`update-latest` 是定期任务入口，会按 `--lookback-days` 指定的最近自然日窗口拉取 `instruments.csv` 中启用标的的最新公开行情，并合并到 `sample/prices.csv`。重复执行不会产生重复行。

数据源映射：

```text
kind=etf      -> ak.fund_etf_hist_em
kind=lof      -> ak.fund_etf_hist_em
kind=a_stock  -> ak.stock_zh_a_hist
kind=hk_stock -> ak.stock_hk_hist
```

当 Eastmoney 历史行情接口不可用时，会自动切换到 AKShare 的备用接口：

```text
kind=etf/lof  -> ak.fund_etf_hist_sina
kind=a_stock  -> ak.stock_zh_a_hist_tx
kind=hk_stock -> ak.stock_hk_daily
```

运行前需要安装依赖：

```bash
python3 -m pip install -r my_notes/finance/tools/portfolio_guard/requirements.txt
```

如果 AKShare 接口临时不可用，命令会报出失败标的，已成功获取的标的不会静默伪造数据。

## 定期运行

例如每天晚上 22:30 更新最近行情并生成一次报告：

```cron
30 22 * * * cd /data/dnn/qinzq/repository/current && python3 my_notes/finance/tools/portfolio_guard/portfolio_guard.py update-latest --instruments my_notes/finance/tools/portfolio_guard/instruments.csv --prices my_notes/finance/tools/portfolio_guard/sample/prices.csv --lookback-days 10 && python3 my_notes/finance/tools/portfolio_guard/portfolio_guard.py analyze --holdings my_notes/finance/tools/portfolio_guard/sample/holdings.csv --prices my_notes/finance/tools/portfolio_guard/sample/prices.csv --config my_notes/finance/tools/portfolio_guard/config.example.yaml --out-dir my_notes/finance/tools/portfolio_guard/reports
```

Windows 计划任务可使用同一条 `update-latest` 命令，工作目录设为仓库根目录；`sample/prices.csv` 会被原地合并更新。

## 后续可扩展

- 增加 yfinance 数据源：美股和部分港股。
- 增加从券商导出的 CSV 自动映射。
- 增加交易费用、申赎费、QDII 限购状态字段。
- 增加组合级历史净值和目标组合回测。
