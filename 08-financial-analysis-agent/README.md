# 📈 Financial Analysis Agent

An intelligent stock analysis agent built with **LangGraph** that fetches real-time market data, computes technical and fundamental indicators, and provides comprehensive investment analysis.

> ⚠️ **Disclaimer**: This tool is for educational purposes only. Not financial advice.

## Features

- 💹 **Real-time stock data** via Yahoo Finance (yfinance)
- 📊 **Technical indicators**: SMA-20, SMA-50, volatility, period returns
- 📉 **Fundamental ratios**: P/E, P/B, P/S, EV/EBITDA, ROE, ROA, Debt/Equity
- 🔄 **Multi-stock comparison**: Side-by-side performance analysis
- 💡 **Investment insights**: Synthesized analysis with risk/opportunity assessment

## Architecture

```
Ticker Symbol → Financial Agent → [Stock Info] → [Price History] → [Ratios] → [Comparison]
                     ↑___________________________|
                     (fetches all relevant data)
                           ↓
                 Comprehensive Investment Analysis
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Data**: yfinance (Yahoo Finance), pandas, numpy
- **Tools**: Stock Info Fetcher, Price History Analyzer, Ratio Calculator, Comparison Tool
- **Pattern**: Data-augmented analytical agent

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
```

## Usage

```bash
python agent.py
```

Or use programmatically:
```python
from agent import analyze_stock

# Comprehensive analysis
analysis = analyze_stock("MSFT", "comprehensive")
print(analysis)

# Compare multiple stocks
from agent import build_financial_graph
from langchain_core.messages import HumanMessage

app = build_financial_graph()
result = app.invoke({
    "messages": [HumanMessage(content="Compare AAPL, MSFT, and GOOGL over the last year")],
    "ticker": "multiple",
    "analysis_type": "comparison"
})
```

## Analysis Components

| Component | Tool | Data Source |
|-----------|------|-------------|
| Company Overview | `get_stock_info` | Yahoo Finance |
| Price History | `get_price_history` | Historical OHLCV data |
| Financial Ratios | `get_financial_ratios` | Yahoo Finance fundamentals |
| Peer Comparison | `compare_stocks` | Multi-ticker analysis |

## Technical Indicators

- **SMA-20**: 20-day Simple Moving Average
- **SMA-50**: 50-day Simple Moving Average
- **Annualized Volatility**: (Daily std dev × √252)
- **Period Return**: % change over analysis period
- **Beta**: Market sensitivity coefficient

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
