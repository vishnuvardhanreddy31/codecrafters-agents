"""
Financial Analysis Agent
Uses LangGraph with yfinance tools to analyze stocks, compute metrics,
detect trends, and generate investment insights.
"""

import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# ── State ──────────────────────────────────────────────────────────────────────
class FinancialState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    ticker: str
    analysis_type: str


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def get_stock_info(ticker: str) -> str:
    """Retrieve current stock information and key metrics for a given ticker symbol."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        fields = [
            ("Company", info.get("longName", "N/A")),
            ("Sector", info.get("sector", "N/A")),
            ("Industry", info.get("industry", "N/A")),
            ("Current Price", f"${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}"),
            ("Market Cap", f"${info.get('marketCap', 0):,}"),
            ("P/E Ratio", info.get("trailingPE", "N/A")),
            ("Forward P/E", info.get("forwardPE", "N/A")),
            ("EPS (TTM)", info.get("trailingEps", "N/A")),
            ("Dividend Yield", f"{info.get('dividendYield', 0) * 100:.2f}%"),
            ("52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}"),
            ("52-Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}"),
            ("Analyst Target", f"${info.get('targetMeanPrice', 'N/A')}"),
            ("Beta", info.get("beta", "N/A")),
        ]
        return "\n".join(f"  {k}: {v}" for k, v in fields)
    except Exception as e:
        return f"Error fetching data for {ticker}: {e}"


@tool
def get_price_history(ticker: str, period: str = "6mo") -> str:
    """Get historical price data and calculate technical indicators.
    Period options: 1mo, 3mo, 6mo, 1y, 2y, 5y"""
    try:
        import yfinance as yf
        import pandas as pd

        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        if hist.empty:
            return f"No historical data found for {ticker}."

        close = hist["Close"]
        returns = close.pct_change().dropna()

        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]
        start = close.iloc[0]
        period_return = ((current - start) / start) * 100

        volatility = returns.std() * (252 ** 0.5) * 100  # annualized

        return (
            f"Price History ({period}) for {ticker.upper()}:\n"
            f"  Start Price: ${start:.2f}\n"
            f"  Current Price: ${current:.2f}\n"
            f"  Period Return: {period_return:.2f}%\n"
            f"  20-Day SMA: ${sma_20:.2f} ({'above' if current > sma_20 else 'below'} price)\n"
            f"  50-Day SMA: ${sma_50:.2f} ({'above' if current > sma_50 else 'below'} price)\n"
            f"  Annualized Volatility: {volatility:.1f}%\n"
            f"  Max Drawdown Period: {close.min():.2f} - {close.max():.2f}\n"
            f"  Data points: {len(hist)} trading days"
        )
    except Exception as e:
        return f"Error: {e}"


@tool
def compare_stocks(tickers: str, period: str = "1y") -> str:
    """Compare multiple stocks by performance over a period.
    tickers: comma-separated list (e.g., 'AAPL,MSFT,GOOGL')"""
    try:
        import yfinance as yf

        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        results = []
        for t in ticker_list:
            try:
                stock = yf.Ticker(t)
                hist = stock.history(period=period)
                if hist.empty:
                    results.append(f"  {t}: No data")
                    continue
                close = hist["Close"]
                ret = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
                volatility = close.pct_change().dropna().std() * (252 ** 0.5) * 100
                results.append(
                    f"  {t}: {ret:+.1f}% return, {volatility:.1f}% volatility"
                )
            except Exception:
                results.append(f"  {t}: Error fetching data")

        return f"Stock Comparison ({period}):\n" + "\n".join(results)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_financial_ratios(ticker: str) -> str:
    """Calculate key financial ratios for fundamental analysis."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        ratios = [
            ("P/E Ratio", info.get("trailingPE", "N/A")),
            ("P/B Ratio", info.get("priceToBook", "N/A")),
            ("P/S Ratio", info.get("priceToSalesTrailing12Months", "N/A")),
            ("EV/EBITDA", info.get("enterpriseToEbitda", "N/A")),
            ("Profit Margin", f"{info.get('profitMargins', 0) * 100:.1f}%"),
            ("ROE", f"{info.get('returnOnEquity', 0) * 100:.1f}%"),
            ("ROA", f"{info.get('returnOnAssets', 0) * 100:.1f}%"),
            ("Debt/Equity", info.get("debtToEquity", "N/A")),
            ("Current Ratio", info.get("currentRatio", "N/A")),
            ("Free Cash Flow", f"${info.get('freeCashflow', 0):,}"),
        ]
        return f"Financial Ratios for {ticker.upper()}:\n" + "\n".join(
            f"  {k}: {v}" for k, v in ratios
        )
    except Exception as e:
        return f"Error: {e}"


tools = [get_stock_info, get_price_history, compare_stocks, get_financial_ratios]

SYSTEM_PROMPT = """You are a professional financial analyst. Use the available tools to:
1. Retrieve current stock data and key metrics
2. Analyze price history and technical indicators
3. Compute fundamental financial ratios
4. Compare performance against peers when relevant
5. Synthesize findings into a clear investment analysis

Important: This is for educational purposes only. Always include a disclaimer that
this is not professional financial advice."""


def build_financial_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    def analyst(state: FinancialState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    tool_node = ToolNode(tools)
    graph = StateGraph(FinancialState)
    graph.add_node("analyst", analyst)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "analyst")
    graph.add_conditional_edges("analyst", tools_condition)
    graph.add_edge("tools", "analyst")
    return graph.compile()


def analyze_stock(ticker: str, analysis_type: str = "comprehensive") -> str:
    app = build_financial_graph()
    prompt = (
        f"Perform a {analysis_type} analysis of {ticker.upper()} stock.\n"
        f"System: {SYSTEM_PROMPT}\n\n"
        "Use all available tools to gather data, then provide a structured analysis with:\n"
        "- Company overview\n"
        "- Current valuation metrics\n"
        "- Price performance and technical levels\n"
        "- Financial health assessment\n"
        "- Key risks and opportunities\n"
        "- Investment outlook summary"
    )
    state: FinancialState = {
        "messages": [HumanMessage(content=prompt)],
        "ticker": ticker,
        "analysis_type": analysis_type,
    }
    result = app.invoke(state)
    return result["messages"][-1].content


def main():
    print("📈 Financial Analysis Agent\n" + "=" * 60)
    ticker = input("Enter stock ticker (e.g., AAPL, MSFT, TSLA): ").strip() or "AAPL"
    print(f"\nAnalyzing {ticker.upper()}...\n")
    analysis = analyze_stock(ticker)
    print(analysis)
    print("\n⚠️  Disclaimer: This analysis is for educational purposes only.")
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()
