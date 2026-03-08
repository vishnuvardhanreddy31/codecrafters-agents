# 📊 Data Analysis Agent

An intelligent data analysis agent built with **LangGraph** that autonomously explores datasets, computes statistics, detects outliers, generates visualizations, and provides natural language insights.

## Features

- 📂 **Automatic CSV loading** with schema detection
- 📈 **Statistical summaries**: mean, median, std, quartiles
- 🔗 **Correlation analysis** for numeric columns
- 🚨 **Outlier detection** using IQR method
- 📉 **Chart generation**: histogram, scatter, bar, line plots
- 💡 **Natural language insights** and actionable recommendations

## Architecture

```
Dataset + Question → Analyst Agent → [Load CSV] → [Correlations] → [Outlier Detection] → [Charts]
                           ↑___________________________|
                           (uses tools as needed)
                                 ↓
                    Comprehensive Analysis Report + Charts
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Libraries**: pandas, numpy, matplotlib, seaborn
- **Tools**: CSV Loader, Correlation Analyzer, Outlier Detector, Chart Generator
- **Pattern**: Tool-augmented analytical agent

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
from agent import analyze_dataset

# Analyze your own CSV
result = analyze_dataset(
    filepath="data.csv",
    question="Find trends in sales over time and identify top-performing products"
)
print(result)
```

## Supported Analyses

| Analysis Type | Tool | Output |
|--------------|------|--------|
| Data overview | `load_and_describe_csv` | Shape, dtypes, nulls, statistics |
| Correlations | `compute_correlations` | Correlation matrix for numeric cols |
| Outlier detection | `detect_outliers` | IQR bounds + outlier values |
| Visualization | `generate_chart` | PNG chart file |

## Supported Chart Types

- `histogram` - Distribution of a single column
- `scatter` - Relationship between two columns
- `bar` - Frequency of categorical values
- `line` - Trend over an index

## Example Output

```
Dataset: 200 rows × 5 columns
Numeric columns: age, income, spending_score

Key Insights:
- Income shows strong right skew (mean: $55K, outliers up to $150K)
- Significant correlation between age and spending_score (r=0.63)
- 12 outliers detected in income column using IQR method
- Chart saved as 'chart.png'
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
