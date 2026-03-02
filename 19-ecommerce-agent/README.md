# 🛍️ E-Commerce Agent

An intelligent shopping assistant built with **LangGraph** that helps customers discover products, compare options, manage their cart, and receive personalized recommendations through natural conversation.

## Features

- 🔍 **Smart product search**: Filter by category, price, rating, and features
- 🎯 **Personalized recommendations**: Based on budget, interests, and browsing history
- 🛒 **Cart management**: Add items, view totals with tax and shipping
- 💰 **Deal highlighting**: Automatic detection of discounted items
- 📦 **Stock checking**: Real-time availability status
- 💬 **Conversational UX**: Natural language shopping experience

## Architecture

```
Customer Message → Shopping Agent → [Search Products] | [Get Details] | [Add to Cart]
                        ↓
              [View Cart] | [Recommendations]
                        ↓
              Personalized Shopping Response
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Tools**: Product Search, Product Details, Cart Manager, Recommendation Engine
- **Pattern**: Conversational commerce agent with session state

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

Interactive shopping session:
```
You: I need wireless headphones under $300
Agent: Found 1 match! 🎧 Wireless Noise-Canceling Headphones
       $212.49 (was $249.99, 15% off) | ⭐ 4.7/5 ✅ In Stock

You: Add them to my cart
Agent: ✅ Added! Cart total: $212.49 (1 item)

You: What deals do you have in electronics?
Agent: [Lists all discounted electronics with savings]

You: Show my cart
Agent: 🛒 Your Cart:
       [P001] Wireless Headphones - $212.49 x 1 = $212.49
       Subtotal: $212.49 | Tax: $17.00 | FREE Shipping
       TOTAL: $229.49
```

## Product Catalog

| Category | Sample Products | Price Range |
|----------|----------------|-------------|
| Electronics | Headphones, TV, Keyboard, Camera | $50-$650 |
| Furniture | Ergonomic Chair, Desk accessories | $50-$400 |
| Sports | Running Shoes, Yoga Mat | $70-$130 |
| Kitchen | Coffee Maker | $180 |
| Lifestyle | Water Bottle | $30 |

## Key Features

### Smart Search
```python
results = search_products(
    query="wireless",
    category="electronics",
    max_price=300,
    min_rating=4.5
)
```

### Cart Calculation
- Subtotal of all items
- 8% tax calculation
- Free shipping on orders ≥ $50
- Running total updates

### Recommendation Engine
- Scores based on rating, discounts, view history, and interests
- Budget-aware filtering
- Highlights active promotions

## Extending with Real Data

```python
# Replace PRODUCT_CATALOG with database calls
from database import get_products

PRODUCT_CATALOG = get_products(active=True)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
