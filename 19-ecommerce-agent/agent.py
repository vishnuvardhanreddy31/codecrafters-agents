"""
E-Commerce Agent
Uses LangGraph to provide intelligent product recommendations, shopping assistance,
price comparisons, and personalized deal discovery.
"""

import os
import random
from typing import Annotated, TypedDict, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ── Product catalog (mock) ─────────────────────────────────────────────────────
PRODUCT_CATALOG = [
    {
        "id": "P001", "name": "Wireless Noise-Canceling Headphones",
        "category": "electronics", "price": 249.99, "rating": 4.7,
        "brand": "SoundPro", "features": ["ANC", "30hr battery", "Bluetooth 5.3"],
        "in_stock": True, "discount": 15,
    },
    {
        "id": "P002", "name": "Ergonomic Office Chair",
        "category": "furniture", "price": 399.99, "rating": 4.5,
        "brand": "ComfortPlus", "features": ["lumbar support", "adjustable height", "mesh back"],
        "in_stock": True, "discount": 0,
    },
    {
        "id": "P003", "name": "4K Smart TV 55-inch",
        "category": "electronics", "price": 649.99, "rating": 4.6,
        "brand": "VisionMax", "features": ["4K HDR", "Smart TV", "Voice control", "120Hz"],
        "in_stock": True, "discount": 20,
    },
    {
        "id": "P004", "name": "Stainless Steel Water Bottle 32oz",
        "category": "lifestyle", "price": 29.99, "rating": 4.8,
        "brand": "HydroFlow", "features": ["insulated", "leak-proof", "BPA-free"],
        "in_stock": True, "discount": 0,
    },
    {
        "id": "P005", "name": "Running Shoes Men's",
        "category": "sports", "price": 129.99, "rating": 4.4,
        "brand": "StridePro", "features": ["breathable", "cushioned", "lightweight"],
        "in_stock": True, "discount": 10,
    },
    {
        "id": "P006", "name": "Coffee Maker with Grinder",
        "category": "kitchen", "price": 179.99, "rating": 4.6,
        "brand": "BrewMaster", "features": ["built-in grinder", "programmable", "12-cup"],
        "in_stock": False, "discount": 0,
    },
    {
        "id": "P007", "name": "Mechanical Keyboard",
        "category": "electronics", "price": 149.99, "rating": 4.7,
        "brand": "TypeForce", "features": ["RGB", "wireless", "Cherry MX switches"],
        "in_stock": True, "discount": 5,
    },
    {
        "id": "P008", "name": "Yoga Mat Premium",
        "category": "sports", "price": 69.99, "rating": 4.9,
        "brand": "ZenFlex", "features": ["non-slip", "eco-friendly", "6mm thick"],
        "in_stock": True, "discount": 0,
    },
    {
        "id": "P009", "name": "Laptop Stand Adjustable",
        "category": "electronics", "price": 49.99, "rating": 4.5,
        "brand": "DeskPro", "features": ["aluminum", "adjustable", "portable"],
        "in_stock": True, "discount": 0,
    },
    {
        "id": "P010", "name": "Smart Home Security Camera",
        "category": "electronics", "price": 89.99, "rating": 4.3,
        "brand": "SafeGuard", "features": ["1080p", "night vision", "two-way audio", "motion detect"],
        "in_stock": True, "discount": 25,
    },
]

# ── User preferences (session-based) ──────────────────────────────────────────
_user_cart: List[dict] = []
_user_wishlist: List[dict] = []
_view_history: List[str] = []


# ── State ──────────────────────────────────────────────────────────────────────
class EcommerceState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    budget: Optional[float]
    preferences: List[str]


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def search_products(
    query: str,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    min_rating: float = 0.0,
) -> str:
    """Search for products by keyword, category, price range, and rating.
    category options: electronics, furniture, lifestyle, sports, kitchen
    Returns matching products with prices, ratings, and availability."""
    results = []
    query_lower = query.lower()

    for product in PRODUCT_CATALOG:
        # Apply filters
        if category and product["category"] != category.lower():
            continue
        if max_price and product["price"] > max_price:
            continue
        if product["rating"] < min_rating:
            continue
        # Check if query matches
        if (query_lower in product["name"].lower()
                or query_lower in product["category"]
                or any(query_lower in f.lower() for f in product["features"])
                or query_lower in product["brand"].lower()):
            results.append(product)

    if not results:
        return f"No products found for '{query}'. Try different keywords or broader filters."

    _view_history.extend(r["id"] for r in results[:3])

    output = f"Found {len(results)} product(s) for '{query}':\n\n"
    for p in results:
        discounted = p["price"] * (1 - p["discount"] / 100) if p["discount"] else p["price"]
        price_str = (
            f"${discounted:.2f} (was ${p['price']:.2f}, {p['discount']}% off)"
            if p["discount"]
            else f"${p['price']:.2f}"
        )
        stock = "✅ In Stock" if p["in_stock"] else "❌ Out of Stock"
        output += (
            f"[{p['id']}] {p['name']}\n"
            f"  Brand: {p['brand']} | ⭐ {p['rating']}/5 | {stock}\n"
            f"  Price: {price_str}\n"
            f"  Features: {', '.join(p['features'])}\n\n"
        )
    return output


@tool
def get_product_details(product_id: str) -> str:
    """Get comprehensive details about a specific product by ID."""
    for p in PRODUCT_CATALOG:
        if p["id"] == product_id.upper():
            discounted = p["price"] * (1 - p["discount"] / 100) if p["discount"] else p["price"]
            _view_history.append(product_id)
            return (
                f"Product Details: {p['name']}\n"
                f"  ID: {p['id']}\n"
                f"  Brand: {p['brand']}\n"
                f"  Category: {p['category'].title()}\n"
                f"  Price: ${discounted:.2f}"
                + (f" (Save {p['discount']}%)" if p["discount"] else "")
                + f"\n  Rating: ⭐ {p['rating']}/5\n"
                f"  Stock: {'In Stock ✅' if p['in_stock'] else 'Out of Stock ❌'}\n"
                f"  Features:\n"
                + "\n".join(f"    • {f}" for f in p["features"])
                + f"\n  Estimated delivery: 3-5 business days"
            )
    return f"Product {product_id} not found."


@tool
def add_to_cart(product_id: str, quantity: int = 1) -> str:
    """Add a product to the shopping cart."""
    for p in PRODUCT_CATALOG:
        if p["id"] == product_id.upper():
            if not p["in_stock"]:
                return f"Sorry, {p['name']} is currently out of stock."
            price = p["price"] * (1 - p["discount"] / 100) if p["discount"] else p["price"]
            cart_item = {**p, "quantity": quantity, "final_price": price}
            # Update quantity if already in cart
            for item in _user_cart:
                if item["id"] == product_id.upper():
                    item["quantity"] += quantity
                    return f"✅ Updated cart: {p['name']} x{item['quantity']}"
            _user_cart.append(cart_item)
            total = sum(i["final_price"] * i["quantity"] for i in _user_cart)
            return (
                f"✅ Added to cart: {p['name']} x{quantity} (${price:.2f} each)\n"
                f"   Cart total: ${total:.2f} ({len(_user_cart)} items)"
            )
    return f"Product {product_id} not found."


@tool
def view_cart() -> str:
    """View all items in the shopping cart with totals."""
    if not _user_cart:
        return "Your cart is empty. Search for products to add!"
    result = "🛒 Your Shopping Cart:\n" + "=" * 40 + "\n"
    subtotal = 0
    for item in _user_cart:
        line_total = item["final_price"] * item["quantity"]
        subtotal += line_total
        result += (
            f"  [{item['id']}] {item['name']}\n"
            f"       ${item['final_price']:.2f} x {item['quantity']} = ${line_total:.2f}\n"
        )
    tax = subtotal * 0.08
    shipping = 0 if subtotal >= 50 else 9.99
    total = subtotal + tax + shipping

    result += (
        "=" * 40 + "\n"
        f"  Subtotal: ${subtotal:.2f}\n"
        f"  Tax (8%): ${tax:.2f}\n"
        f"  Shipping: {'FREE' if shipping == 0 else f'${shipping:.2f}'}\n"
        f"  TOTAL: ${total:.2f}\n"
    )
    if shipping > 0:
        needed = 50 - subtotal
        result += f"  💡 Add ${needed:.2f} more for FREE shipping!\n"
    return result


@tool
def get_recommendations(budget: Optional[float] = None, interests: str = "") -> str:
    """Get personalized product recommendations based on budget and interests."""
    recommendations = []
    interest_list = [i.strip().lower() for i in interests.split(",") if i.strip()]

    # Filter and score products
    for product in PRODUCT_CATALOG:
        if not product["in_stock"]:
            continue
        price = product["price"] * (1 - product["discount"] / 100)
        if budget and price > budget:
            continue

        score = product["rating"]
        if product["discount"] >= 15:
            score += 0.5  # Boost discounted items
        if product["id"] in _view_history:
            score += 0.3  # Boost previously viewed

        for interest in interest_list:
            if interest in product["category"] or any(interest in f.lower() for f in product["features"]):
                score += 1.0

        recommendations.append((score, product, price))

    recommendations.sort(reverse=True)
    top_recs = recommendations[:5]

    if not top_recs:
        return "No recommendations match your criteria."

    result = "🎯 Personalized Recommendations for You:\n\n"
    for i, (score, p, price) in enumerate(top_recs, 1):
        deal = f" 🔥 {p['discount']}% OFF!" if p["discount"] else ""
        result += (
            f"{i}. {p['name']}{deal}\n"
            f"   ${price:.2f} | ⭐ {p['rating']}/5 | {p['brand']}\n"
            f"   {', '.join(p['features'][:2])}\n\n"
        )
    return result


tools = [search_products, get_product_details, add_to_cart, view_cart, get_recommendations]

SYSTEM_PROMPT = """You are a knowledgeable and helpful e-commerce shopping assistant.

Your role:
- Help customers find the perfect products for their needs
- Use search_products to find relevant items
- Use get_product_details for specific product information
- Use add_to_cart when customers want to purchase
- Use view_cart to show cart status and totals
- Use get_recommendations for personalized suggestions

Be conversational, helpful, and proactive. Highlight deals and savings.
Ask clarifying questions to understand needs better. Mention when items are
on sale or low in stock. Always suggest complementary products."""


def build_ecommerce_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5).bind_tools(tools)

    def shopper(state: EcommerceState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        return {"messages": [llm.invoke(messages)]}

    tool_node = ToolNode(tools)
    graph = StateGraph(EcommerceState)
    graph.add_node("shopper", shopper)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "shopper")
    graph.add_conditional_edges("shopper", tools_condition)
    graph.add_edge("tools", "shopper")
    return graph.compile()


def main():
    app = build_ecommerce_graph()
    state: EcommerceState = {
        "messages": [],
        "user_id": "GUEST-001",
        "budget": None,
        "preferences": [],
    }

    print("🛍️  E-Commerce Shopping Agent")
    print("=" * 60)
    print("Welcome! I'm your personal shopping assistant.")
    print("Try: 'I need headphones under $300' or 'Show me electronics deals'")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            if _user_cart:
                final = app.invoke(
                    {**state, "messages": state["messages"] + [HumanMessage(content="Show my cart")]}
                )
                print(f"\nAgent: {final['messages'][-1].content}")
            print("Agent: Thanks for shopping! Have a great day! 🛍️")
            break
        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(state)
        state["messages"] = result["messages"]
        print(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
