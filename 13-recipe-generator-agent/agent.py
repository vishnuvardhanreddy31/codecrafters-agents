"""
Recipe Generator Agent
Uses LangGraph to create personalized recipes based on available ingredients,
dietary restrictions, cuisine preferences, and cooking skill level.
"""

import os
from typing import Annotated, TypedDict, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ── Nutrition database (sample) ────────────────────────────────────────────────
NUTRITION_DATA = {
    "chicken breast": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0},
    "salmon": {"calories": 208, "protein": 20, "fat": 13, "carbs": 0},
    "rice": {"calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28},
    "pasta": {"calories": 131, "protein": 5, "fat": 1.1, "carbs": 25},
    "eggs": {"calories": 155, "protein": 13, "fat": 11, "carbs": 1.1},
    "broccoli": {"calories": 34, "protein": 2.8, "fat": 0.4, "carbs": 7},
    "spinach": {"calories": 23, "protein": 2.9, "fat": 0.4, "carbs": 3.6},
    "tomatoes": {"calories": 18, "protein": 0.9, "fat": 0.2, "carbs": 3.9},
    "olive oil": {"calories": 884, "protein": 0, "fat": 100, "carbs": 0},
    "bread": {"calories": 265, "protein": 9, "fat": 3.2, "carbs": 49},
    "cheese": {"calories": 402, "protein": 25, "fat": 33, "carbs": 1.3},
    "milk": {"calories": 61, "protein": 3.2, "fat": 3.3, "carbs": 4.8},
    "lentils": {"calories": 116, "protein": 9, "fat": 0.4, "carbs": 20},
    "chickpeas": {"calories": 164, "protein": 9, "fat": 2.6, "carbs": 27},
    "quinoa": {"calories": 120, "protein": 4.1, "fat": 1.9, "carbs": 22},
}

DIETARY_SUBSTITUTIONS = {
    "vegan": {
        "chicken": "tofu or tempeh",
        "eggs": "flax egg (1 tbsp flaxseed + 3 tbsp water)",
        "milk": "oat milk or almond milk",
        "cheese": "nutritional yeast or vegan cheese",
        "butter": "coconut oil or vegan butter",
        "honey": "maple syrup or agave",
    },
    "gluten-free": {
        "flour": "almond flour or rice flour",
        "pasta": "rice pasta or zucchini noodles",
        "bread": "gluten-free bread",
        "soy sauce": "tamari",
        "breadcrumbs": "almond meal or crushed rice crackers",
    },
    "dairy-free": {
        "milk": "oat milk, almond milk, or coconut milk",
        "cream": "coconut cream",
        "cheese": "dairy-free cheese",
        "butter": "coconut oil or vegan butter",
        "yogurt": "coconut yogurt or soy yogurt",
    },
}


# ── State ──────────────────────────────────────────────────────────────────────
class RecipeState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    ingredients: List[str]
    dietary_restrictions: List[str]
    cuisine_preference: str
    skill_level: str
    servings: int


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def check_ingredient_compatibility(ingredients: str) -> str:
    """Check if a list of ingredients work well together and suggest pairings.
    ingredients: comma-separated ingredient list."""
    ingredient_list = [i.strip().lower() for i in ingredients.split(",")]

    flavor_profiles = {
        "mediterranean": {"olive oil", "tomatoes", "garlic", "lemon", "oregano", "basil"},
        "asian": {"ginger", "soy sauce", "sesame oil", "garlic", "scallions"},
        "mexican": {"lime", "cilantro", "cumin", "chili", "avocado", "tomatoes"},
        "italian": {"basil", "tomatoes", "garlic", "parmesan", "olive oil", "pasta"},
        "indian": {"turmeric", "cumin", "coriander", "garam masala", "ginger", "garlic"},
    }

    best_match = None
    best_score = 0
    for cuisine, profile in flavor_profiles.items():
        score = len(set(ingredient_list) & profile)
        if score > best_score:
            best_score, best_match = score, cuisine

    result = f"Ingredient Analysis for: {', '.join(ingredient_list)}\n"
    if best_match and best_score > 0:
        result += f"  Best cuisine match: {best_match.title()} (score: {best_score})\n"

    # Check for classic pairings
    classic_pairs = [
        ({"garlic", "olive oil"}, "classic Mediterranean base"),
        ({"eggs", "cheese"}, "great for frittatas/omelettes"),
        ({"chicken", "lemon"}, "classic combination"),
        ({"spinach", "eggs"}, "nutritious protein combo"),
        ({"tomatoes", "basil"}, "Italian classic"),
    ]
    for pair, note in classic_pairs:
        if pair.issubset(set(ingredient_list)):
            result += f"  ✅ {note}\n"

    return result


@tool
def calculate_nutrition(recipe_ingredients: str, servings: int = 4) -> str:
    """Calculate estimated nutritional information per serving for a recipe.
    recipe_ingredients: comma-separated list of ingredients with quantities
    (e.g., '200g chicken breast, 100g rice, 50g broccoli')"""
    import re
    total = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0}

    for item in recipe_ingredients.split(","):
        item = item.strip().lower()
        # Extract quantity in grams
        match = re.search(r'(\d+)\s*g', item)
        grams = int(match.group(1)) if match else 100  # default 100g

        for ingredient, per_100g in NUTRITION_DATA.items():
            if ingredient in item:
                factor = grams / 100
                for nutrient in total:
                    total[nutrient] += per_100g[nutrient] * factor
                break

    per_serving = {k: v / servings for k, v in total.items()}
    return (
        f"Estimated Nutrition per serving ({servings} servings):\n"
        f"  Calories: {per_serving['calories']:.0f} kcal\n"
        f"  Protein: {per_serving['protein']:.1f}g\n"
        f"  Fat: {per_serving['fat']:.1f}g\n"
        f"  Carbohydrates: {per_serving['carbs']:.1f}g"
    )


@tool
def get_dietary_substitutions(ingredient: str, dietary_restriction: str) -> str:
    """Get dietary-appropriate substitutions for an ingredient.
    dietary_restriction: vegan | gluten-free | dairy-free"""
    restriction = dietary_restriction.lower()
    if restriction not in DIETARY_SUBSTITUTIONS:
        return f"Unknown dietary restriction: {dietary_restriction}. Try: vegan, gluten-free, dairy-free"

    ingredient_lower = ingredient.lower()
    subs = DIETARY_SUBSTITUTIONS[restriction]

    for key, substitution in subs.items():
        if key in ingredient_lower or ingredient_lower in key:
            return f"For {dietary_restriction}: Replace '{ingredient}' with {substitution}"

    return f"No specific {dietary_restriction} substitution found for '{ingredient}'. Check with a nutritionist."


@tool
def estimate_cooking_time(dish_type: str, method: str = "stovetop") -> str:
    """Estimate cooking time for a dish based on type and method.
    dish_type: breakfast | lunch | dinner | dessert | snack | soup
    method: stovetop | oven | slow_cooker | air_fryer | instant_pot"""
    time_estimates = {
        ("breakfast", "stovetop"): "10-20 minutes",
        ("breakfast", "oven"): "20-35 minutes",
        ("lunch", "stovetop"): "20-30 minutes",
        ("dinner", "stovetop"): "30-45 minutes",
        ("dinner", "oven"): "45-75 minutes",
        ("dinner", "slow_cooker"): "6-8 hours (low) or 3-4 hours (high)",
        ("soup", "stovetop"): "30-60 minutes",
        ("dessert", "oven"): "25-45 minutes",
        ("snack", "stovetop"): "5-15 minutes",
        ("snack", "air_fryer"): "8-15 minutes",
    }
    key = (dish_type.lower(), method.lower())
    time = time_estimates.get(key, "20-40 minutes (estimate)")
    return f"Estimated cooking time for {dish_type} ({method}): {time}"


tools = [
    check_ingredient_compatibility,
    calculate_nutrition,
    get_dietary_substitutions,
    estimate_cooking_time,
]

SYSTEM_PROMPT = """You are a professional chef and nutritionist helping create
personalized recipes.

When creating recipes:
1. Use check_ingredient_compatibility to validate ingredient combinations
2. Calculate nutrition with calculate_nutrition for the recipe
3. Use get_dietary_substitutions for any dietary restrictions
4. Provide estimate_cooking_time for planning

Always provide:
- Recipe name and brief description
- Ingredient list with quantities
- Step-by-step instructions
- Nutritional information per serving
- Tips and variations
- Storage instructions"""


def build_recipe_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7).bind_tools(tools)

    def chef(state: RecipeState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        return {"messages": [llm.invoke(messages)]}

    tool_node = ToolNode(tools)
    graph = StateGraph(RecipeState)
    graph.add_node("chef", chef)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chef")
    graph.add_conditional_edges("chef", tools_condition)
    graph.add_edge("tools", "chef")
    return graph.compile()


def generate_recipe(
    ingredients: List[str],
    dietary_restrictions: List[str] = None,
    cuisine: str = "any",
    skill_level: str = "intermediate",
    servings: int = 4,
) -> str:
    app = build_recipe_graph()
    dietary_str = ", ".join(dietary_restrictions) if dietary_restrictions else "none"
    prompt = (
        f"Create a delicious recipe using these ingredients: {', '.join(ingredients)}\n"
        f"Dietary restrictions: {dietary_str}\n"
        f"Cuisine preference: {cuisine}\n"
        f"Cooking skill level: {skill_level}\n"
        f"Servings: {servings}\n\n"
        "Use the tools to check ingredient compatibility, calculate nutrition, "
        "handle dietary substitutions if needed, and estimate cooking time. "
        "Then provide a complete, detailed recipe."
    )
    state: RecipeState = {
        "messages": [HumanMessage(content=prompt)],
        "ingredients": ingredients,
        "dietary_restrictions": dietary_restrictions or [],
        "cuisine_preference": cuisine,
        "skill_level": skill_level,
        "servings": servings,
    }
    result = app.invoke(state)
    return result["messages"][-1].content


def main():
    print("👨‍🍳 Recipe Generator Agent\n" + "=" * 60)
    ingredients = ["chicken breast", "broccoli", "garlic", "olive oil", "lemon", "rice"]
    dietary = ["gluten-free"]
    print(f"Ingredients: {', '.join(ingredients)}")
    print(f"Dietary restrictions: {', '.join(dietary)}")
    print(f"Cuisine: Mediterranean | Skill: Beginner | Servings: 4\n")
    recipe = generate_recipe(
        ingredients=ingredients,
        dietary_restrictions=dietary,
        cuisine="Mediterranean",
        skill_level="beginner",
        servings=4,
    )
    print(recipe)
    print("\n✅ Recipe generation complete!")


if __name__ == "__main__":
    main()
