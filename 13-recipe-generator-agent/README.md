# 👨‍🍳 Recipe Generator Agent

An intelligent recipe creation agent built with **LangGraph** that generates personalized recipes based on available ingredients, dietary restrictions, cuisine preferences, and cooking skill level, complete with nutritional information.

## Features

- 🥘 **Ingredient-based recipes**: Creates recipes from what you have
- 🥦 **Dietary adaptations**: Vegan, gluten-free, dairy-free substitutions
- 🍽️ **Cuisine matching**: Mediterranean, Asian, Mexican, Italian, Indian styles
- 📊 **Nutritional analysis**: Calories, protein, fat, carbs per serving
- ⏱️ **Time estimates**: Cooking time based on method and dish type
- 👨‍🍳 **Skill-appropriate**: Adjusts complexity for beginner/intermediate/advanced

## Architecture

```
Ingredients + Preferences → Chef Agent → [Compatibility Check] → [Nutrition Calc]
                                ↓
                     [Dietary Substitutions] → [Time Estimate]
                                ↓
                   Complete Recipe with Instructions
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Tools**: Ingredient Compatibility Checker, Nutrition Calculator, Substitution Guide, Time Estimator
- **Pattern**: Tool-augmented creative agent

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
from agent import generate_recipe

recipe = generate_recipe(
    ingredients=["tofu", "broccoli", "ginger", "soy sauce", "sesame oil", "rice"],
    dietary_restrictions=["vegan", "gluten-free"],
    cuisine="Asian",
    skill_level="beginner",
    servings=2
)
print(recipe)
```

## Dietary Substitutions

| Original | Vegan | Gluten-Free | Dairy-Free |
|----------|-------|-------------|------------|
| Eggs | Flax egg | ✅ (eggs are GF) | ✅ |
| Milk | Oat/almond milk | ✅ | Oat/almond milk |
| Pasta | ✅ | Rice pasta | ✅ |
| Flour | ✅ | Almond/rice flour | ✅ |
| Butter | Coconut oil | ✅ | Coconut oil |

## Nutrition Database

The agent has built-in nutrition data (per 100g) for 15 common ingredients including chicken, salmon, rice, pasta, eggs, various vegetables, and pantry staples.

## Output Format

Each recipe includes:
1. **Recipe Name** with brief description
2. **Ingredients List** with precise quantities
3. **Step-by-Step Instructions** numbered and clear
4. **Nutritional Information** per serving
5. **Chef Tips** for best results
6. **Variations** for different tastes
7. **Storage Instructions**

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
