# CrewAI Meal Planning System — Notes

## Overview

This project demonstrates how to build a **multi-agent meal planning system** using **CrewAI**.

Key concepts used:

* **CrewAI Agents and Tasks**
* **Structured Outputs with Pydantic**
* **YAML configuration**
* **@CrewBase class**
* **Sequential multi-agent workflow**
* **Shared LLM**
* **External tools (Serper Web Search)**

The system automates meal planning, grocery shopping organization, budgeting, and leftover management.

---

# Crew Structure

The system uses **five specialized agents**:

| Agent              | Responsibility                                      |
| ------------------ | --------------------------------------------------- |
| Meal Planner       | Finds recipes that meet dietary needs and budget    |
| Shopping Organizer | Converts ingredients into a structured grocery list |
| Budget Advisor     | Ensures the meal plan stays within budget           |
| Leftovers Manager  | Suggests how to reuse leftover ingredients          |
| Summary Agent      | Combines everything into a final guide              |

### Workflow

Tasks run **sequentially**.

```
Meal Planner
     ↓
Shopping Organizer
     ↓
Budget Advisor
     ↓
Leftovers Manager
     ↓
Summary Agent
```

Each task’s **output becomes the input for the next task**.

---

# Shared LLM

All agents use a **shared LLM**.

Example:

* **Granite model**
* Hosted on **WatsonX**

Purpose:

* reasoning
* generating responses
* completing tasks

---

# Structured Outputs with Pydantic

Pydantic models ensure:

* clean data
* validation
* consistent structure
* easy data exchange between agents

---

# Pydantic Data Models

## 1. GroceryItem

Represents a single grocery item.

```python
class GroceryItem(BaseModel):
    name: str
    quantity: str
    estimated_price: float
    store_category: str
```

Example:

```
{
  "name": "Chicken Breast",
  "quantity": "2 lbs",
  "estimated_price": 10.50,
  "store_category": "Meat"
}
```

---

## 2. MealPlan

Represents a meal recipe.

```python
class MealPlan(BaseModel):
    meal_name: str
    cooking_difficulty: str
    servings: int
    ingredients: List[GroceryItem]
```

Example:

```
Chicken Stir Fry
Serves: 4
Ingredients: 6 items
```

---

## 3. ShoppingCategory

Groups grocery items by store section.

```python
class ShoppingCategory(BaseModel):
    section_name: str
    items: List[GroceryItem]
    total_estimated_cost: float
```

Example sections:

* Produce
* Dairy
* Meat
* Frozen Foods

---

## 4. GroceryShoppingPlan (Final Data Model)

Combines all components.

```python
class GroceryShoppingPlan(BaseModel):
    total_budget: float
    meal_plans: List[MealPlan]
    shopping_categories: List[ShoppingCategory]
    shopping_tips: List[str]
```

Purpose:

* track meals
* organize grocery list
* track spending
* provide shopping tips

---

# Agents in Python

Agents are defined using:

* role
* goal
* backstory
* LLM

Example:

```
Agent(
  role="Meal Planner",
  goal="Find recipes that match dietary needs and budget",
  backstory="Expert chef and nutrition planner",
  llm=shared_llm
)
```

---

# Tasks

Each agent receives a **task**.

Task properties:

* description
* expected output
* assigned agent
* context
* output format

Example:

```
Task(
  description="Generate a meal plan",
  expected_output="Structured meal plan",
  agent=meal_planner,
  output_pydantic=MealPlan
)
```

---

# External Tool: Serper Web Search

Used by agents that require **real-time information**.

Examples:

* recipe searches
* ingredient prices

Agents using Serper:

* Meal Planner
* Budget Advisor

---

# Meal Planner Agent

### Responsibilities

* find recipes
* respect dietary needs
* stay within budget

### Features

* Uses **Serper Web Search**
* Outputs structured **MealPlan**

Output stored as:

```
shopping_list.json
```

---

# Shopping Organizer Agent

### Responsibilities

* convert ingredients into grocery list
* group items by store section
* estimate quantities and costs

### Input

```
MealPlan
```

### Output

```
GroceryShoppingPlan
```

Structured nested data.

---

# Budget Advisor Agent

### Responsibilities

* analyze costs
* ensure budget compliance
* suggest savings

### Uses

* Serper Web Search for price estimates

### Output

Markdown file:

```
shopping_guide.md
```

---

# YAML Configuration

Instead of defining everything in Python, some agents are defined in **YAML**.

Advantages:

* easier configuration
* no code changes required
* better maintainability

Example components defined in YAML:

* **Leftovers Agent**
* **Leftovers Task**

---

# CrewBase Class

CrewAI provides a **@CrewBase decorator**.

Purpose:

* load agents and tasks from YAML
* simplify integration

Example structure:

```python
@CrewBase
class LeftoverCrew:
    
    @agent
    def leftover_manager(self):
        ...

    @task
    def leftover_task(self):
        ...
```

CrewBase automatically:

* loads YAML files
* connects them to the Python class

---

# Using CrewBase

1. Create class in a Python file
2. Import it into your notebook
3. Instantiate it

Example:

```python
leftovers_cb = LeftoverCrew(llm=shared_llm)
```

Access components:

```
leftovers_cb.leftover_manager()
leftovers_cb.leftover_task()
```

---

# Summary Agent

Final agent in the pipeline.

### Responsibilities

Combine outputs from:

* Meal Planner
* Shopping Organizer
* Budget Advisor
* Leftovers Manager

### Final Output

A **complete meal planning guide** including:

* recipes
* shopping lists
* budget advice
* leftover ideas

---

# Creating the Crew

All agents and tasks are grouped into a **Crew**.

Example:

```python
complete_grocery_crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.sequential
)
```

---

# Running the Workflow

Start execution with:

```python
crew.kickoff(user_input)
```

Execution flow:

```
User Input
   ↓
Agents Execute Sequentially
   ↓
Final Meal Planning Report
```

---

# Output Formats

Different tasks produce different formats:

| Format           | Usage                        |
| ---------------- | ---------------------------- |
| JSON             | Structured shopping lists    |
| Markdown         | User-friendly guides         |
| Pydantic objects | Internal agent communication |

---

# Key Takeaways

### CrewAI

* Builds **multi-agent AI workflows**
* Agents have **roles, goals, and tasks**

### Shared LLM

* Provides consistent reasoning across agents

### Pydantic

Ensures:

* structured outputs
* validated data
* reliable agent communication

### Serper Tool

Provides:

* real-time recipes
* price data

### YAML

* separates configuration from code
* easier updates

### CrewBase

* automatically loads YAML agents/tasks
* simplifies multi-agent architecture

### Final Result

An **automated meal planning system** that produces:

* recipes
* organized shopping lists
* budget analysis
* leftover suggestions
* complete meal planning guide

