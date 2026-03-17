# Orchestrator Design Pattern – Revision Notes

## 1. Overview

The **Orchestrator Design Pattern** enables **dynamic AI workflows** where the system determines:

- How many tasks are needed
- Which agents should perform them
- How tasks run in parallel
- How results are combined

Unlike earlier patterns, **workflow complexity does NOT need to be known beforehand**.

---

# 2. Static vs Dynamic Workflows

| Workflow Type | Description |
|---|---|
| Static workflows | Structure defined in advance |
| Dynamic workflows | Tasks generated at runtime |

Earlier patterns (Sequential, Routing, Parallel) are **static**.

The **Orchestrator Pattern is dynamic** because it decides **how many worker agents are needed during execution**.

---

# 3. Real-World Analogy

### Cruise Ship Party Planner

1. Guests request multiple dishes
2. Party planner analyzes request
3. Assigns chefs specializing in each cuisine
4. Chefs cook dishes in parallel
5. Head chef combines results into a menu

Mapping to AI:

| Real World | AI System |
|---|---|
| Party planner | Orchestrator |
| Chefs | Worker agents |
| Dish preparation | Task execution |
| Final dinner guide | Synthesized output |

---

# 4. Orchestrator Architecture

Main components:

1. **Planner (Orchestrator)**
2. **Assign Workers Node**
3. **Worker Agents**
4. **Synthesizer Node**

Workflow:

```

User Request
↓
Orchestrator (Planner)
↓
Assign Workers
↓
Parallel Worker Agents
↓
Synthesizer
↓
Final Output

```id="u9b2o1"

---

# 5. State Variables

State variables store workflow data.

| Variable | Purpose |
|---|---|
| meals | User input request |
| sections | Structured dish objects created by orchestrator |
| completed_menu | Worker outputs (shared list) |
| final_meal_guide | Final formatted output |

Important:

`completed_menu` is **shared between state and worker state**.

---

# 6. Worker State

Workers use a **separate state container**.

### Worker State Contains

- Dish information
- Shared menu list

This allows:

- **Shared context access**
- **Task-specific isolation**

---

# 7. Structured Output with Dish Objects

The orchestrator generates structured objects:

| Field | Description |
|---|---|
| name | Dish name |
| ingredients | Required ingredients |
| location | Cuisine origin |

Example:

```

Dish(
name="Pasta",
ingredients=["pasta", "tomato sauce"],
location="Italy"
)

```id="e9f0z0"

These objects populate the **sections list**.

---

# 8. Orchestrator Node

### Purpose

Breaks user request into structured tasks.

### Process

1. Receive input meal request
2. Use prompt to generate dish objects
3. Output structured task list

Example request:

```

Prepare:

* Italian pasta
* Mexican tacos
* Indian curry
* Thai stir fry
* American burgers

```id="xk4rc7"

Output becomes the **sections list**.

---

# 9. Assign Workers Node

This node dynamically creates workers.

### Function

- Reads `sections` list
- Creates workers for each dish
- Uses **LangGraph `send()` function**

Example:

```

send(section → worker)

```id="7i5gex"

Each worker receives one dish.

Workers run **in parallel**.

---

# 10. Worker Nodes (Chef Workers)

Each worker:

1. Receives dish object
2. Extracts:
   - name
   - cuisine
   - ingredients
3. Generates cooking instructions

Prompt example:

```

You are a chef specializing in {location}.
Explain how to cook {name} using {ingredients}.

```id="q4r8wh"

Output includes:

- chef introduction
- detailed recipe
- step-by-step instructions

---

# 11. Updating Shared State

Workers append outputs to:

```

completed_menu

```id="u8d7p3"

Using:

```

operator.add

```id="7q6o6v"

This builds a list of completed recipes.

---

# 12. Synthesizer Node

The synthesizer combines results.

### Input

`completed_menu`

### Process

- Merge all worker outputs
- Format results
- Produce final guide

### Output

```

final_meal_guide

```id="3j9k4b"

Example result:

```

Dinner Guide

## Italian Pasta Recipe

## Mexican Tacos Recipe

## Indian Curry Recipe

```

---

# 13. Workflow Construction (LangGraph)

Steps to build the graph:

1. Initialize **state graph**
2. Add nodes:
   - orchestrator
   - chef_worker
   - synthesizer
3. Add **assign_workers conditional routing**
4. Add edges

Workflow structure:

```

Start → Orchestrator
↓
Assign Workers
↓ ↓ ↓ ↓
Workers (parallel)
↓
Synthesizer
↓
End

```id="y4n0zw"

---

# 14. Execution Flow

Example request:

```

Prepare pasta, tacos, curry, stir fry, burgers

```id="m92rf8"

Execution steps:

1. Orchestrator creates dish objects
2. Assign node distributes tasks
3. Workers generate recipes
4. Outputs stored in `completed_menu`
5. Synthesizer merges results
6. Final meal guide returned

---

# 15. Advantages of the Orchestrator Pattern

- Dynamic task creation
- Scalable agent coordination
- Parallel processing
- Flexible workflow execution
- Handles unknown complexity

---

# 16. Key Concepts

### Orchestrator
Breaks tasks into smaller pieces.

### Workers
Execute specialized tasks.

### Shared State
Stores common workflow data.

### Worker State
Stores task-specific information.

### Synthesizer
Combines outputs into final result.

---

# 17. Key Takeaways

- The **Orchestrator Pattern** manages **dynamic workflows**.
- A central orchestrator **assigns tasks to worker agents**.
- Workers execute tasks **in parallel**.
- **State and worker state** manage shared and task-specific data.
- A **synthesizer node merges outputs** into a final unified result.
```
