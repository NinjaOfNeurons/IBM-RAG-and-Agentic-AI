# Evaluator–Optimizer Design Pattern – Revision Notes

## 1. Overview

The **Evaluator–Optimizer Pattern** is used when an AI system must **iteratively improve its output until it meets specific criteria**.

The workflow involves two main components:

1. **Generator (Optimizer)** → creates or improves outputs  
2. **Evaluator** → reviews outputs and provides feedback

If the result does not meet the required standard, **feedback is sent back to the generator**, creating a **reflection loop**.

---

# 2. Basic Workflow

```

User Query
↓
Generator (LLM)
↓
Evaluator (LLM)
↓
Decision
↓      ↓
Accepted  Rejected
↓
Feedback Loop
↓
Generator

```id="eo_flow"

The process repeats **until the output satisfies the target criteria or iteration limits are reached**.

---

# 3. Example Use Case: AI Investment Advisor

The system builds an **investment strategy** based on an **investor profile**.

Multiple **LLM personas** simulate real-world investors:

| Persona | Role |
|---|---|
| Cathie Wood | Generates high-risk innovation strategies |
| Warren Buffett | Evaluates risk and provides feedback |
| Ray Dalio | Refines the strategy using feedback |

---

# 4. State Variables

State variables track the workflow data.

| Variable | Purpose |
|---|---|
| investor_profile | Input describing investor preferences |
| investment_plan | Generated investment strategy |
| target_grade | Desired risk level |
| grade | Evaluator's risk score |
| feedback | Suggestions for improvement |
| iteration | Tracks number of refinement cycles |

---

# 5. Risk Grading Node

### Purpose
Determine **target investment risk level** based on investor profile.

### Components

**Grade Prompt**
- Instructions for the LLM to assign risk level.

Risk levels:

- Ultra Conservative
- Conservative
- Moderate
- Aggressive
- High Risk

---

### Workflow

```

Investor Profile
↓
Grade LLM
↓
Target Risk Grade

```id="riskgrade"

The result is stored in:

```

target_grade

```id="riskgradevar"

---

# 6. Generator Node

The **generator creates the investment plan**.

It has two stages.

---

## 6.1 Initial Generator (Cathie Wood)

### Characteristics

- High-risk investor
- Focus on innovation
- Aggressive growth strategies

### Input

- Investor profile
- Target risk grade

### Output

```

investment_plan

```id="genplan"

---

## 6.2 Revised Generator (Ray Dalio)

Used when the evaluator rejects the plan.

### Inputs

- Investor profile
- Evaluator feedback
- Previous plan

### Role

Refine the strategy while aligning with the target risk grade.

---

# 7. Evaluator Node

The evaluator reviews the investment strategy.

### Persona: Warren Buffett

Characteristics:

- Conservative investor
- Focus on fundamentals
- Emphasizes capital preservation

---

### Evaluation Criteria

- Risk tolerance alignment
- Investment fundamentals
- Capital preservation

---

### Structured Output Schema

| Field | Description |
|---|---|
| grade | Risk classification |
| feedback | Explanation for grading |

Example output:

```

grade: Conservative
feedback: Portfolio includes several speculative assets

```id="evalschema"

---

# 8. Evaluation Logic

The evaluator node performs three tasks:

1. **Assess investment plan**
2. **Generate risk grade**
3. **Provide feedback**

The iteration counter increases with each evaluation.

---

# 9. Routing Logic

A routing function decides whether the workflow continues.

### Conditions

| Condition | Result |
|---|---|
| Grade = Target Grade | Accept plan |
| Grade ≠ Target Grade | Send feedback to generator |
| Iteration limit reached | Stop process |

---

### Decision Flow

```

Evaluator Result
↓
Compare Grade with Target
↓
Match? → Accept
No Match → Send Feedback → Generator

```id="routinglogic"

---

# 10. Reflection Loop

The reflection loop repeatedly improves the investment plan.

```

Generator
↓
Evaluator
↓
Feedback
↓
Refined Generator
↓
Evaluator

```id="reflectionloop"

The process continues **until the strategy matches the required risk profile**.

---

# 11. LangGraph Workflow Structure

Nodes used in the graph:

| Node | Role |
|---|---|
| risk_grader | Determines target risk level |
| generator | Creates investment strategy |
| evaluator | Grades the strategy |
| router | Determines next step |

---

### Graph Structure

```

Start
↓
Risk Grader
↓
Generator
↓
Evaluator
↓
Router
↓      ↓
Accept   Revise
↓
Generator

```id="graphstructure"

---

# 12. Iteration Control

To prevent infinite loops, the system tracks:

```

iteration_count

```id="itercount"

If the maximum iteration threshold is reached:

- The workflow stops
- The best available plan is returned.

---

# 13. Advantages of Evaluator–Optimizer Pattern

- Improves output quality
- Enables self-refinement
- Supports structured feedback
- Simulates expert review processes

---

# 14. Common Use Cases

| Use Case | Example |
|---|---|
| Investment planning | Risk-adjusted portfolios |
| Code improvement | Automated debugging |
| Writing refinement | Essay revisions |
| Product design | Iterative optimization |

---

# 15. Key Concepts

### Generator
Creates or refines outputs.

### Evaluator
Reviews outputs and provides structured feedback.

### Reflection Loop
Iterative improvement cycle.

### State Variables
Track workflow data and progress.

---

# 16. Key Takeaways

- The **Evaluator–Optimizer Pattern** refines outputs through iterative evaluation.
- **Generators produce solutions**, while **evaluators assess quality and provide feedback**.
- **Reflection loops** continue until the result meets the desired criteria.
- **LangGraph** connects grading, generation, evaluation, and routing nodes into a complete workflow.
