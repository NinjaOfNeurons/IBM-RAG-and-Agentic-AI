

# AI Agents & Compound AI Systems – Notes

## 1. Introduction to AI Agents

AI agents are **systems where a Large Language Model (LLM) is responsible for reasoning, planning, and deciding actions** to solve complex tasks.

They extend simple AI models by combining them with:

* Tools
* Databases
* External programs
* Memory
* Control logic

This creates **compound AI systems** that are more powerful and flexible than standalone models.

---

# 2. Shift in Generative AI

Generative AI is evolving from:

### Monolithic Models

A **single model performing tasks independently**.

Limitations:

* Only knows what it was trained on
* Cannot access private or real-time data
* Hard to adapt without retraining
* Limited problem-solving ability

Example:
If a model is asked:

> "How many vacation days do I have left?"

The model **cannot answer correctly** because it doesn't have access to personal data.

---

### Compound AI Systems

A **system built around models with multiple components**.

Components may include:

* Databases
* APIs
* Search tools
* External programs
* Verification systems

Example workflow:

1. User asks: *How many vacation days do I have left?*
2. LLM converts the question into a **database query**
3. Query retrieves data from vacation database
4. Result returns to LLM
5. LLM generates the final response

Final Answer Example:

> "You have 10 vacation days left."

---

# 3. Characteristics of Compound AI Systems

## Modular Architecture

Systems contain multiple components:

### AI Models

Examples:

* Large Language Models (LLMs)
* Tuned models
* Image generation models

### Programmatic Components

Examples:

* Query decomposition programs
* Output verification
* Database search systems
* External tools
* APIs

This modular design allows:

* Faster development
* Easier updates
* Better scalability

---

# 4. Control Logic in AI Systems

### Programmatic Control Logic

The **developer defines fixed paths** for the system.

Example:
A RAG system that **always searches a vacation database**.

Problem:
If the user asks:

> "What's the weather tomorrow?"

The system fails because it **only searches the vacation database**.

---

# 5. Retrieval-Augmented Generation (RAG)

RAG is one of the most common **compound AI architectures**.

Basic pipeline:

1. User query
2. Search relevant documents
3. Retrieve information
4. Provide context to LLM
5. Generate final answer

Limitation:

* Fixed workflow
* Cannot adapt to unrelated queries

---

# 6. Agentic AI Systems

Instead of **programmed logic**, the **LLM controls the system behavior**.

This is possible due to improved **reasoning abilities of modern LLMs**.

### Key Idea

Let the model:

* Plan
* Decide actions
* Use tools
* Adapt strategies

---

# 7. Fast Thinking vs Slow Thinking

### Fast Thinking (Programmatic Systems)

* Fixed workflow
* Immediate response
* No reasoning steps

Example:

```
Query → Search Database → Return Result
```

---

### Slow Thinking (Agentic Systems)

* Breaks problems into steps
* Creates a plan
* Uses tools
* Adjusts strategy

Example process:

```
Understand problem
→ Create plan
→ Execute steps
→ Evaluate results
→ Adjust if needed
→ Produce final answer
```

---

# 8. Core Components of AI Agents

## 1. Reasoning

The LLM **analyzes and plans how to solve the problem**.

Capabilities:

* Break problems into steps
* Decide strategy
* Evaluate intermediate results

---

## 2. Acting (Tool Use)

Agents interact with **external tools**.

Tools can include:

* Web search
* Database queries
* Calculators
* APIs
* Code execution
* Other AI models

Examples:

* Weather API
* Translation model
* Mathematical calculator

---

## 3. Memory

Agents can store and access information.

### Types of Memory

#### Short-term memory

* Internal reasoning logs
* Previous steps

#### Long-term memory

* User conversation history
* Personal preferences
* Past queries

This allows **personalized interactions**.

---

# 9. ReAct Agent Framework

One popular method for building agents is **ReAct**.

ReAct stands for:

**Reasoning + Acting**

Workflow:

1. User query
2. LLM analyzes problem
3. LLM creates plan
4. LLM calls tools
5. Observe tool output
6. Adjust plan if necessary
7. Produce final answer

Loop continues until the problem is solved.

---

# 10. Example: Vacation Sunscreen Planning

User query:

> "How many two-ounce sunscreen bottles should I bring for my Florida vacation?"

Agent reasoning steps may include:

### Step 1 – Retrieve Vacation Days

Check memory or database.

### Step 2 – Estimate Sun Exposure

Use weather forecast tools.

### Step 3 – Find Recommended Sunscreen Usage

Query public health sources.

### Step 4 – Perform Calculation

Use calculator tool.

### Step 5 – Convert to Bottle Quantity

Determine number of **2-oz bottles needed**.

This demonstrates **multi-step reasoning with tool usage**.

---

# 11. Advantages of Agentic Systems

* Can solve **complex multi-step problems**
* Dynamic decision making
* Flexible workflows
* Can interact with multiple tools
* Adaptable to different tasks

---

# 12. When NOT to Use Agents

Agentic systems are **not always optimal**.

For **narrow and predictable problems**, programmatic systems are better.

Example:
A vacation policy query system.

Advantages of programmatic systems:

* Faster
* Cheaper
* More predictable
* Less looping

---

# 13. When to Use Agents

Agents are useful for:

* Complex workflows
* Open-ended tasks
* Multi-tool systems
* Dynamic problem solving

Examples:

* Autonomous coding assistants
* Research agents
* GitHub issue solvers
* Task automation systems

---

# 14. Sliding Scale of LLM Autonomy

AI systems exist on a **spectrum of autonomy**.

| Type                  | Control              | Example            |
| --------------------- | -------------------- | ------------------ |
| Programmatic Systems  | Developer controlled | RAG                |
| Hybrid Systems        | Partial autonomy     | Tool-augmented LLM |
| Fully Agentic Systems | LLM controlled       | Autonomous agents  |

Developers choose **the appropriate level of autonomy**.

---

# 15. Current State of Agentic AI

Agentic AI is still **in early stages**, but improving rapidly due to:

* Better reasoning models
* Improved tool integration
* Advanced system design

Most real-world systems still include:

### Human-in-the-loop

Humans verify outputs and control critical decisions.
