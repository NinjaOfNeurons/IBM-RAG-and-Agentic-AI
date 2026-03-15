# ReAct Agents – Building Agents that Reason Before Acting

## 1. Overview

* **ReAct (Reason + Act)** agents combine:

  * **Step-by-step reasoning**
  * **Tool usage**
* Designed for **complex tasks requiring external data**.
* Implemented using **LangGraph** to orchestrate reasoning and tool calls.

---

# 2. Core Idea of ReAct

Agents reason and act in a loop until they reach a final answer.

**Reasoning Pattern**

```
Thought → Action → Action Input → Observation → Final Answer
```

### Components

| Component        | Description                     |
| ---------------- | ------------------------------- |
| **Thought**      | Agent reasoning about next step |
| **Action**       | Tool the agent decides to use   |
| **Action Input** | Input provided to the tool      |
| **Observation**  | Tool result                     |
| **Final Answer** | Final response to the user      |

---

# 3. Example Workflow

### User Query

> "What's the weather in Tokyo and what should I wear?"

### Step-by-step reasoning

1. **Thought**
   Need to check Tokyo weather.

2. **Action**
   `search_tool`

3. **Action Input**
   `"Tokyo weather today"`

4. **Observation**
   `22°C, sunny`

5. **Thought**
   Recommend clothing based on weather.

6. **Action**
   `recommend_clothing`

7. **Action Input**
   `"22°C sunny weather"`

8. **Observation**
   `t-shirt, shorts, sunglasses`

9. **Final Answer**

   * Tokyo is **22°C and sunny**
   * Wear **light clothing (t-shirt, shorts, sunglasses)**

---

# 4. ReAct Agent Architecture

### Components

**1. LLM (Reasoning Engine)**

* Generates thoughts
* Decides tool actions

**2. Tools**
Examples:

* `search_tool` → weather lookup
* `recommend_clothing` → outfit suggestions

**3. Environment**

* External information accessed via tools.

---

# 5. Implementation in LangGraph

## Step 1 – Define Tools

Example tools:

* `search_tool` → uses Tavily search.
* `recommend_clothing` → suggests clothing based on keywords.

---

## Step 2 – Define Agent State

Agent state stores conversation history.

```
messages: Sequence[BaseMessage]
```

Message types:

* `HumanMessage`
* `AIMessage`
* `ToolMessage`

Each node appends messages using **add_messages**.

---

## Step 3 – Create Prompt

System prompt instructs the agent to:

* Think **step-by-step**
* Use **tools when needed**
* Produce structured reasoning

The **agent_scratchpad** stores:

```
Thought
Action
Action Input
Observation
```

---

## Step 4 – Build Agent

Steps:

1. Load GPT model.
2. Bind tools to the model.
3. Chain prompt → model → tools.

---

# 6. LangGraph Workflow

### Nodes

| Node      | Purpose             |
| --------- | ------------------- |
| **agent** | Runs LLM reasoning  |
| **tools** | Executes tool calls |

---

### Graph Flow

```
User Input
     ↓
Agent Node (LLM reasoning)
     ↓
Should Continue?
  ↙        ↘
Tools      End
  ↓
Agent
```

---

# 7. Conditional Flow Control

`should_continue()` checks:

* If **AI message contains tool call**

  * → continue to **tools node**
* If **no tool calls**

  * → go to **END**

---

# 8. Execution Example

User query:

```
What's the weather in Zurich?
What should I wear?
```

### Process

1. Human message added to state.
2. LLM calls **search tool**.
3. Tool returns weather.
4. LLM calls **clothing tool**.
5. Tool returns recommendation.
6. LLM generates **final answer**.

---

# 9. Key Takeaways

* **ReAct agents combine reasoning and action**.
* They follow a structured loop:

```
Thought → Action → Action Input → Observation
```

* **Tool outputs guide future reasoning**.
* **LangGraph** orchestrates the workflow through nodes and conditional routing.
* The loop continues until **no more tool calls are required**.

---
