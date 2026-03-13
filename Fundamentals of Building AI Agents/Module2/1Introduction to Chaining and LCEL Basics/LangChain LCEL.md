
# LangChain LCEL (LangChain Expression Language) Chaining Method

## 1. Overview

**LangChain Expression Language (LCEL)** is the modern way to build chains in LangChain.

It uses the **pipe operator (`|`)** to connect components and create a clear **data flow pipeline**.

### Key Advantages
- Clean and readable workflows
- Better composability
- Flexible chain building
- Easier debugging and maintenance
- Supports parallel and sequential execution

LCEL replaces the **older `LLMChain` pattern** with a more modular approach.

---

# 2. What is LCEL?

LCEL is a **pattern for building LangChain applications** that connects components like:

- prompts
- LLMs
- retrievers
- tools
- output parsers

using the **pipe operator (`|`)**.

### Basic Flow

```

Input → Prompt Template → LLM → Output Parser → Result

```

---

# 3. Steps to Build an LCEL Chain

Typical LCEL workflow:

1. Define a **prompt template**
2. Create a **PromptTemplate instance**
3. Connect components using **pipe operator (`|`)**
4. Run the chain using **invoke()**

---

# 4. Prompt Templates

Prompts are defined using **variables inside curly braces `{}`**.

Example template:

```

Tell me a {adjective} joke about {topic}

```

Variables are filled during execution.

Example input:

```

{
"adjective": "funny",
"topic": "cats"
}

```

---

# 5. Runnables in LangChain

**Runnables** are building blocks that connect different components.

Examples of runnable components:

- PromptTemplate
- LLM
- OutputParser
- Functions
- Dictionaries

They enable pipelines that pass outputs from one step to the next.

---

# 6. Runnable Composition Primitives

LangChain provides two core runnable types.

## 1. RunnableSequence

Runs components **sequentially**.

```

Component1 → Component2 → Component3

````

Output of one component becomes input for the next.

---

## 2. RunnableParallel

Runs **multiple components simultaneously** using the same input.

Example tasks:

- summarization
- translation
- sentiment analysis

All run at the same time.

---

# 7. Pipe Operator (`|`)

LCEL replaces `RunnableSequence` with the **pipe operator**.

Example:

```python
chain = prompt | llm | StrOutputParser()
````

Flow:

```
Prompt → LLM → Output Parser
```

Benefits:

* Shorter syntax
* Easier readability
* Clear pipeline structure

---

# 8. Type Coercion in LCEL

LCEL automatically converts certain Python objects into runnables.

This process is called **type coercion**.

| Object     | Converted To     |
| ---------- | ---------------- |
| function   | RunnableLambda   |
| dictionary | RunnableParallel |

This happens automatically.

---

# 9. RunnableLambda

Used to convert a **function into a runnable component**.

Example:

```python
RunnableLambda(format_prompt)
```

Purpose:

* transforms input
* prepares data for the next component

---

# 10. Sequential Chain Example

Example LCEL chain:

```python
chain = RunnableLambda(format_prompt) | llm | StrOutputParser()
```

Execution flow:

1. Input dictionary passed to `format_prompt`
2. Function formats the prompt
3. Prompt sent to LLM
4. LLM generates response
5. Output parser extracts text

---

# 11. Parallel Execution Example

Using a dictionary creates **RunnableParallel** automatically.

Example structure:

```python
{
 "summary": summary_chain,
 "translation": translation_chain,
 "sentiment": sentiment_chain
}
```

All chains receive the **same input**.

### Output

```
{
 "summary": "...",
 "translation": "...",
 "sentiment": "positive"
}
```

---

# 12. Example LCEL Workflow

Example pipeline:

```
Input
  ↓
RunnableLambda (format prompt)
  ↓
LLM
  ↓
StrOutputParser
  ↓
Final Output
```

This pipeline is created using **pipe operators**.

---

# 13. Running an LCEL Chain

Use the `invoke()` method.

Example:

```python
chain.invoke({
 "adjective": "funny",
 "content": "dogs"
})
```

The chain processes the input and returns the response.

---

# 14. LCEL vs Traditional LangChain Chains

| Feature            | Traditional Chains | LCEL              |
| ------------------ | ------------------ | ----------------- |
| Syntax             | More verbose       | Clean pipeline    |
| Composition        | Limited            | Highly composable |
| Readability        | Moderate           | High              |
| Parallel execution | Harder             | Built-in          |
| Recommended        | Older approach     | Modern approach   |

---

# 15. When to Use LCEL

Best suited for:

* prompt pipelines
* simple orchestration
* reusable chain components
* LLM-based workflows

---

# 16. When to Use LangGraph Instead

Use **LangGraph** for more complex workflows:

* multi-agent systems
* conditional branching
* long-running processes
* complex state management

However, **LCEL can still be used inside LangGraph nodes**.

---

# 17. Key Benefits of LCEL

### 1. Parallel Execution

Run multiple tasks simultaneously.

### 2. Async Support

Supports asynchronous execution.

### 3. Simplified Streaming

Allows streaming responses.

### 4. Automatic Tracing

Helps track workflow execution.

### 5. Clean Data Flow

Pipe operator clearly shows component connections.

---

# 18. Key Concepts Recap

### LCEL Pattern

* Uses **pipe operator (`|`)**
* Connects runnable components

---

### Prompt Templates

* Use `{variables}`

Example:

```
Write a {tone} summary of {topic}
```

---

### RunnableSequence

Sequential execution.

```
A → B → C
```

---

### RunnableParallel

Parallel execution.

```
Input → [Task1, Task2, Task3]
```

---

### Type Coercion

Automatic conversion:

```
function → RunnableLambda
dict → RunnableParallel
