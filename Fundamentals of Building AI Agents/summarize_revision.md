

# AI Agents Course – Ultra Short Revision Notes

## 1. Role of Tools in LLMs
- Enable access to **external data sources** (databases, APIs).
- Support **RAG (Retrieval-Augmented Generation)**.
- Process **images, audio, video → multimodal reasoning**.
- Manage **long conversations**.
- Perform **real-world actions via APIs**.

---

# 2. Embedded Tool Calling
- Improves **accuracy**
- Reduces **hallucinations**
- Centralizes tool handling in frameworks (e.g., LangChain).

Tools must define:
- Purpose
- Expected output
- Input/output format

---

# 3. Agent Reasoning Loop
Agent workflow:

1. Receive task  
2. Select tool  
3. Execute tool  
4. Review result  
5. Feed result back  
6. Repeat until **final answer**

---

# 4. Zero-Shot ReAct Agent
- Uses **zero-shot reasoning**
- Solves **unseen tasks**
- Best for **simple or structured problems**

---

# 5. Building Agents with LangChain
Key steps:
- Choose **LLM supporting tools & reasoning**
- Use **JSON-serializable tools**
- Select **agent strategy** based on task complexity

Important function:
- `create_react_agent`

Customization:
- Prompt templates
- Tool lists
- Reasoning behavior

---

# 6. Toolkits
Example: **Math Toolkit**
- Addition
- Subtraction
- Multiplication
- Division

Agents use these tools for calculations.

---

# 7. Agent Interaction
Use:

```

.invoke()

```

Purpose:
- Simulate chat
- Send messages
- Receive structured responses

---

# 8. Manual Invocation
Instead of automatic tool calls:

- Verify **inputs/outputs**
- Adjust actions manually

Benefits:
- Better **control**
- Improved **safety**
- Lower **cost**
- Higher **accuracy**

---

# 9. Pandas DataFrame Agent
Allows **natural language data analysis**.

Agent:
- Generates **Python code**
- Interacts with **Pandas DataFrame**

Supports:
- Filtering
- Aggregation
- Visualization

---

# 10. LCEL (LangChain Expression Language)

Used to structure workflows.

Key features:
- `|` **pipe operator** → sequential flow
- **Prompt templates** with variables `{}`

Components:
- `RunnableSequence` → sequential execution
- `RunnableParallel` → parallel execution

LCEL automatically converts functions using **type coercion**.

---

# 11. Best Practices for Agents
- Start in **sandbox environments**
- Write **clear prompts**
- Validate results with **human expertise**
- **Iteratively refine queries**

---

# 12. AI SQL Agents
Benefits:
- Query databases using **natural language**
- No deep SQL knowledge needed.

Implementation steps:
1. Create **Python virtual environment**
2. Install **LangChain + LLM libraries**
3. Launch **SQL server**
4. Create **database connector**
5. Run **natural language queries**

Agent:
- Converts **NL → SQL**
- Executes query
- Returns results.


