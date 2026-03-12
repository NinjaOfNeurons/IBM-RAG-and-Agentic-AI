
# Build Effective AI Tools for Advanced LLMs

## 1. LLM vs Agent

### Large Language Model (LLM)
- Understands and generates human language.
- Performs **text prediction** based on input prompts.
- Limited to the knowledge available during training unless connected to tools.

### Agent
- An **LLM with additional capabilities**.
- Can **use tools to interact with external systems**.
- Performs **goal-driven actions and multi-step reasoning**.

**Key Difference**
- LLM → Language understanding and generation.
- Agent → Decision-making + tool usage.

---

# 2. What is a Tool?

A **tool** is a function that allows an LLM to perform actions beyond text generation.

### Tools Enable LLMs To
- Access **live data** (news, APIs, databases)
- Perform **actions** (send emails, trigger workflows)
- Ensure **mathematical/logical accuracy**
- Retrieve **private or enterprise data**
- Perform **multi-step reasoning**

**Result:**  
A passive LLM becomes an **active, goal-driven system (agent).**

---

# 3. Tool Calling Workflow

Example: *What is 3 + 2?*

1. User sends query.
2. LLM identifies the **appropriate tool**.
3. LLM extracts parameters (`3`, `2`).
4. Tool receives structured inputs.
5. Tool executes the operation.
6. Tool returns result (`5`).

---

# 4. Key Components of a Tool

### 1. Descriptive Name
Use clear names.

Example:
```

add_numbers

```

### 2. Standardized Inputs
- Usually **strings** or **JSON**
- Must clearly define input types.

Example:
```

inputs: str

```

### 3. Documentation (Docstring)
A good docstring includes:
- Purpose
- Parameters
- Outputs
- Examples
- Limitations

### 4. Function Body
Contains the logic that processes the input.

Example tasks:
- Extract numbers
- Convert strings → integers
- Calculate sum

### 5. Consistent Output
Usually returns structured data like:

```

{
"result": 5
}

```

---

# 5. Importance of Good Docstrings

Docstrings help the LLM:
- Understand **what the tool does**
- Know **how to call it**
- Know **expected inputs/outputs**

A strong docstring contains:

- Tool description
- Parameter explanations
- Input/output format
- Examples

---

# 6. Creating Tools in LangChain

## Tool Class

LangChain can wrap a Python function into a tool.

Example structure:

```

Tool(
name="add_numbers",
func=add_numbers,
description="Adds numbers from a text input"
)

```

### Calling the Tool

```

tool.invoke("What is the sum of 10, 20, and 30?")

```

Output:

```

60

```

---

# 7. Limitations of Simple Tools

Example limitation:

If input contains:

```

ten, 20, 30

```

The tool may only detect digits (`20`, `30`) and output:

```

50

```

Because:
- Basic tools may only parse numeric digits.

---

# 8. Tool Decorator (Modern LangChain)

LangChain now supports the **@tool decorator**.

Benefits:
- Cleaner syntax
- Automatic structured tool creation
- Better support for **function calling models**

Example:

```

@tool
def add_numbers(inputs: str):

```

This creates a **structured tool automatically**.

---

# 9. Structured Tools

Structured tools allow **multiple typed inputs**.

Example parameters:

```

numbers: List[float]
absolute: bool

```

### Advantages
- Supports multiple parameters
- Handles structured JSON inputs
- More flexible for complex tasks

Example input:

```

{
"numbers": [1.2, -2.3, 3.4],
"absolute": true
}

```

---

# 10. Example Tool with Options

```

add_numbers_with_options

```

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| numbers | List[float] | numbers to sum |
| absolute | bool | sum absolute values |

### Behavior

| Input | Output |
|------|------|
| absolute = false | normal sum |
| absolute = true | sum of absolute values |

Example:

```

numbers = [-1.2, -2.0, -3.0]

```

Result:

- absolute=false → `-6.2`
- absolute=true → `6.2`

---

# 11. Complex Tool Outputs

Tools can return **variable outputs**.

Example return type:

```

Dict[str, Union[float, str]]

```

Meaning:
- Output is a dictionary
- Value can be:
  - a float (successful result)
  - a string (error message)

Example outputs:

Success:

```

{
"result": 10.5
}

```

Error:

```

{
"result": "No numbers found"
}

```

---

# 12. Important Implementation Notes

### Debugging Tools
- Debugging LangChain tools can be **challenging**.
- Testing is essential.

### Version Control
LangChain evolves quickly → keep track of versions.

### LLM Compatibility
Some models:
- Cannot handle **multiple inputs**
- Cannot parse **complex outputs**

Testing is required.

---

# 13. Key Takeaways

### Tools
- Callable functions triggered by LLMs.
- Allow interaction with external systems.

### Agents
- LLM + tools + decision-making logic.

### Best Practices

1. Use **clear tool names**.
2. Provide **strong docstrings**.
3. Define **structured inputs**.
4. Return **consistent outputs**.
5. Test across **different LLMs**.

---

# 14. Overall Workflow

```

User Query
↓
LLM analyzes intent
↓
LLM selects tool
↓
Inputs extracted
↓
Tool executed
↓
Structured output returned
↓
LLM forms final response

```
