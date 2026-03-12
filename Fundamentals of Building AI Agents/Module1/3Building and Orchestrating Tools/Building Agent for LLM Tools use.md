# Build Intelligent Agents for Dynamic LLM Tool Use

## 1. Overview

This lesson explains how to build **intelligent agents in LangChain** that:

- Use tools
- Perform multi-step reasoning
- Interact with real-world data
- Execute dynamic workflows

Agents combine **LLMs + Tools + Reasoning Strategy**.

---

# 2. What is an Agent?

In **LangChain**, an agent is a system that:

- Uses an **LLM for reasoning**
- Calls **tools to perform actions**
- Decides **what to do next based on results**

### Key Capabilities

Agents can:

- Reason about tasks
- Call tools dynamically
- Handle multi-step workflows
- Generate final answers based on tool outputs

**Difference**

| Component | Role |
|---|---|
| LLM | Generates language and reasoning |
| Tool | Performs specific operations |
| Agent | Decides when and how to use tools |

---

# 3. Key Factors When Building an Agent

## 1. Choice of LLM

Not all LLMs support:

- Tool calling
- Structured inputs
- Multi-step reasoning

Model selection affects **agent capabilities**.

---

## 2. Tool Structure

Tools must have:

- JSON-serializable inputs
- JSON-serializable outputs

Preferred format:

**Structured tools with defined schemas**

Benefits:
- Easier parsing
- More reliable tool calls
- Better compatibility with agents

---

## 3. Agent Strategy

Different agent strategies exist depending on task complexity.

Examples:

| Agent Type | Best For |
|---|---|
| Zero-shot ReAct | Simple reasoning tasks |
| Structured Chat ReAct | Tools with typed inputs |
| OpenAI Functions | Structured tool calling |

---

# 4. LangChain Agent Architecture

An agent consists of:

```

User Input
↓
LLM reasoning
↓
Tool selection
↓
Tool execution
↓
Observation
↓
Next reasoning step
↓
Final answer

```

This process forms a **reasoning loop**.

---

# 5. Agent Reasoning Loop

Agents follow a **ReAct cycle**:

1. **Receive Query**
2. **Reason about the problem**
3. **Choose a tool**
4. **Execute tool**
5. **Observe result**
6. **Plan next step**
7. **Generate final response**

---

# 6. ReAct Framework

**ReAct = Reason + Act**

Steps:

1. Reason about the problem
2. Take action (call tool)
3. Observe tool output
4. Plan next step
5. Repeat until answer is reached

This allows **multi-step decision making**.

---

# 7. Zero-Shot ReAct Agent

A **Zero-Shot ReAct Agent** can solve problems without prior examples.

Characteristics:

- Uses step-by-step reasoning
- Selects tools dynamically
- Works well for simple tasks

Example tool:

```

add_numbers()

```

---

# 8. Creating an Agent with `initialize_agent`

LangChain provides a simple function:

```

initialize_agent()

```

Purpose:
- Quickly combine **LLM + tools + strategy**

Example parameters:

```

initialize_agent(
tools=[add_tool],
llm=llm,
agent="zero-shot-react-description",
verbose=True,
handle_parsing_errors=True
)

```

### Important Parameters

| Parameter | Purpose |
|---|---|
| tools | List of tools agent can use |
| llm | Language model |
| agent | Agent strategy |
| verbose | Shows reasoning steps |
| handle_parsing_errors | Allows recovery from tool formatting issues |

---

# 9. Example Agent Task

User query:

```

In 2023:
US GDP = 27.72T
Canada GDP = 2.14T
Mexico GDP = 1.79T

What is the total?

```

Agent process:

1. LLM recognizes need to **sum numbers**
2. Chooses **AddTool**
3. Sends numbers to tool
4. Tool calculates sum
5. Agent formats final response

Result:

```

Total GDP ≈ $31.55 trillion

```

---

# 10. Calling Agents

### `run()` method

Used for simple agents.

Example:

```

agent.run("Calculate this...")

```

---

### `invoke()` method

Preferred for:

- debugging
- complex agents
- structured inputs

Example:

```

agent.invoke({"input": "calculate total"})

```

Returns structured output:

```

{
"input": "...",
"output": "result"
}

```

---

# 11. Tool Compatibility with Agents

Different agents expect different tool formats.

| Agent Type | Tool Input Format |
|---|---|
| zero-shot-react-description | Plain strings |
| structured-chat-zero-shot-react-description | Typed inputs |
| openai-functions | Structured JSON |

Choosing the wrong combination can cause **parsing errors**.

---

# 12. Structured Tool Example

Tool:

```

add_numbers_with_options()

```

Parameters:

| Parameter | Type | Description |
|---|---|---|
| numbers | List[float] | Numbers to add |
| absolute | bool | Use absolute values |

Example input:

```

{
"numbers": [-3, 2, 4],
"absolute": true
}

```

Result:

```

9

```

---

# 13. Structured Chat ReAct Agent

Agent type:

```

structured-chat-zero-shot-react-description

```

Advantages:

- Supports **multiple inputs**
- Handles **structured tools**
- Easier debugging

---

# 14. Using Different LLMs

Agents can work with multiple models.

Examples:

- IBM Granite (watsonx.ai)
- OpenAI models

Different LLMs support different capabilities.

---

# 15. OpenAI Functions Agent

Agent type:

```

openai-functions

```

Purpose:

- Supports **structured outputs**
- Works well with tools returning **JSON or dictionaries**

Example tool output:

```

{
"result": 10.2
}

```

---

# 16. Handling Complex Tool Outputs

Some tools return complex structures:

Example:

```

Dict[str, Union[float, str]]

```

Certain LLM-agent combinations may struggle to parse these outputs.

Solution:
Use agents supporting structured outputs.

Examples:

- openai-functions
- structured-chat agents

---

# 17. Debugging Agents

Helpful practices:

### Enable verbose mode

```

verbose=True

```

Shows reasoning steps:

- Thought
- Action
- Observation

---

### Use `invoke()`

Better for:

- debugging workflows
- tracking inputs and outputs

---

### Experimentation

LangChain evolves rapidly.

Agent design requires:

- testing different LLMs
- trying different agent types
- adjusting tool formats

---

# 18. Key Takeaways

### Agents
- Combine **LLM + tools + reasoning loop**
- Enable **real-world actions**

### Tools
- Must have clear inputs and outputs
- Prefer structured formats

### Agent Types

| Agent | Use Case |
|---|---|
| Zero-Shot ReAct | Simple tool use |
| Structured Chat ReAct | Typed inputs |
| OpenAI Functions | Complex structured outputs |

---

# 19. Overall Agent Workflow

```

User Question
↓
LLM Reasoning
↓
Select Tool
↓
Execute Tool
↓
Observe Output
↓
Repeat if needed
↓
Final Response

```

---

