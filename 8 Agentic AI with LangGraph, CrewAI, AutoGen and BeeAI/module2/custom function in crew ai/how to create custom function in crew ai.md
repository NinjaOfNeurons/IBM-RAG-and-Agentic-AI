# Extending CrewAI with Custom Functions — Notes

## Overview

This video explains how to extend **CrewAI agents** using **custom functions (tools)**.

Key topics:

* Creating **custom tools**
* Assigning tools to **agents**
* Assigning tools to **tasks**
* **Agent-centric vs Task-centric workflows**
* Using tools like **PDF Search** and **Serper Web Search**

Custom tools allow agents to perform **specific domain actions**, improving flexibility and control.

---

# Initial Setup

First initialize the **LLM** used by agents.

Example:

```python
from crewai import LLM

llm = LLM(
    model="ibm/granite-13b-chat-v2",
    temperature=0
)
```

The LLM:

* powers all agents
* handles reasoning
* processes tool outputs

---

# Custom Tools in CrewAI

Tools are **functions that agents can call** to perform tasks.

CrewAI provides:

* built-in tools
* support for **custom tools**

Custom tools are created using the **@tool decorator**.

```python
from crewai.tools import tool
```

---

# Creating Custom Tools

## Addition Tool

```python
from crewai.tools import tool

@tool
def add_numbers(numbers: list) -> int:
    """Adds a list of numbers and returns the sum."""
    return sum(numbers)
```

Purpose:

* performs addition
* returns the total

---

## Multiplication Tool

```python
@tool
def multiply_numbers(numbers: list) -> int:
    """Multiplies numbers and returns the product."""
    result = 1
    for n in numbers:
        result *= n
    return result
```

Purpose:

* multiplies numbers
* returns product

---

# Calculator Agent

The **Calculator Agent** uses these tools.

### Agent Configuration

```python
calculator_agent = Agent(
    role="Calculator",
    goal="Extract numbers and perform arithmetic operations",
    backstory="Expert at interpreting numeric instructions",
    tools=[add_numbers, multiply_numbers],
    llm=llm
)
```

Agent abilities:

* parse natural language
* extract numbers
* choose the correct tool
* perform calculations

---

# Running a Task

Example task:

```python
task = Task(
    description="Add all numbers from the input text",
    agent=calculator_agent
)
```

Crew execution:

```python
crew = Crew(
    agents=[calculator_agent],
    tasks=[task]
)

crew.kickoff(inputs={"text": "Add 7 and 8 also 9 and 10"})
```

---

# Example Execution Flow

Input:

```
Add 7 and 8 also 9 and 10
```

Process:

```
Text Input
   ↓
Agent extracts numbers
   ↓
Agent selects tool
   ↓
Tool performs calculation
   ↓
Result returned
```

Output:

```
34
```

---

# Assigning Tools to Agents and Tasks

There are **two main approaches**:

| Approach      | Description            |
| ------------- | ---------------------- |
| Agent-Centric | Tools belong to agents |
| Task-Centric  | Tools belong to tasks  |

---

# Example System: Daily Dish Q&A Assistant

A chatbot that answers questions about a restaurant.

Data source:

```
DailyDishFAQ.pdf
```

Contains:

* phone number
* hours
* location
* parking info

---

# Tools Used

## PDF Search Tool

```python
PDFSearchTool
```

Purpose:

* search inside PDF documents
* retrieve relevant information

Features:

* Retrieval-Augmented Generation (RAG)
* uses **HuggingFace sentence transformers**

---

## SerperDevTool

```python
SerperDevTool
```

Purpose:

* perform **real-time web searches**

Requirements:

* API key

Used when the answer **is not found in the PDF**.

---

# Agent-Centric Tool Assignment

In this approach:

* **tools are attached to the agent**
* agent decides which tool to use

---

## Inquiry Specialist Agent

```python
agent = Agent(
    role="Inquiry Specialist",
    goal="Answer Daily Dish questions",
    backstory="Expert assistant with access to FAQ and web search",
    tools=[pdf_search_tool, serper_tool],
    llm=llm
)
```

Agent abilities:

* analyze user query
* decide which tool to use
* generate response

---

# Agent-Centric Task

```python
task = Task(
    description="Answer customer questions using FAQ or web search",
    expected_output="Clear, helpful response",
    agent=agent
)
```

---

# Agent-Centric Flow

Example question:

```
What are your phone number, hours, and parking?
```

Process:

```
User Question
      ↓
Agent analyzes query
      ↓
Agent selects PDF tool
      ↓
PDF search retrieves info
      ↓
Agent generates final answer
```

---

# Task-Centric Workflow

In this approach:

* **tools are attached to tasks**
* agent follows instructions step-by-step

Agent **does not choose tools**.

---

# Task-Centric Agent

```python
agent = Agent(
    role="Customer Service Specialist",
    goal="Provide support using guided tasks",
    backstory="Follows structured processes to answer questions",
    llm=llm
)
```

No tools assigned here.

---

# Task 1 — FAQ Search

```python
task1 = Task(
    description="Search the FAQ document",
    tools=[pdf_search_tool],
    agent=agent
)
```

Purpose:

* retrieve relevant information

---

# Task 2 — Response Drafting

```python
task2 = Task(
    description="Draft a friendly response using FAQ results",
    agent=agent
)
```

Purpose:

* format answer
* generate user-friendly response

---

# Task-Centric Crew

```python
crew = Crew(
    agents=[agent],
    tasks=[task1, task2],
    process=Process.sequential
)
```

Tasks execute in **sequence**.

---

# Task-Centric Flow

Example query:

```
What is your phone number, hours, and parking?
```

Process:

```
User Question
      ↓
Task 1: Search FAQ
      ↓
PDF Tool retrieves info
      ↓
Task 2: Draft Response
      ↓
Agent formats reply
      ↓
Final Answer
```

---

# Chatbot Loop

Example interactive chatbot:

```python
while True:
    question = input("Ask a question: ")
    result = crew.kickoff(inputs={"question": question})
    print(result)
```

---

# Agent-Centric vs Task-Centric

| Feature         | Agent-Centric | Task-Centric  |
| --------------- | ------------- | ------------- |
| Tool Assignment | Agent         | Task          |
| Tool Selection  | Agent decides | Fixed by task |
| Flexibility     | High          | Lower         |
| Control         | Lower         | Higher        |
| Workflow        | Dynamic       | Structured    |

---

# When to Use Each

### Agent-Centric

Use when:

* agent must **choose between tools**
* tasks are **dynamic**
* decision making is required

Example:

* research assistants
* search agents

---

### Task-Centric

Use when:

* workflow must be **strict**
* steps must run **in order**
* tools should be **isolated**

Example:

* data pipelines
* document processing
* customer service workflows

---

# Key Takeaways

### Custom Tools

* extend CrewAI capabilities
* implement domain-specific logic

### @tool Decorator

Used to register Python functions as tools.

---

### Agent-Centric Workflow

* tools attached to agent
* agent decides which tool to use

---

### Task-Centric Workflow

* tools attached to tasks
* agent follows fixed process

---

### Benefits of Custom Tools

* domain-specific capabilities
* greater flexibility
* better automation
* improved control over agent behavior
