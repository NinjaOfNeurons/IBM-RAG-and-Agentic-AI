# Building AI Agents with Open-Source Frameworks – Revision Notes

## 1. Agentic AI Overview

**Agentic AI** refers to AI systems that can:
- Think
- Plan
- Take actions

Unlike simple chat systems, agentic systems can **perform multi-step reasoning, decision-making, and tool usage**.

Open-source frameworks help developers build these systems by providing:
- Agent infrastructure
- Workflow coordination
- Tool integrations
- Memory handling

---

# 2. Major Agentic Frameworks

The video focuses on four frameworks:

1. CrewAI  
2. LangGraph  
3. AutoGen  
4. BeeAI  

Each framework structures **multi-agent workflows differently**.

---

# 3. CrewAI

## Core Idea
CrewAI simulates **a team of virtual experts** collaborating to complete tasks.

Agents have:
- Roles
- Goals
- Backstories
- Tools

Tasks are assigned to agents and executed through a **crew object**.

---

## CrewAI Workflow

Typical pipeline:

Generator → Evaluator → Final Output

Process:

1. Generator produces output
2. Evaluator checks quality
3. If rejected → feedback to generator
4. If accepted → finalize output

---

## CrewAI Implementation Steps

### 1. Setup Environment
- Import libraries
- Configure LLM
- Add tools (e.g., search tools)

### 2. Define Agents
Example agents:
- Researcher
- Writer

Each agent includes:
- Role
- Goal
- Backstory

---

### 3. Create Tasks

Tasks include:
- Description
- Assigned agent
- Expected output

---

### 4. Create Crew

Combine:
- Agents
- Tasks

Set execution mode:
- Sequential processing

---

## CrewAI Strengths

- Role-based agent collaboration
- Easy team simulation
- Structured task coordination

### Limitations
- Limited flexibility
- Harder debugging

---

# 4. LangGraph

## Core Idea
LangGraph builds **custom workflows using Directed Acyclic Graphs (DAGs).**

In this framework:

- **Nodes = agents or LLM calls**
- **Edges = workflow transitions**

---

## Key Features

- Fine-grained workflow control
- Advanced memory handling
- Error recovery
- Custom routing logic

But:
- Requires **more verbose code**

---

## LangGraph Architecture

### State Object
Stores shared data between agents.

Example:
- conversation history
- evaluation results
- task state

---

### Nodes

Nodes represent steps such as:

- Generator
- Evaluator
- Tool calls

Each node processes inputs and updates state.

---

### Router Nodes

Routers decide **workflow direction**.

Example logic:

```

If evaluation = fail → return to generator
If evaluation = pass → finish

```

Routers enable **dynamic workflow branching**.

---

### Building the Graph

Steps:

1. Define nodes
2. Define edges
3. Add routing logic
4. Execute workflow

---

## LangGraph Strengths

- Highly flexible workflows
- Advanced debugging
- Strong state control

### Best Use Cases

- Complex automation
- Multi-step enterprise workflows
- Finance or healthcare systems

---

# 5. AutoGen

## Core Idea
AutoGen enables **multi-agent conversations**.

Agents collaborate through **dialogue-based workflows**.

Originally developed by **Microsoft**.

---

## Key Features

- Agent-to-agent communication
- Human-in-the-loop systems
- Built-in code execution
- Chat-based workflow

---

## Example System: Study Assistant

Agents:

1. **Student Agent**
   - Provides topic

2. **Concept Analysis Agent**
   - Explains key concepts

3. **Study Tips Agent**
   - Suggests learning strategies

---

## Agent Definition

Each agent includes:

- Name
- System message
- LLM configuration

System messages define **agent responsibilities**.

---

## Group Chat Configuration

Components:

### Group Chat
Defines how agents interact.

### Key Parameters

`max_round`
- Number of conversation turns

`speaker_selection_method = round_robin`
- Agents speak in ordered rotation

---

### Group Chat Manager

Responsible for:
- Coordinating conversations
- Managing turn-taking
- Orchestrating agent interactions

---

## AutoGen Strengths

- Easy prototyping
- Natural conversation workflows
- Good for collaborative systems

### Best Use Cases

- Virtual assistants
- Customer support bots
- Coding assistants
- Research assistants

---

# 6. BeeAI

## Core Idea

BeeAI is a **modular framework** for building intelligent agent workflows with tool integration.

Supports:

- Sequential execution
- Parallel execution
- Multi-agent collaboration

---

## Example Workflow

Agents collaborating to create a **location report**.

### Agents

1. **Researcher**
   - Uses Wikipedia
   - Provides historical context

2. **Weather Forecaster**
   - Uses weather API
   - Provides live weather data

3. **Data Synthesizer**
   - Combines outputs
   - Generates final report

---

## Workflow Execution

Agents can run:

- **Sequentially**
- **In parallel**

Final output = **combined summary**

Example use case:
- Travel assistant
- Educational chatbot

---

# 7. Framework Comparison

| Framework | Workflow Style | Strength |
|---|---|---|
| CrewAI | Role-based teams | Simple collaboration |
| LangGraph | Graph workflows (DAGs) | High flexibility |
| AutoGen | Dialogue-driven | Conversational agents |
| BeeAI | Modular workflows | Tool-integrated automation |

---

# 8. Workflow Patterns

Common patterns supported by these frameworks:

### Reflection Pattern
Generator → Evaluator → Improvement loop

### Delegation Pattern
Tasks assigned to specialized agents.

### Turn-Based Pattern
Agents interact through conversation rounds.

---

# 9. Tradeoffs Between Frameworks

| Factor | CrewAI | LangGraph | AutoGen | BeeAI |
|---|---|---|---|---|
| Ease of Use | High | Medium | High | Medium |
| Flexibility | Low | Very High | Medium | High |
| Code Complexity | Low | High | Medium | Medium |
| Debugging | Hard | Strong | Medium | Medium |

---

# 10. Key Takeaways

- Agentic frameworks simplify building **autonomous AI systems**.
- Multi-agent systems allow **specialized agents to collaborate**.
- Each framework follows a **different workflow architecture**.

### Summary

CrewAI → Role-based teams  
LangGraph → Graph-based workflows  
AutoGen → Conversation-based collaboration  
BeeAI → Modular tool-driven systems

Choosing the right framework depends on:
- workflow complexity
- flexibility requirements
- debugging needs
- deployment scale
