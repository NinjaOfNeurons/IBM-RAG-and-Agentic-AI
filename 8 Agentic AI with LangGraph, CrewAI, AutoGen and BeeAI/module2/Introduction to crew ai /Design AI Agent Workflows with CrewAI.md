# Design AI Agent Workflows with CrewAI – Revision Notes

## 1. Overview

**CrewAI** is a framework for building **multi-agent AI systems** where agents collaborate to complete tasks.

CrewAI is built around **four core components**:

1. **Task**
2. **Agent**
3. **Tool**
4. **Flow**

These components work together to create **structured multi-step workflows**.

---

# 2. Core Components of CrewAI

## 2.1 Task

A **task** defines **what work should be done**.

Key parameters:

| Parameter | Description |
|---|---|
| description | What the agent should do |
| expected_output | Desired result format |
| agent | The agent responsible for the task |

Example:

```

Description: Analyze generative AI breakthroughs
Expected Output: Detailed summary

```id="task_example"

Tasks act like a **director giving instructions to actors**.

---

## 2.2 Agent

An **agent** is an LLM-powered entity that performs tasks.

Agents are defined using **structured prompts**.

Key parameters:

| Parameter | Purpose |
|---|---|
| role | Type of expert the agent represents |
| goal | Objective guiding the agent |
| backstory | Context shaping agent behavior |
| tools | External capabilities |
| verbose | Enables detailed logs |

Example structure:

```

Role: Senior Research Analyst
Goal: Uncover cutting-edge insights
Backstory: Expert analyst specializing in emerging technologies

```id="agent_structure"

Agents simulate **human-like personalities and expertise**.

---

## 2.3 Tools

**Tools** extend the capabilities of agents.

Examples:

- Search APIs
- Databases
- Web scraping
- External APIs

Example tool:

```

SerperDevTool

```id="tool_example"

This allows **real-time web search**.

Tools can be used by:

- Agents
- Tasks

---

## 2.4 Flow

**Flow** defines how tasks execute and how agents interact.

Two common execution patterns:

### Sequential Flow

Tasks run **one after another**.

```

Task 1 → Task 2 → Task 3

```id="seqflow"

Output from one task becomes input to the next.

---

### Hierarchical Flow

A **manager agent** assigns tasks dynamically.

Used when:

- workflow complexity varies
- tasks must be delegated dynamically

---

# 3. Crew Object

The **Crew object** orchestrates the workflow.

It combines:

- agents
- tasks
- tools
- execution flow

Example configuration:

```

Crew(
agents=[agent1, agent2],
tasks=[task1, task2],
process=Process.sequential
)

```id="crew_object"

---

# 4. Example: CrewAI Content Pipeline

This system uses **two agents working sequentially**.

Workflow:

```

Research Analyst → Content Writer

```id="pipeline_flow"

---

## 4.1 Research Analyst Agent

Role:

```

Senior Research Analyst

```id="role_ra"

Goal:

```

Uncover cutting-edge insights

```

Backstory:

Expert in analyzing emerging technology trends.

Tools:

- Web search tool
- Shared LLM

Output:

Detailed research summary.

---

## 4.2 Writer Agent

Role:

```

Tech Content Strategist

```id="role_writer"

Goal:

Create engaging, well-structured content.

Backstory:

Expert at simplifying complex topics for general audiences.

Output:

```

Four-paragraph blog post

```id="writer_output"

---

# 5. Workflow Execution

Steps:

1. Crew object initialized
2. Research agent performs research task
3. Research output generated
4. Writer agent uses research findings
5. Writer generates blog article
6. Crew completes execution

---

# 6. Running the Crew

Execution starts using:

```

crew.kickoff()

```id="kickoff"

Input example:

```

topic = "Generative AI breakthroughs"

```id="topicinput"

The topic variable is passed into the task description.

---

# 7. Crew Output Object

After execution, results are stored in a **Crew Output Object**.

It contains multiple fields.

---

## 7.1 Raw Output

```

result.raw

```id="rawfield"

Contains the **final combined output** from all agents.

---

## 7.2 Task Outputs

```

result.tasks_output

```id="taskoutput"

Contains **individual task results**.

Example:

| Task | Output |
|---|---|
| Research task | Research report |
| Writing task | Blog article |

---

## 7.3 Token Usage

Tracks model usage and cost.

```

result.token_usage

```id="tokenusage"

Metrics include:

| Metric | Meaning |
|---|---|
| prompt_tokens | Input tokens |
| completion_tokens | Generated tokens |
| total_tokens | Total usage |

Useful for:

- cost monitoring
- performance analysis

---

# 8. Execution Variability

Outputs may change across runs because:

- LLM responses are probabilistic
- prompts may vary slightly
- tool results may differ

Thus:

```

same input ≠ identical output

```id="variation"

---

# 9. Key Advantages of CrewAI

- Clear role-based agent design
- Human-like collaboration
- Easy multi-agent orchestration
- Tool integration
- Built-in workflow management

---

# 10. Key Takeaways

- **CrewAI** enables multi-agent collaboration with defined roles and tasks.
- **Tasks** specify what work should be done.
- **Agents** are LLM-powered workers guided by role, goal, and backstory.
- **Tools** extend agent capabilities through APIs or search engines.
- **Flows** control how tasks execute (sequential or hierarchical).
- The **Crew object** orchestrates agents, tasks, and workflows.
- The **Crew Output Object** stores final results, task outputs, and token usage metrics.
```
