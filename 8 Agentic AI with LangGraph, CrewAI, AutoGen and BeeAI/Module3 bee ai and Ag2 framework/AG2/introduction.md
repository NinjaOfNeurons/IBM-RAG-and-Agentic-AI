# 📘 Introduction to AG2 (AutoGen) — Notes

## 🔹 What is AG2?

* **AG2 (formerly AutoGen)** is an open-source framework for building **collaborative AI agents**.
* Enables multiple agents (e.g., researcher, analyst, writer) to work together.
* Supports **role-based interactions** instead of relying on a single LLM.
* **Provider-agnostic**: works with models like OpenAI, Anthropic, etc.
* Designed for **real-world deployment** with scalability and error handling.

---

## ⚙️ Setup & Configuration

* Install AG2 with optional dependencies for different LLM providers.
* Import core components:

  * Agent types
  * Orchestration tools
* Configure models using `LLMConfig`:

  * Shared config across agents OR
  * Custom config per agent
* Example choice: GPT models for balance of **cost + capability**.

---

## 🧠 Core Concepts

### 1. Conversable Agent

* Fundamental building block.
* Agents can:

  * Send messages
  * Receive messages
  * Respond autonomously
* Example:

  * **Student agent** → asks questions
  * **Tutor agent** → provides explanations

---

### 2. Human-in-the-Loop (HITL)

* Integrates human oversight into workflows.
* Modes:

  * `always` → requires input at every step
  * `never` → fully autonomous
  * `terminate` → input only at the end
* Useful for:

  * Finance
  * Healthcare
  * Legal systems

---

### 3. Multi-Agent Orchestration

* Multiple agents collaborate on tasks.
* Enables **distributed reasoning** and specialization.

---

### 4. Tools Integration

* Agents can use:

  * APIs
  * External systems
  * Code execution
* Extends agent capabilities beyond text generation.

---

### 5. Structured Outputs

* Ensures:

  * Consistency
  * Reusability
  * Easier downstream processing

---

## 🤖 Agent Design

### Key Idea: Specialized Roles

* Define agents using **system messages**.
* Examples:

  * Technical expert → precise, code-focused
  * Creative writer → storytelling & analogies

---

## 💬 Conversational Workflow

### Example: Student–Tutor Chat

* Use `initiate_chat()` to start interaction.
* Parameters:

  * `max_turns` → prevents infinite loops
  * `summary_method="reflection_with_llm"` → generates final summary

### Benefits:

* Better than single prompts
* Produces:

  * Context-rich responses
  * Structured reasoning
  * Higher-quality outputs

---

## 🧪 Assistant + UserProxy Pattern

### Roles:

* **Assistant Agent**

  * Writes code
  * Solves problems
* **UserProxy Agent**

  * Executes code
  * Provides feedback

### Features:

* Safe execution via sandbox (local environment)
* `max_consecutive_auto_reply` prevents loops

### Example Task:

* Generate Python code to plot a sine wave
* Execute and save as `sine_wave.png`

⚠️ **Caution**: Always use discretion with code execution.

---

## 👤 Human-in-the-Loop Example

### Bug Triage System

* AI:

  * Classifies bugs
  * Suggests actions
* Human:

  * Approves/rejects decisions

✔ Ensures:

* Safety
* Accountability
* Oversight

---

## 🔄 Orchestration Patterns

### 1. Two-Agent Chat

* Simple back-and-forth communication

### 2. Group Chat

* Multiple agents collaborate
* Managed by **GroupChatManager**

#### Speaker Selection Modes:

* `auto` → LLM chooses next speaker
* `round_robin` → fixed order
* `manual` → human selects
* `random` → random selection

---

### 3. Sequential Chat

* Step-by-step refinement

### 4. Nested Chat

* Sub-conversations reused as workflows

---

## 🧩 Group Chat Example

### Roles:

* **Lesson Planner** → creates content
* **Reviewer** → gives feedback
* **Teacher** → supervises & terminates

### Flow:

1. Teacher starts: *"Make a lesson about the moon"*
2. Agents collaborate dynamically
3. Ends when teacher says `"done"`

---

## 🚀 Key Advantages of AG2

* ✅ Role-based collaboration
* ✅ Scalable architecture
* ✅ Flexible LLM integration
* ✅ Human oversight support
* ✅ Better outputs than single-agent systems

---

## 🧾 Summary

* AG2 enables **collaborative AI systems** using multiple specialized agents.
* Core strengths:

  * Conversable agents
  * Human-in-the-loop control
  * Multi-agent orchestration
  * Tool integration
* Ideal for solving **complex, real-world problems** through structured interaction.
