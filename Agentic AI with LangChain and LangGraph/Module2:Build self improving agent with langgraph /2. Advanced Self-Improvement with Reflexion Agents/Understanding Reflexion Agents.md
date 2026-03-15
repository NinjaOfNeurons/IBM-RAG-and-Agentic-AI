# Reflexion Agents – Key Notes

## 1. Overview

* Reflexion agents **enhance AI responses** via iterative self-critique and tool use.
* They **extend reflection agents** by adding **citations, real-time data, and verifiable claims**, not just improved opinions.

---

## 2. Reflection vs. Reflexion

* **Reflection agents:** Iteratively review & refine outputs.
* **Reflexion agents:** Go further:

  * Incorporate **new information post-training**.
  * Provide **structured, citation-backed outputs**.
  * Support **self-improving cycles**.

---

## 3. Core Workflow

1. **User query:** e.g., "I need more minerals in my diet."
2. **Generator/Responder LLM:** Creates initial response.

   * Example: "Try spinach (iron/magnesium), almonds (magnesium), dairy (calcium)."
3. **Self-Critique:** LLM evaluates its own output.
4. **Tool Integration:** Calls APIs or web search for real-time info.
5. **Structured Output:** LLM outputs **schemas** instead of plain text:

   * Fields: `query`, `response`, `self_critique`, `references`, `tool_query`

---

## 4. Revisor Role

* Takes **responder output + tool results**.
* Revises response, adds **citations & references**.
* Outputs follow **same schema**.
* Iterative updates continue until a **predetermined stopping point**.

---

## 5. Iterative Cycle

* **Responder → Tool → Revisor → Response List**
* Each iteration:

  * Improves clarity, accuracy, and usefulness.
  * Incorporates **feedback from prior runs**.
  * Updates references based on latest external information.

---

## 6. Strengths of Reflexion Agents

* **Self-improvement:** Identify & fix weaknesses.
* **Real-time data integration:** Web search, APIs, etc.
* **Structured, transparent outputs:** Easy to trace reasoning & references.
* **Citation-backed responses:** Enhances reliability & verifiability.

---

## 7. Example Schema Fields

| Field           | Description          |
| --------------- | -------------------- |
| `query`         | User input           |
| `response`      | LLM’s answer         |
| `self_critique` | LLM evaluates output |
| `references`    | Sources/citations    |
| `tool_query`    | Search/API queries   |

---

### Summary

* Reflexion agents = **iterative generator + self-critique + tool integration + citations**.
* **Improves over reflection agents** by enabling structured, verifiable, and up-to-date responses.
* Workflow is **loop-based**, leveraging **response lists** to refine knowledge over multiple iterations.

