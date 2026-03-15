# Building Reflexion Agents – Key Notes

## 1. Overview

* Goal: **Build a Reflexion Agent** using **prompt engineering, schema design, and iterative feedback loops**.
* Roles:

  * **Responder node:** Generates initial answers.
  * **Revisor node:** Refines answers with **evidence, citations, and improved reasoning**.
* Uses external tools (e.g., **Tavily Search**) for real-time information.

---

## 2. Setting Up Environment

* **Imports:** Load necessary Python/LLM libraries.
* **Tavily Search Tool:**

  * Configure API key.
  * Return **up to 5 results per query**.
  * Test with sample queries (e.g., breakfast recipes).
  * Returns structured JSON: `{title, URL, content}`.
* **ChatOpenAI Model:** Instantiate GPT model for initial responses.

---

## 3. Prompt Engineering

* Define **system messages** to assign LLM personas:

  * **Dr. Paul Saladino:** Focus on **controversial health approaches** (carnivore diet, fasting).
  * **Dr. Peter Attia:** Focus on **longevity and evidence-based guidance**.
* Use `first_instruction` variable for **custom prompts** (e.g., 250-word responses).
* Chain system messages to LLM to guide output behavior.

---

## 4. Schema Design

* **AnswerQuestion class:**

  * Fields: `answer`, `reflections`, `search_queries`.
  * Integrates **Reflection class**.
  * Captures:

    * **Missing fields**: Info required.
    * **Superfluous fields**: Unnecessary info.
* **Reviser schema:** Subclass of AnswerQuestion.

  * Adds `citations` field for **evidence-based revisions**.

---

## 5. Iterative Feedback Loop

1. **Responder Node:** Generates initial answer (e.g., carnivore diet).
2. **Extract Search Queries:** Function to parse **tool_call parameters**.
3. **Execute Tools Node:** Call Tavily Search using queries.
4. **Revisor Node:** Refines answer with:

   * Tool outputs.
   * Original critique.
   * Evidence-backed citations.
5. **Response List:** Stores **user messages, AI messages, and tool outputs** for iteration.
6. **Conditional Node (Event Loop):** Controls **number of iterations** (e.g., max 4).

---

## 6. Graph Construction (LangGraph)

* Nodes:

  1. **Respond Node** → initial_chain
  2. **Execute Tools Node** → runs search queries
  3. **Revisor Node** → revisor_chain
* Edges:

  * Respond → Execute Tools → Revisor
  * Revisor → Event Loop → decide iterate/exit
* Entry point: **Draft Responder Node**
* Iterative loop outputs **structured AI + tool messages**.

---

## 7. Example Workflow

* **Query:** "I'm pre-diabetic, need to lower blood sugar, and have heart issues."
* **Responder output:** General animal-based nutrition advice (eggs, fatty meats, organ foods).
* **Revisor output:**

  * Adds **5 scientific citations**.
  * Includes measurable outcomes (e.g., postprandial glucose).
  * Improves precision (processed vs unprocessed foods).
  * Still partially misses heart health considerations.

---

## 8. Key Takeaways

* **Search tools** enhance AI responses with real-time external data.
* **Prompt engineering + schema design** ensures **structured, reflective outputs**.
* **AnswerQuestion + Reflection schemas**:

  * Capture answers, flag missing/superfluous info, generate search queries.
* **LangGraph chaining**: Creates **iterative feedback loops** connecting responder, tools, and revisor.
* **Message graph orchestration**: Controls **node routing, iterations, and flow**, producing improved, citation-backed responses.
