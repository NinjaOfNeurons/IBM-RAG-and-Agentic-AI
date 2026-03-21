# 🧠 MCP vs API – Notes

---

## 📌 Why LLMs Need External Integration

* LLMs need to interact with:

  * External data sources
  * Services
  * Tools

* Traditionally done using:

  * **APIs (Application Programming Interfaces)**

* New approach (late 2024):

  * **MCP (Model Context Protocol)** introduced by Anthropic

---

## 🔌 What is MCP?

* **MCP = Open standard protocol**
* Standardizes how:

  * AI applications
  * LLMs
  * External data/services
    → connect and interact

---

## 🔋 MCP Analogy (USB-C)

* MCP is like a **USB-C port for AI apps**

| Real World               | MCP Equivalent |
| ------------------------ | -------------- |
| Laptop                   | MCP Host       |
| USB-C Port               | MCP Protocol   |
| Devices (monitor, drive) | MCP Servers    |

✅ Key Idea:

> Different systems work together using one common standard

---

## 🏗️ MCP Architecture

* **MCP Host**

  * Runs multiple MCP clients

* **MCP Client**

  * Opens **JSON-RPC 2.0 session**
  * Connects to MCP servers

* **MCP Server**

  * Exposes capabilities:

    * Database access
    * Code repositories
    * Email services

---

## 🎯 Core Capabilities of MCP

### 1. 📊 Context Retrieval

* Documents
* Knowledge bases
* Database records

---

### 2. 🛠️ Tool Execution

* Web search
* API calls
* Calculations

---

## 🧩 MCP Primitives

### 1. 🔧 Tools

* Actions/functions AI can execute

* Examples:

  * `get_weather`
  * `create_event`

* Include:

  * Name
  * Description
  * Input/output schema

---

### 2. 📂 Resources

* Read-only data
* Examples:

  * Files
  * Database schema
  * Documents

---

### 3. 📝 Prompt Templates

* Predefined prompts
* Help guide LLM behavior

---

## 🔍 Dynamic Discovery (Key Feature)

* MCP servers expose:

  * `tools/list`
  * `resources/list`
  * `prompts/list`

✅ Result:

* AI agents can:

  * Discover capabilities at runtime
  * Use new tools **without redeployment**

---

## 🌐 What is an API?

* **API = set of rules** for communication between systems
* Allows:

  * Data exchange
  * Service access

---

## ⚙️ API Key Concepts

* Acts as **abstraction layer**
* Client does NOT need:

  * Internal implementation details

---

## 🌍 REST API (Most Common Type)

* Uses **HTTP protocol**

### Common Methods:

| Method | Purpose       |
| ------ | ------------- |
| GET    | Retrieve data |
| POST   | Create data   |
| PUT    | Update data   |
| DELETE | Remove data   |

---

### 📌 Example

```http
GET /books/123   → Fetch book details
POST /loans      → Borrow a book
```

* Responses typically in:

  * JSON format

---

## 🤖 APIs in AI Systems

* LLMs often exposed via REST APIs
* AI agents use APIs to:

  * Perform searches
  * Access services

---

## ⚖️ MCP vs API – Key Differences

| Feature         | MCP               | API                |
| --------------- | ----------------- | ------------------ |
| Standardization | Unified protocol  | Varies by API      |
| Discovery       | Dynamic (runtime) | Manual (docs)      |
| Integration     | Plug-and-play     | Custom integration |
| Focus           | AI agents & tools | General software   |
| Flexibility     | High              | Moderate           |
| Reusability     | High              | Limited            |

---

## 🔑 Key Insight

> MCP builds on APIs but adds a **standardized, AI-native layer** for tool usage and context retrieval.

---

## ⚡ Final Takeaway

* APIs = **Basic connectivity**
* MCP = **Smart, standardized connectivity for AI agents**

> MCP is like upgrading from “custom cables everywhere” → to a universal plug system for AI.
