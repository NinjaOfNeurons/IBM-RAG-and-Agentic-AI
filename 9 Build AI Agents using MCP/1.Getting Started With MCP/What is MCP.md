# 🧠 Model Context Protocol (MCP) – Notes

## 📌 What is MCP?

* **MCP (Model Context Protocol)** = Open-source standard
* Used to **connect AI agents to external data sources**
* Supports:

  * Databases (Relational / NoSQL)
  * APIs
  * Local files / code

---

## 🏗️ Core Components of MCP

### 1. 🖥️ MCP Host

* Top-level application
* Can include **one or multiple MCP clients**
* Examples:

  * Chat applications
  * IDE code assistants

---

### 2. 🔌 MCP Client

* Lives inside the **MCP host**
* Responsible for:

  * Communicating with MCP servers
  * Requesting tools

---

### 3. 🗄️ MCP Server

* Provides **tools and data access**
* Connects to:

  * Databases (SQL / NoSQL)
  * APIs (any standard)
  * Local files or code

---

### 4. 🌐 MCP Protocol

* Acts as a **transport layer**
* Enables communication between:

  * MCP Host / Client ↔ MCP Server

---

## 🔗 Architecture Overview

```
User → MCP Host (Client) ↔ MCP Protocol ↔ MCP Server → Data Sources
```

* A host can connect to **multiple servers**
* Servers can expose **multiple tools**

---

## ⚙️ How MCP Works (Step-by-Step)

### 🧩 Step 1: User Query

* User asks a question (e.g., weather, customer count)

---

### 🔍 Step 2: Tool Discovery

* MCP Host → requests available tools from MCP Server
* MCP Server → returns available tools

---

### 🤖 Step 3: LLM Interaction

* MCP Host sends:

  * User query
  * Available tools
    → to the **LLM**

---

### 🧠 Step 4: Tool Selection

* LLM decides:

  * Which tool(s) to use

---

### 📡 Step 5: Tool Execution

* MCP Host calls relevant MCP Server(s)
* MCP Server:

  * Executes:

    * Database queries
    * API calls
    * Local code

---

### 🔁 Step 6: Response Flow

* MCP Server → returns result
* MCP Host → sends result back to LLM

---

### ✅ Step 7: Final Answer

* LLM generates final response
* Returned to user via host (e.g., chat app)

---

## 🔄 Key Capabilities

* Supports **multiple MCP servers**
* Enables **tool-based AI workflows**
* Works with:

  * Any database type
  * Any API standard
* Allows **chained / multiple tool calls**

---

## 💡 Why Use MCP?

* Standardized way to connect agents to data
* Makes agents:

  * More modular
  * More scalable
* Useful for:

  * AI agents
  * Developer tools
  * Intelligent assistants

---