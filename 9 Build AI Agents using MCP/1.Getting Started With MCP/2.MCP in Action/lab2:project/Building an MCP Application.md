# 📘 Building an MCP Application with Python

## 🎯 Learning Objectives

After this video, you will be able to:

* Create MCP servers using **STDIO** and **HTTP** transports
* Use **Multi-Server MCP Client** to connect and manage multiple MCP servers
* Build a **Langraph React agent** powered by GPT-5
* Create a looping CLI tool for interacting with a session-persistent agent

---

## 🏗️ MCP Application Overview

* **Transport Layer**: How MCP clients and servers communicate

  * **STDIO** → Local servers
  * **HTTP** → Remote servers
* **Workflow**:

  1. User sends a prompt to the agent
  2. Agent selects an MCP tool based on the prompt
  3. Agent extracts parameters (`X`) and sends to MCP server via client & transport
  4. MCP server processes input → returns result (`Y`)
  5. Agent receives output → produces final response to user

---

## 🔗 Multi-Server MCP Client

* Can connect to **multiple MCP servers simultaneously**
* Supports **both STDIO and HTTP servers**
* No strict limit on the number of servers; practical limit depends on resources and design
* Handles **transport details automatically** → lets you focus on application logic

### Example Servers

1. **MetMuseum-MCP Server (STDIO)**

   * Access to **Metropolitan Museum of Art** database
   * Over 400,000 artworks, images, metadata, historical info

2. **Context 7 Server (HTTP)**

   * LLM-optimized **library and framework documentation**
   * Returns AI-readable docs

---

## 🧩 Agent Integration

### Steps:

1. **Import Multi-Server MCP Client** and create client object
2. Provide server connection info:

   * Context 7 → HTTP server credentials
   * MetMuseum → STDIO server via NPX
3. **Import agent components**:

   * `create_react_agent` → builds React agent + language model
   * `InMemorySaver` → stores conversation memory across multiple exchanges
   * `asyncio` → enables non-blocking async communication
4. **Fetch tools** from all connected servers:

   * `get_tool()` returns a list of available tool objects
   * Example tools:

     * `resolve-library-id` (Context 7)
     * MetMuseum search tools

---

## 🤖 Agent Setup

1. **Initialize LLM object**
2. **Initialize Memory Saver** → keeps context for multi-turn conversations
3. **Set thread/session ID** → supports multiple conversations simultaneously
4. **Create agent** with available tools
5. **Prompt agent** with role and capabilities

   * Introduces tools for software documentation and museum exploration

---

## 💬 Building a CLI Chatbot

* Loop for user interaction:

  1. Menu: `1` to ask a question, `2` to exit
  2. If `1`: Prompt user → feed input to agent → print response
  3. If `2`: Exit loop

* Example Interaction:

  * User: *“What is the Met?”*
  * Agent: Returns detailed info about **Metropolitan Museum of Art** (location, founding date, collections, images)

* Async execution:

  ```python
  async def main():
      # async MCP application code
  asyncio.run(main())
  ```

---

## 🧠 Key Takeaways

* Two MCP server subtypes: **STDIO** (local), **HTTP** (remote)
* **Multi-Server MCP Client** allows simultaneous connections
* **InMemorySaver** stores conversation history across exchanges
* Async-await pattern ensures **non-blocking communication**
* MCP servers + agent = **persistent, interactive LLM-powered application**
* Wrapping in `async main()` is required when running Python scripts (.py)

