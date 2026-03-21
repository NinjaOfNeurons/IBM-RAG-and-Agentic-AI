# 🖥️ Hello World of MCP Servers with FastMCP

## 🎯 Learning Objectives

After watching this video, you’ll be able to:

* Create MCP servers using **STDIO** and **HTTP** transports
* Register **custom tools, resources, and prompts** to an MCP server
* Test MCP servers with client connections and manual tool calls
* Create **multi-server clients** and **ReAct agents**

---

## 🏗️ MCP Server Basics

* **FastMCP servers** = remote function libraries
* **Transport** = how clients and servers communicate

  * **STDIO** → local server, communicates via stdin/stdout pipes
  * **HTTP** → remote server, communicates via standard HTTP requests

---

## 🛠️ Tools, Resources, and Prompts

### 1️⃣ Tools

* Defined using the `@mcp.tool` decorator (similar to LangChain’s `@tool`)
* Example: Calculator tools

  ```python
  @mcp.tool
  def add(a: int, b: int) -> int:
      "Add two integers"
      return a + b
  ```
* **Function docstrings** guide the LLM on tool purpose
* Supports **type hints** for inputs and outputs

### 2️⃣ Resources

* Like **filing cabinets** for AI access
* Defined using `@mcp.resource` with a **URI template**

  ```python
  @mcp.resource("file:///endpoint/{name}")
  def get_file(name: str):
      return open(f"path/{name}").read()
  ```
* Client requests → URI → extracted parameters → function execution → result returned

### 3️⃣ Prompts

* **Reusable templates** for common tasks
* Example: `code=n=1` → returns “please review this code, n=1”

---

## 🧪 Testing MCP Servers

### In-Memory Transport

* Fast, local testing when **client and server are in the same process**
* Async example:

  ```python
  async with client as c:
      result = await c.call_tool("add", {"a": 5, "b": 4})
      print(result)  # 9
  ```

### HTTP Transport

* Server can run **anywhere**, accessed via URL
* Setup:

  ```python
  transport_http = StreamableHTTPTransport(url="http://localhost:8000/mcp")
  client = Client(transport_http)
  ```
* Async call pattern same as in-memory

### STDIO Transport

* Server runs locally as a **child process**
* Client spawns server and communicates via stdin/stdout
* Python file example: `stdio_server.py` → run server and connect client:

  ```python
  stdio_client = StdioClient(python_file="stdio_server.py")
  ```
* Async call pattern identical to HTTP transport

---

## 🤖 Building an MCP Agent

* Import tools and LLM:

  ```python
  from fastmcp.agent import create_react_agent, chat_openAI
  ```
* Load MCP tools into **LangChain format**:

  ```python
  tools = load_mcp_tools(session)
  agent = create_react_agent(llm, tools)
  ```
* **Agent workflow**:

  1. Parse prompt
  2. Extract parameters (e.g., 1 and 2 for addition)
  3. Call MCP server tool
  4. Receive result → LLM converts to natural language

---

## 🌐 Multi-Server MCP Client

* Connect multiple MCP servers using a dictionary:

  ```python
  servers = {
      "MetMuseum": {"type": "stdio", "file": "metmuseum_server.py"},
      "Context7": {"type": "http", "url": "http://context7.com/mcp"}
  }
  client = MultiServerMCPClient(servers)
  ```
* Access **all tools** across servers, build **ReAct agent**, and invoke tools using `await`

---

## ✅ Key Takeaways

* MCP **transport** determines server-client communication: STDIO (local) vs HTTP (remote)
* Tools, resources, and prompts provide **reusable functionality** for agents
* **In-memory transport** = fast local testing
* **Multi-server MCP client** allows multiple servers to be used simultaneously
* Agent integration is identical across transports; only configuration differs
