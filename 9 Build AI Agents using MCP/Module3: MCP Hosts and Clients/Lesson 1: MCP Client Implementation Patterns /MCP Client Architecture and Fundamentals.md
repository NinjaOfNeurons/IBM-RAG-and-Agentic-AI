# 🏗️ MCP Client Architecture and Fundamentals

## 🎯 Learning Objectives

After watching this video, you’ll be able to:

* Describe MCP’s **three-layer architecture**
* Understand **MCP client communications, lifecycle phases, and capability negotiation**
* Describe MCP server primitives and STDIO transport
* Explain **multi-server client architecture**
* Compare **TypeScript vs Python MCP SDKs**
* Test and validate MCP clients

---

## 🏛️ MCP Three-Layer Architecture

1. **Host Process**

   * Orchestrates client instances
   * Enforces security policies
   * Aggregates context across multiple servers

2. **MCP Clients**

   * One client instance → one server connection
   * Multiple clients needed for multiple servers
   * Independent lifecycle management

3. **MCP Servers**

   * Expose capabilities through **primitives**: Tools, Resources, Prompts

> ✅ Multiple applications can connect to the same server simultaneously, each with its own client-server connection.

---

## 📡 MCP Communications

* Built on **JSON RPC 2.0** with three message types:

  1. **Requests** – have an ID, expect result/error
  2. **Responses** – contain result or error
  3. **Notifications** – fire-and-forget, no response expected
* Transported via **STDIO** or **Streamable HTTP**

### Common Client Methods

* `list_tools`, `call_tool`
* `list_resources`, `read_resource`
* `list_prompts`, `get_prompt`

### Server-initiated Requests

* Servers can request **sampling** or client notifications

---

## ⏱️ MCP Client Lifecycle Phases

1. **Initialization Phase**

   * Client sends `initialized` request with protocol version & capabilities
   * Server responds with its capabilities
   * Client completes handshake with `initialized` notification

2. **Operation Phase**

   * Bidirectional communication begins
   * Clients discover tools, resources, prompts
   * Clients invoke tools, read resources, get prompts
   * Servers may request sampling

3. **Shutdown Phase**

   * Client sends `shutdown` request
   * Server responds
   * Connection closes and cleans up

---

## 🛠️ Capability Negotiation

* Both sides declare **capabilities** during initialization

  * **Client** → experimental features, sampling, file access
  * **Server** → tools, resources, prompts, notifications support
* Determines what operations are possible → **graceful degradation**

---

## 🔧 MCP Server Primitives

1. **Tools** → perform actions

   * Examples: `read_file`, `query_database`, `fetch_api`
   * Defined with JSON schema: name, description, input schema

2. **Resources** → provide data via URIs

   * Static or dynamic
   * Examples: `file://`, `custom://`

3. **Prompts** → reusable templates for LLMs

   * Formatted messages ready for AI consumption

---

## ⚡ STDIO Transport

* Local, process-based transport
* Client **spawns server process** and communicates via stdin/stdout
* Messages are **newline-delimited JSON**
* Server logs errors to stderr
* Secure for local-only use; no authentication required

---

## 🌐 Multi-Server MCP Client

* Host manages multiple client instances, one per server
* Aggregates capabilities: tools, resources, prompts
* Routes requests from LLM to the appropriate client based on tool/resource name

---

## 🟦 TypeScript vs Python SDK

| Feature            | TypeScript SDK             | Python SDK                              |
| ------------------ | -------------------------- | --------------------------------------- |
| Platform           | Node.js, event-driven I/O  | AsyncIO for asynchronous ops            |
| Transport          | HTTP & SSE                 | Excellent STDIO support                 |
| Server Integration | Native with web frameworks | Works with FastMCP for rapid server dev |
| Use Case           | Web apps                   | Data science, notebooks, scripts        |
| Compatibility      | Cross-compatible           | Cross-compatible                        |

> Both SDKs are production-ready and fully interoperable

---

## ✅ MCP Client Testing

* Use **test servers** included in SDKs
* Validate:

  * Initialization handshake and capability negotiation
  * All three lifecycle phases (init, operation, shutdown)
  * JSON RPC formatting
  * Transport layer (STDIO or HTTP)
  * Error handling (malformed responses, timeouts)
  * Capability detection compliance
* **MCP Inspector** → visualize message flow and debug

---

## 📝 Key Takeaways

* MCP defines a **three-layer architecture** with one-to-one client-server connections
* JSON RPC 2.0 underpins bidirectional messaging
* Connections go through **initialization, operation, and shutdown**
* STDIO transport is ideal for local development
* Multi-server clients allow **aggregation and routing across multiple servers**
* Python and TypeScript SDKs are fully compatible, choose based on stack
