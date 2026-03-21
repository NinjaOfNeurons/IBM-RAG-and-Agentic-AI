# 🌐 Streamable HTTP, Roots, and Sampling in MCP

## 🎯 Learning Objectives

After this video, you’ll be able to:

* Compare **STDIO vs Streamable HTTP transports**
* Describe **Streamable HTTP message flow**
* Implement Streamable HTTP in an MCP client
* Explain **sampling** in MCP
* Implement **roots** for file system security
* Manage **multi-transport sessions** in MCP

---

## ⚡ MCP Transport Mechanisms

### 1️⃣ STDIO Transport

* Spawns a **local server subprocess**
* Communicates via **STDIN/STDOUT** with newline-delimited JSON RPC
* Server lifecycle tied to client process
* Ideal for **local development**: file system servers, Git tools

### 2️⃣ Streamable HTTP Transport

* Connects to a **remote server over a network**
* Uses **bidirectional streaming** via a single `/MCP` endpoint
* Server runs independently and can handle **multiple clients simultaneously**
* Replaced **SSE transport** in March 2025
* Benefits:

  * Single endpoint
  * Fully bidirectional communication
  * Reliable and resumable
  * Production-ready

> **Key difference:** STDIO = local, low-latency; Streamable HTTP = remote, scalable

---

## 🔁 Streamable HTTP Message Flow

* Client → sends JSON RPC requests
* Server → responds with results
* Server can initiate requests (e.g., **sampling**)
* Notifications flow **both ways**, no response expected
* Mirrors STDIO’s STDIN/STDOUT pattern but over HTTP with modern reliability

### Implementation Pattern

1. Import `StreamableHTTPClient` from SDK
2. Connect to server URL with `/MCP` endpoint
3. Client returns:

   * `readStream`
   * `writeStream`
   * `sessionInfo`
4. Use **session manager** for:

   * Initialization handshake
   * Capability negotiation
   * Message routing
   * Bidirectional communication
5. Call `initialize()` to start operation

---

## 🧠 Sampling

* **Purpose:** Let servers request LLM reasoning during tool execution
* Example uses: code analysis, summarization, content generation
* Workflow:

  1. Server executes a tool → determines LLM reasoning needed
  2. Server sends `sampling/create` request with:

     * Messages array
     * System prompt
     * Model preferences & generation parameters
  3. Client prompts user for **approval** (security & cost control)
  4. LLM invoked → response flows back to server
* **Optional feature:** Clients declare support; servers handle absence gracefully

---

## 🗂️ Roots

* **Purpose:** Define file system security boundaries for servers
* Clients declare **allowed URLs/directories** during capability negotiation
* Server respects roots → accesses only declared paths
* **Canonical path resolution** prevents traversal attacks (e.g., `/etc/password`)
* Benefits:

  * Sandbox for development
  * Multi-tenant isolation
  * Security audit trails

### Dynamic Roots

* `roots/list_changed` notification informs servers when roots are updated
* Server calls `roots/list` to refresh allowed paths
* Use cases:

  * Temporary access for a task
  * Revoking access after task completion
  * Admin security policy updates

---

## 🔧 Multi-Transport Session Management

* **Session manager** maintains registry mapping server IDs → transport-specific connections
* Each entry tracks:

  * Transport type: STDIO / Streamable HTTP
  * Connection state: connecting, ready, error, closed
  * Active session instance
* Request routing:

  * Lookup which server provides a tool
  * Dispatch request via correct transport
* Enables **hybrid deployments**:

  * STDIO for local servers
  * HTTP for cloud/remote servers
  * Transparent abstraction layer for application logic

---

## ✅ Key Takeaways

* **Streamable HTTP:** modern, bidirectional, reliable, single endpoint transport
* **STDIO:** local, process-bound, low-latency
* **Sampling:** servers request LLM capabilities with user approval
* **Roots:** enforce secure file system boundaries
* **Dynamic roots:** enable real-time security adjustments
* **Multi-transport session management:** flexible, scalable, hybrid deployments
