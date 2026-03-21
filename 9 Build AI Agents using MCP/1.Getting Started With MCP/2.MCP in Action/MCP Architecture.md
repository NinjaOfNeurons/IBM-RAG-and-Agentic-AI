# 📘 MCP Architecture Notes

## 🧩 MCP Client-Server Model Overview

The MCP architecture consists of **three core components**:

### 1. MCP Host

* The **AI application interface** where users interact
* Examples:

  * Chatbots
  * IDE assistants
  * Desktop apps
* Responsibilities:

  * Capture user input (text, commands, files)
  * Send input to AI model
  * Display responses clearly
  * Manage:

    * Conversations
    * Context
    * UI controls
    * Tool integrations

---

### 2. MCP Client

* Acts as a **communication bridge** between host and server

* Runs **inside the host application**

* Key functions:

  * Converts user requests → **JSON-RPC format**
  * Sends structured requests to MCP server
  * Handles:

    * Responses
    * Errors
    * Context relevance
  * Manages session lifecycle:

    * Timeouts
    * Reconnection
    * Interruptions
    * Session termination

* Architecture notes:

  * One host → multiple clients
  * Each client → **one server (1:1 relationship)**

---

### 3. MCP Server

* External service providing **tools and context**
* Responsibilities:

  * Convert requests → real actions
  * Connect to:

    * Databases
    * APIs
    * Local files
* Provides reusable services via MCP protocol
* Common integrations:

  * Slack
  * GitHub
  * Docker
  * Web search

---

## ⚙️ MCP Core Primitives

MCP defines **3 types of primitives**:

### 1. Tools

* Perform **actions**
* Examples:

  * Calculations
  * Sending messages
  * File operations
  * API calls

---

### 2. Resources

* Provide **data only (no actions)**
* Examples:

  * File contents
  * Database records
  * API responses

---

### 3. Prompts

* **Reusable templates/workflows**
* Help structure communication between:

  * LLM
  * MCP server

---

## 🏗️ MCP Architecture Layers

### 1. Data Layer

* Based on **JSON-RPC protocol**
* Defines:

  * Client-server communication
  * Lifecycle management
  * Core primitives

#### JSON-RPC Message Types:

* **Request** → requires response
* **Response** → reply to request
* **Notification** → no response needed

---

### 2. Transport Layer

* Handles **communication + security**
* Responsibilities:

  * Connection setup
  * Message transmission
  * Authentication

---

## 🔌 MCP Transport Mechanisms

### 1. STDIO (Standard Input/Output)

* Best for **local integrations**
* Features:

  * Lightweight
  * Synchronous communication
* Use cases:

  * Local files
  * Local databases
  * Local APIs

---

### 2. Streamable HTTP

* Designed for **remote server communication**

* Features:

  * HTTP POST communication
  * Optional streaming (Server-Sent Events)
  * Supports authentication:

    * Bearer tokens
    * API keys
    * Custom headers

* Recommended auth:

  * **OAuth framework**

---

## 🔁 Communication Flow

### Client → Server

* MCP messages → converted to **JSON-RPC**

### Server → Client

* JSON-RPC responses → converted back to **MCP messages**

---

## 🧠 Key Takeaways

* MCP has **3 core components**:

  * Host
  * Client
  * Server
* MCP primitives:

  * Tools (actions)
  * Resources (data)
  * Prompts (templates)
* Two architecture layers:

  * Data layer (protocol)
  * Transport layer (communication)
* Two transport mechanisms:

  * STDIO → local
  * Streamable HTTP → remote
