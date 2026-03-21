# 🔒 MCP Security: Permissions and Elicitation

## 🎯 Learning Objectives

After this video, you’ll be able to:

* Explain **MCP’s security mechanisms**: permissions and elicitation
* Apply **client-side policies** via `permissions.json`
* Implement **interactive approval workflows**
* Analyze **audit logs** and enforce effective permission management

---

## 🛡️ Why MCP Security Matters

* AI systems now **execute tools and interact with real systems**, not just generate text.
* Unchecked actions can lead to **data leaks, system changes, or unintended outcomes**.
* MCP addresses this with **structured controls** to make actions:

  * **Intentional**
  * **Visible**
  * **Accountable**

---

## ⚙️ Key Security Mechanisms

### 1️⃣ Permissions

* **Client-side policies** controlling tool execution **before server contact**.

* MCP defines **three policies**:

  | Policy    | Behavior                          | Example                                  |
  | --------- | --------------------------------- | ---------------------------------------- |
  | **Allow** | Executes tool immediately, logged | Reading files, listing directories       |
  | **Deny**  | Blocks tool entirely              | Deleting databases, exposing secrets     |
  | **Ask**   | Requires explicit user approval   | Writing files, sending emails, API calls |

* Policies can be **global per tool** or **argument-specific**.

  * Example: allow reading `test.txt`, ask for `production.yaml`.

* **Stored in `permissions.json`** → travels with client, editable for customization.

---

### 2️⃣ Elicitation

* **Server-initiated structured input** via **JSON schemas**.
* Ensures:

  * **Validated inputs**
  * **Informed user consent**
* Workflow:

  1. Server sends **elicitation request** (JSON schema with fields, types, validation rules).
  2. Client presents to user.
  3. User submits validated data, declines, or cancels.
* Use cases:

  * Multi-step workflows
  * Destructive operations
  * Missing parameters
  * Compliance or security acknowledgments

---

## ⚖️ Risk-Based Controls

MCP clients classify operations by **risk tiers** with default permission policies:

| Risk Level | Description                                 | Default Policy                  |
| ---------- | ------------------------------------------- | ------------------------------- |
| Critical   | System-level control (commands, security)   | Deny unless explicitly approved |
| High       | Destructive (delete files, drop databases)  | Ask + elicitation               |
| Medium     | Modifies data (write files, update records) | Ask                             |
| Low        | Read-only actions (list or read files)      | Allow                           |

> Risk assessment ensures **stronger safeguards for higher-risk actions**.

---

## 📝 Permission Enforcement Flow

1. **LLM decides to call a tool** → generates function call with arguments.
2. **Client checks `permissions.json`**:

   * **Allow** → executes immediately, logs action.
   * **Deny** → rejects before contacting server, logs denial.
   * **Ask** → prompts user for approval with full context.

---

## 🛠️ Audit Logging

* **Essential for compliance and security monitoring**.
* Log entries include:

  * Timestamp
  * Tool name & arguments
  * Applied policy
  * Risk level
  * Outcome (allowed, denied, approved)
* Elicitation logs capture: schema + submitted data
* Suggested format: **append-only JSON lines** with metadata (user ID, session ID, server identity)

---

## 🔄 Interactive Workflows

* Ask policy + elicitation may combine:

  1. Client approval prompt
  2. Server-initiated elicitation for confirmation or extra data
* Example: deleting a file → user types file name + reason → client validates → server executes

---

## 🔑 Best Practices for Permissions

* **Least privilege**: start with `deny all`, allow only necessary tools.
* **Environment-specific policies**:

  * Development → more tools allowed
  * Staging → approval required
  * Production → deny by default
* **User-based policies**: different roles get appropriate permissions
* **Temporary permissions**: grant and revoke automatically
* **Role templates**: Reader, Editor, Admin
* **Version control**: track `permissions.json` changes and enable rollback

---

## ✅ Key Takeaways

* MCP security combines **permissions** and **elicitation** to enforce safe, auditable tool execution.
* **Permissions**: client-side, allow/deny/ask, risk-aware, stored in `permissions.json`.
* **Elicitation**: server-initiated, JSON schema-based structured input.
* **Audit logs**: record policy enforcement, user decisions, and input validation.
* Interactive workflows may involve **both client approval and server-elicited input**, ensuring informed, intentional actions.
