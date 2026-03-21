# 🏗️ Build an Enhanced MCP Server with FastMCP

## 🎯 Learning Objectives

After watching this video, you’ll be able to:

* Set up a Python workspace and MCP server
* Use **MCP context** for logging, progress reporting, and user input elicitation
* Create **tools, resources, and prompts** for specific use cases

---

## 🖥️ Workspace Setup

1. **Clone the project template** from GitHub
2. **Create a virtual environment**:

   ```bash
   python -m venv .venv
   ```
3. **Activate the environment**
4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
5. Navigate to the **project root** to start development

---

## 🛠️ MCP Server Setup

* Open `server.py`
* Import required dependencies
* Define:

  * `BASE_DIR` for project root
  * User **elicit schemas** for input requests
  * MCP server object
* Helper function to get **absolute paths** from `BASE_DIR`

---

## ⚡ MCP Server Primitives

MCP servers are built using **three main primitives**: **Tools**, **Resources**, and **Prompts**.
All of these can access **MCP context** for logging, progress reporting, and user input elicitation.

### 1️⃣ Tools

* **Active operations** (e.g., create/delete files, call databases)

* Example: **File tools**

  ```python
  @mcp.tool
  def write_file(path: str, content: str):
      abs_path = get_path(path)
      os.makedirs(os.path.dirname(abs_path), exist_ok=True)
      context.report_progress("Writing file...")
      with open(abs_path, "w") as f:
          f.write(content)
      context.log("File written successfully")
  ```

* **Delete tool** checks file existence and deletes:

  ```python
  @mcp.tool
  def delete_file(path: str):
      abs_path = get_path(path)
      if os.path.isfile(abs_path):
          os.remove(abs_path)
          context.log("File deleted")
      else:
          context.log_error("Path is not a file")
  ```

---

### 2️⃣ Resources

* **Passive access** to data (static or dynamic)

* Defined with **URIs** or **resource templates**

* Example: **Fetch file content**

  ```python
  @mcp.resource("file:///{name}")
  def fetch_file(name: str):
      path = get_path(name)
      if os.path.isfile(path):
          return open(path).read()
      return {"error": "File not found"}
  ```

* **List directory contents**:

  ```python
  @mcp.resource("dir://.")
  def list_dir():
      items = [{"name": f, "type": "file" if os.path.isfile(f) else "dir"} for f in os.listdir(".")]
      return {"items": items}
  ```

> ⚠️ Note: `file:///` uses three forward slashes because `file://` with two slashes expects a host.

---

### 3️⃣ Prompts

* **Reusable templates** for structured tasks

* Example: **Code review**

  ```python
  @mcp.prompt
  def code_review(file_path: str):
      if not os.path.exists(file_path):
          return {"error": "File not found"}
      content = open(file_path).read()
      return f"Please review the following code:\n{content}"
  ```

* **User elicitation** allows dynamic input from users:

  ```python
  file_to_doc = context.elicit("Which file should I document?", type="file")
  doc_name = context.elicit("Name of the new documentation file?")
  ```

---

## 🧩 MCP Context Features

* **Logging** → for debugging and audit trails
* **Progress reporting** → track long-running operations
* **User elicitation** → request structured input from users

Use context inside tools, prompts, and resources for richer interactions.

---

## ▶️ Entry Point

* The MCP server must have a **main entry point** to start and listen for client connections:

  ```python
  if __name__ == "__main__":
      mcp.run()
  ```
* Supports **STDIO transport** for local clients
* Clients can then connect and coordinate **tools, resources, prompts**

---

## ✅ Key Takeaways

* MCP extends LLM capabilities beyond text generation
* **Tools** → perform actions (write, delete, fetch, etc.)
* **Resources** → controlled access to data
* **Prompts** → reusable workflows with parameters and elicitation
* **Context system** → logging, progress updates, and user input handling
* MCP servers expose these capabilities, and MCP clients orchestrate interactions with an LLM
\