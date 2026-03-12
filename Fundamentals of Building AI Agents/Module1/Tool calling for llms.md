
# Tool Calling in LLMs – Notes

## 1. What is Tool Calling?

**Tool Calling** is a technique that allows a **Large Language Model (LLM)** to access **external tools such as APIs, databases, or code** to retrieve real-time or dynamic information.

It makes the LLM **context-aware of external data sources** instead of relying only on its training data.

Examples of tools:
- APIs (Weather API, Payment API)
- Databases
- Search engines
- Code interpreters
- Calculators
- External AI models

---

# 2. Basic Tool Calling Architecture

Tool calling typically happens through a **chat-based interaction between a client application and an LLM**.

### Components

1. **Client Application**
2. **Large Language Model (LLM)**
3. **Tool Definitions**
4. **External Tools (APIs, databases, code)**

### Basic Flow

1. Client sends **messages + tool definitions** to the LLM
2. LLM analyzes the user query
3. LLM decides **which tool should be called**
4. Client executes the tool
5. Tool response is sent back to the LLM
6. LLM generates the **final answer**

---

# 3. Tool Definition

Developers must define tools for the LLM to use.

A **tool definition** typically includes:

### 1. Tool Name
Unique identifier for the tool.

Example:
```

get_weather

```

### 2. Tool Description
Explains **what the tool does and when to use it**.

Example:
```

Returns current weather information for a given city.

```

### 3. Input Parameters
Defines **required inputs for the tool call**.

Example:
```

city: string
unit: celsius | fahrenheit

```

---

# 4. Example: Weather Query

### User Question
```

What is the temperature in Miami?

```

### Available Tool
```

Weather API

```

### Step-by-Step Process

#### Step 1 – Send Message + Tools
Client sends:

- User query
- Tool definitions

to the LLM.

---

#### Step 2 – LLM Chooses Tool

The LLM analyzes:

- The user question
- The available tools

LLM response might be:

```

Call tool: get_weather
city: Miami

```

---

#### Step 3 – Tool Execution

The **client application calls the API**:

```

GET /weather?city=Miami

```

Example response:

```

Temperature: 71°F

```

---

#### Step 4 – Send Tool Response to LLM

Client sends:

```

Tool result: 71°F

```

to the LLM.

---

#### Step 5 – Final Response

The LLM generates the final answer:

```

The weather in Miami is currently 71°F.

```

---

# 5. Downsides of Traditional Tool Calling

Traditional tool calling can have some issues.

### 1. Hallucinated Tool Calls
The LLM might suggest **tools that don't exist**.

Example:
```

Call tool: get_moon_weather

```

---

### 2. Incorrect Tool Usage
The LLM might:

- Pass wrong parameters
- Use the wrong tool
- Format requests incorrectly

---

### 3. Execution Responsibility
The **application developer must handle tool execution manually**.

---

# 6. Embedded Tool Calling

**Embedded Tool Calling** improves reliability by introducing a **library or framework between the application and the LLM**.

Instead of the application directly executing tools, the **library manages tool definitions and execution**.

---

# 7. Embedded Tool Calling Architecture

### Components

1. Client Application
2. Tool Library / Framework
3. LLM
4. External Tools

---

### Flow

1. Application sends message
2. Library attaches tool definitions
3. Message + tools sent to LLM
4. LLM selects tool
5. Tool call sent to library
6. Library executes tool
7. Tool result returned to LLM
8. LLM produces final answer

---

# 8. Embedded Tool Calling Example

User query:

```

What is the temperature in Miami?

```

### Step 1 – Application Sends Query

Application sends message to the **library**.

---

### Step 2 – Library Adds Tools

Library appends tool definitions and sends:

```

Message + Tools

```

to the LLM.

---

### Step 3 – LLM Requests Tool

LLM chooses:

```

get_weather(city="Miami")

```

---

### Step 4 – Library Executes Tool

Library automatically:

- Calls the weather API
- Retrieves weather data

Example result:

```

71°F

```

---

### Step 5 – Final Answer

The library sends the tool result back to the LLM.

LLM generates final response:

```

The temperature in Miami is 71°F.

```

---

# 9. Advantages of Embedded Tool Calling

### 1. Reduced Hallucination
Libraries help ensure **correct tool usage**.

---

### 2. Automatic Tool Execution
Developers **do not need to manually manage tool calls**.

---

### 3. Error Handling
Libraries can:

- Retry failed tool calls
- Validate parameters
- Fix formatting issues

---

### 4. Better Reliability
Frameworks make **tool usage more consistent and robust**.

---

# 10. Common Tool Types

LLMs can interact with many types of tools.

### APIs
- Weather APIs
- Payment APIs
- Maps APIs

### Databases
- SQL databases
- Vector databases

### Code Execution
- Python interpreter
- Data analysis tools

### External Models
- Translation models
- Image generation models

---

# 11. Traditional vs Embedded Tool Calling

| Feature | Traditional Tool Calling | Embedded Tool Calling |
|------|------|------|
| Tool execution | Handled by application | Handled by library |
| Reliability | Lower | Higher |
| Hallucination risk | Higher | Lower |
| Error handling | Manual | Automated |
| Development complexity | Higher | Lower |

---

# 12. Key Takeaway

**Tool Calling allows LLMs to interact with external systems**, enabling them to access real-time information and perform complex tasks.

Two approaches exist:

- **Traditional Tool Calling** – developer manages execution
- **Embedded Tool Calling** – libraries manage tools and execution automatically

Embedded tool calling provides **better reliability, automation, and reduced hallucinations**.
