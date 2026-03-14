
# Implementing LangChain AI-Powered SQL Agent – Quick Notes

## 1. Environment Setup
1. Create Python virtual environment

```sql
   virtualenv my_env
```

1. Install required libraries

   * ibm-watsonx-ai
   * langchain
   * mysql-connector-python

These libraries enable:

* AI model integration
* LangChain functionality
* Database connectivity

---

# 2. Launch MySQL Server

Steps:

1. Open **MySQL window**
2. Click **Create**
3. Launch **MySQL server**
4. Wait ~15 seconds for activation
5. Open **MySQL CLI** to access the terminal

---

# 3. Load Chinook Sample Database

Chinook Database:

* Sample **digital media store dataset**
* Contains tables like:

  * Artists
  * Albums
  * Tracks
  * Media types
* Connected via **Entity Relationship Diagram (ERD)**

### Download Database

```bash
wget <chinook-mysql.sql URL>
```

### Load SQL Script

```sql
SOURCE chinook-mysql.sql;
```

### Verify Database

```sql
SHOW DATABASES;
```

### Example Test Query

```sql
USE Chinook;
SELECT COUNT(*) FROM Album;
```

Expected result:
**347 albums**

---

# 4. Load IBM watsonx.ai Granite LLM

### Step 1: Store Credentials

Create dictionary containing:

* API key
* Authentication details

### Step 2: Configure Model Parameters

Important parameters:

**MAX_NEW_TOKENS**

* Maximum tokens generated in response

**TEMPERATURE**

* Controls randomness
* Lower → deterministic
* Higher → creative

### Step 3: Set Project Information

* project_id = `"skills-network"`
* space_id = `None` (if not used)

Optional:

```python
verify=False
```

Used in unsecured/local environments.

---

# 5. Initialize watsonx Model in LangChain

Import classes:

* `Model` from `ibm_watsonx_ai.foundation_models`
* `WatsonxLLM` from `langchain_ibm`

Provide:

* model_id
* credentials
* params
* project_id
* space_id (optional)

`WatsonxLLM` allows integration with:

* LangChain chains
* SQL agents
* chatbots
* vector stores
* reasoning pipelines

---

# 6. Connect to MySQL Database

Define connection parameters:

* mysql_username
* mysql_password
* mysql_host
* mysql_port (default **3306**)
* database_name = **Chinook**

### Build Connection URI

LangChain method:

```python
SQLDatabase.from_URI()
```

Creates connection between **LangChain and MySQL database**.

---

# 7. Create SQL Agent

Import:

```python
create_sql_agent
```

Agent connects:
**LLM + Database**

Parameters:

* `llm`
* `db`
* `verbose=True`
* `agent_type="zero-shot"`

Zero-shot agent:

* Performs reasoning before executing actions.

---

# 8. Run Natural Language Query

Example:

```
How many albums are listed in the database?
```

Process:

1. User enters **natural language query**
2. LLM converts query → **SQL**
3. SQL executed on **database**
4. Results returned
5. LLM formats answer in **natural language**

Expected response:
**347 albums**

---

# 9. Verbose Mode

Setting:

```python
verbose=True
```

Displays:

* reasoning steps
* generated SQL queries
* intermediate actions
* final answer

Useful for:

* debugging
* learning agent behavior

---

# 10. Implementation Summary

Steps to implement SQL agent:

1. Create **Python virtual environment**
2. Install required libraries
3. Launch **MySQL server**
4. Load **Chinook database**
5. Configure **IBM watsonx Granite LLM**
6. Connect **LangChain to MySQL**
7. Create **SQL agent**
8. Run **natural language queries**

Result:
Users can query databases using **natural language instead of SQL**.
