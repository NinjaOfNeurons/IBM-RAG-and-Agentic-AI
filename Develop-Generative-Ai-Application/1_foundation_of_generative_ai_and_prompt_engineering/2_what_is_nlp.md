## Natural Language Processing (NLP)

**Natural Language Processing (NLP)** is a field of AI that enables computers to understand, interpret, and generate human language.

---

##  Unstructured vs Structured Data

### 🔹 Unstructured Data (Human Language)

Example:

```
Add eggs and milk to shopping list
```

### 🔹 Structured Data

Example:

```xml
<shopping_list>
  <item>eggs</item>
  <item>milk</item>
</shopping_list>
```

**NLP sits between structured and unstructured data.**

* **Unstructured ➜ Structured** → **NLU (Natural Language Understanding)**
* **Structured ➜ Unstructured** → **NLG (Natural Language Generation)**

---

##  Applications of NLP

### 🔹 NLU Examples

* Intent recognition
* Information extraction
* Named Entity Recognition (NER)

### 🔹 NLG Examples

* Machine Translation
* Virtual Assistants / Chatbots
* Text Summarization
* Content Generation

### 🔹 Other NLP Applications

* Sentiment Analysis
* Spam Detection

---

##  Stages of NLP

1. **Tokenization**
   Breaking text into words or sentences.
   Example:
   `"I am running"` → `["I", "am", "running"]`

2. **Stemming**
   Reducing words to root form (may not be a real word).

   * running → run
   * ran → run
   * better → bet (incorrect root)

3. **Lemmatization**
   Reducing words to meaningful base form (dictionary word).

   * running → run
   * better → good

4. **Part-of-Speech (POS) Tagging**
   Identifies grammatical role of words.

   * "I am going to **make** dinner" → *make = verb*
   * "What **make** is your laptop?" → *make = noun*

5. **Named Entity Recognition (NER)**
   Identifies entities like:

   * Person names
   * Locations
   * Organizations
   * Dates

---

###  Summary

NLP bridges human language and machine understanding by converting:

* Text ➜ Meaning (NLU)
* Meaning ➜ Text (NLG)

It is the foundation of chatbots, search engines, AI assistants, and language models.
