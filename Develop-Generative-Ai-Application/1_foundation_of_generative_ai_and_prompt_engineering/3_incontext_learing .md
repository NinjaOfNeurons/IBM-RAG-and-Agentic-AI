##  In-Context Learning (ICL)

**In-Context Learning** is a prompt engineering method where the model learns a new task from a small set of examples provided directly in the prompt.

*  Does **not** require additional training
*  No fine-tuning needed
*  Learns task behavior from examples inside the prompt

### 🔹 Advantages

* No fine-tuning required
* Reduces time and compute cost
* Can improve performance with good examples

### 🔹 Disadvantages

* Limited by context window size
* Complex tasks may require fine-tuning (gradient-based training)
* Performance depends heavily on prompt quality

---

##  What is a Prompt?

A **prompt** is the instruction or input given to a Large Language Model (LLM) to guide it toward a specific output.

It includes:

* Task instruction
* Input data
* (Optional) examples
* Output format guidance

---

## 🛠️ What is Prompt Engineering?

**Prompt Engineering** is the practice of designing and optimizing prompts to get accurate, relevant, and structured outputs from an LLM.

 It answers:

* **How to ask?**
* **What details to include?**
* **What format should the output follow?**

---

##  Why Prompt Engineering?

* Improves response accuracy
* Reduces hallucinations
* Controls output format
* Makes LLMs more reliable for real-world tasks
* Enables task-specific behavior without retraining

---

###  Summary

In-Context Learning allows LLMs to perform new tasks using only examples in the prompt — making prompt engineering a powerful and cost-effective alternative to fine-tuning.
