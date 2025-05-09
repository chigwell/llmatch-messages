[![PyPI version](https://badge.fury.io/py/llmatch-messages.svg)](https://badge.fury.io/py/llmatch-messages)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/badge/llmatch-messages)](https://pepy.tech/project/llmatch-messages)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://www.linkedin.com/in/eugene-evstafev-716669181/)

# llmatch-messages

`llmatch-messages` is a robust utility for **pattern-verified LLM interactions** built on top of [`langchain-llm7`](https://pypi.org/project/langchain-llm7/) and [`langchain-core`](https://pypi.org/project/langchain-core/). It enables structured, retryable conversations with LLMs by matching regex patterns in their responses, ideal for use cases where **consistency and format** are crucial — such as parsing XML/JSON-style tags, code blocks, or key phrases.

This package is production-oriented, supports retries with exponential backoff, and provides rich diagnostics when responses do not match expectations.

---

## 🔧 Installation

```bash
pip install llmatch-messages
````

This will also install:

* `langchain-llm7==2025.05.91116`
* `langchain-core==0.3.51`

---

## ✨ Example

```python
from llmatch_messages import llmatch
from langchain_llm7 import ChatLLM7
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatLLM7()

response = llmatch(
    llm=llm,
    messages=[
        SystemMessage(content="You are a helpful assistant. Write the output in the format: <image_desc>...</image_desc>"),
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": {"url": "https://llm7.io/logo.png"}},
            ],
        )
    ],
    pattern=r"<image_desc>\s*(.*?)\s*</image_desc>",
    verbose=True,
)

if response["success"]:
    print("Extracted:", response["extracted_data"])
else:
    print("Error:", response["error_message"])
```

---

## 🧠 Key Features

* **Pattern Matching**: Use regex to validate and extract structured parts of LLM responses.
* **Retry Logic**: Automatically retries if output does not conform, using exponential backoff.
* **LangChain Native**: Works seamlessly with `langchain_core.messages` and `langchain-llm7`.
* **Message-Aware**: Operates on `BaseMessage` list (e.g. `HumanMessage`, `SystemMessage`).
* **Detailed Diagnostics**: Verbose mode traces all steps and decision points.
* **Fail-Safe**: Handles malformed LLM responses gracefully and provides fallback messaging.

---

## 🧪 Return Format

The function returns a dictionary:

```python
{
    "success": True | False,
    "extracted_data": Optional[List[str]],
    "final_content": Optional[str],
    "retries_attempted": int,
    "error_message": Optional[str],
    "raw_response": Optional[Any],
}
```

---

## 🪄 Common Use Cases

* Validate XML-like or markdown-formatted LLM outputs.
* Parse code blocks (` ```...``` `), JSON sections, or tags.
* Build structured LLM workflows that require machine-readable responses.

---

## 📄 License

Licensed under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).

---

## 🙋‍♂️ Author

Developed by [Eugene Evstafev](https://www.linkedin.com/in/eugene-evstafev-716669181/), software developer at the University of Cambridge, creator of [llm7.io](https://llm7.io).

For feedback or contributions, feel free to open an issue or PR on [GitHub](https://github.com/chigwell/llmatch-messages).
