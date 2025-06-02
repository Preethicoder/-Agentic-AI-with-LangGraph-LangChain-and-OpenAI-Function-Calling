# 🤖 Agentic AI with LangGraph, LangChain, and OpenAI Function Calling

This project demonstrates a simple yet powerful **agentic AI system** using [LangGraph](https://docs.langchain.com/langgraph/), [LangChain Tools](https://docs.langchain.com/docs/expression-language/tools/), and [OpenAI's Function Calling](https://platform.openai.com/docs/guides/function-calling). The agent can:
- 🌤 Get weather info
- 😂 Tell a joke
- 📝 Recite a short poem

It dynamically decides whether to respond directly or call tools based on user input — mimicking the reasoning and planning behaviors of more complex AI agents.

---

## 🧠 Features

- ✅ Agent loop using LangGraph's state machine
- ✅ Multi-step reasoning (tool → LLM → next tool)
- ✅ Function calling with OpenAI's `gpt-4o-mini` model
- ✅ LangChain `ToolNode` support for modular tool execution
- ✅ Stateful memory across a conversation (in one session)

---

## 📦 Tech Stack

| Component        | Description                             |
|------------------|-----------------------------------------|
| Python           | Core programming language               |
| LangGraph        | Agentic loop control                    |
| LangChain Tools  | For weather, jokes, and poems           |
| OpenAI GPT-4o    | Decision-making & reasoning             |
| Dotenv           | For environment variable management     |

---

## 🚀 Getting Started

### 1. Clone the repo

   ```bash
    git clone https://github.com/yourusername/agentic-ai-langgraph.git
    cd agentic-ai-langgraph
```

### 2. Install dependencies
You’ll need:

---langgraph
---langchain
---openai
---python-dotenv

### 3. Add your OpenAI API Key
Create a .env file in the project root:
OPENAI_API_KEY=your_openai_api_key_here


### 📜 Sample Output
You: What is the weather in Berlin and tell me something hilarious?

### 🧠 Agent Memory (Full Messages So Far):
1. [Human] What is the weather in Berlin and tell me something hilarious?
2. [AI] The weather in Berlin is sunny with 30°C.
3. [Function] Why don't scientists trust atoms? Because they make up everything!

### 🧠 How It Works (LangGraph Flow)
graph TD
    A[HumanMessage] --> B[LLM Node]
    B --> |If Tool Call| C[Tool Node]
    C --> B
    B --> |If Done| D[End]
