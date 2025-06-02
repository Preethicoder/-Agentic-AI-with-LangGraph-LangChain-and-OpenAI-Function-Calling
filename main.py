import os
import operator
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load environment variables (like your OpenAI API key)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==========================
# 🔧 Define Tools
# ==========================

@tool
def get_weather(city: str) -> str:
    """Simulated tool to return weather info for a city."""
    return f"The weather in {city} is sunny with 30°C."

@tool
def give_joke() -> str:
    """Tool to return a joke."""
    return "Why don't scientists trust atoms? Because they make up everything!"

@tool
def give_poem() -> str:
    """Tool to return a short poem."""
    return (
        "Roses are red,\n"
        "Violets are blue,\n"
        "Sugar is sweet,\n"
        "And so are you."
    )

tools = [get_weather, give_joke, give_poem]

# ==========================
# 🤖 Set Up LLM and Tool Binding
# ==========================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
).bind_tools(tools)

# ToolNode will automatically call tools when needed
tool_node = ToolNode(tools)

# ==========================
# 🧠 Define Agent State
# ==========================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# ==========================
# 🔁 Define Graph Logic
# ==========================

# Node: LLM decision logic (whether to call tools or not)
def call_llm(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Decide whether to end or call a tool
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    return "end"

# ==========================
# 🔄 Build the LangGraph Agent
# ==========================

graph = StateGraph(AgentState)

graph.add_node("llm_node", call_llm)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("llm_node")

graph.add_conditional_edges(
    "llm_node",
    should_continue,
    {
        "call_tool": "tool_node",
        "end": END,
    }
)

graph.add_edge("tool_node", "llm_node")

app = graph.compile()

# ==========================
# 💬 Interactive CLI Mode
# ==========================

state = {"messages": []}

print("🤖 Type your questions (type 'exit' or 'quit' to stop):")
while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("👋 Bye!")
        break

    state["messages"].append(HumanMessage(content=user_input))
    state = app.invoke(state)

    print("\n🧠 Agent Memory (Messages So Far):")
    for i, msg in enumerate(state["messages"]):
        role = msg.__class__.__name__.replace("Message", "")  # Human / AI / Function
        print(f"{i + 1}. [{role}] {msg.content if hasattr(msg, 'content') else str(msg)}")
    print("\n" + "=" * 60 + "\n")

# ==========================
# 🧪 Isolated Test Runner (Stateless)
# ==========================

def run_isolated_test(prompt: str):
    print(f"--- Testing: {prompt} ---")
    test_state = {"messages": [HumanMessage(content=prompt)]}
    result = app.invoke(test_state)
    print(result["messages"][-1].content)
    print("\n" + "=" * 50 + "\n")

# Example test runs (optional)
run_isolated_test("What is the weather in Berlin and tell me something hilarious? And for the same city suggest some tourist spot.")
run_isolated_test("And suggest a good restaurant there.")  # Won't have context of 'there'
