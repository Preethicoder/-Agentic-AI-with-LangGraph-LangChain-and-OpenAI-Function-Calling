import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, FunctionMessage
import operator
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Define the get_weather tool
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # In a real application, you'd fetch actual weather data here.
    return f"The weather in {city} is sunny with 30°C."

# Define the give_joke tool
@tool
def give_joke() -> str:
    """Tell a funny joke."""
    return "Why don't scientists trust atoms? Because they make up everything!"

# Define the give_poem tool
@tool
def give_poem() -> str:
    """Recite a short poem."""
    return (
        "Roses are red,\n"
        "Violets are blue,\n"
        "Sugar is sweet,\n"
        "And so are you."
    )

# Define your tools - now including the new joke and poem tools
tools = [get_weather, give_joke, give_poem]

# Define the LLM
llm = ChatOpenAI(model ="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")).bind_tools(tools)

# Create the ToolNode to handle tool execution
# ToolNode directly takes the list of tools and executes them.
tool_node = ToolNode(tools)

# Define the graph state. This is crucial for LangGraph to manage messages and tool calls.
# We'll use a simple state with just 'messages'.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# Define the function for the agent's decision-making (LLM call)
def call_llm(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)
    # The LLM's response might include a tool call.
    return {"messages": [response]}

# Define the function to decide the next step in the graph
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM wants to call a tool, it will return a message with tool_calls.
    if last_message.tool_calls:
        return "call_tool"
    else:
        return "end"

# Create the LangGraph state machine
graph = StateGraph(AgentState)

# Add the nodes
graph.add_node("llm_node", call_llm) # Node for the LLM to decide
graph.add_node("tool_node", tool_node) # Node for executing tools

# Set the entry point
graph.set_entry_point("llm_node")

# Add edges:
# From LLM, decide whether to call a tool or end
graph.add_conditional_edges(
    "llm_node",
    should_continue,
    {
        "call_tool": "tool_node",
        "end": END
    }
)

# After tool execution, go back to the LLM to process the tool's output
graph.add_edge("tool_node", "llm_node")

# Compile the graph
app = graph.compile()
state = {"messages": []}
# --- Test the agent with different inputs ---
while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("👋 Bye!")
        break

    state["messages"].append(HumanMessage(content=user_input))
    state = app.invoke(state)

    # Get the latest AI response
    """ai_message = state["messages"][-1]
    print(f"Agent: {ai_message.content}\n")"""

    # 🔍 DEBUG: Print full message history
    print("\n🧠 Agent Memory (Full Messages So Far):")
    for i, msg in enumerate(state["messages"]):
        role = msg.__class__.__name__.replace("Message", "")  # Human / AI / Function
        print(f"{i + 1}. [{role}] {msg.content if hasattr(msg, 'content') else str(msg)}")
    print("\n" + "=" * 60 + "\n")
# For isolated test cases (reset state each time)
def run_isolated_test(prompt: str):
    print(f"--- Testing: {prompt} ---")
    test_state = {"messages": [HumanMessage(content=prompt)]}
    result = app.invoke(test_state)
    print(result["messages"][-1].content)
    print("\n" + "="*50 + "\n")

run_isolated_test("What is the weather in Berlin and tell me something hilarious ? and for the same city SUGGEST SOME TOURIST SPOT")
run_isolated_test("And suggest a good restaurant there")  # Will not know "there" since it's fresh state
"""print("--- Testing 'What is the weather in Berlin and tell me joke?' ---")
input_message_weather = HumanMessage(content="What is the weather in Berlin? and tell me something hilarious ? and for the same city SUGGEST SOME TOURIST SPOT")
response_weather = app.invoke({"messages": [input_message_weather]})
print(response_weather["messages"][-1].content)
print("\n" + "="*50 + "\n")

print("--- Testing 'Tell me a joke.' ---")
input_message_joke = HumanMessage(content="And suggest a good restaurant there")
response_joke = app.invoke({"messages": [input_message_joke]})
print(response_joke["messages"][-1].content)
print("\n" + "="*50 + "\n")

print("--- Testing 'Write a short poem.' ---")
input_message_poem = HumanMessage(content="Write a short poem.")
response_poem = app.invoke({"messages": [input_message_poem]})
print(response_poem["messages"][-1].content)
print("\n" + "="*50 + "\n")

print("--- Testing 'Hello, how are you?' (no tool call expected) ---")
input_message_greeting = HumanMessage(content="Hello, how are you?")
response_greeting = app.invoke({"messages": [input_message_greeting]})
print(response_greeting["messages"][-1].content)
print("\n" + "="*50 + "\n")"""