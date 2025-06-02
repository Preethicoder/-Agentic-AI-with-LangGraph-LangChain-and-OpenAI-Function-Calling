from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolExecutor, create_tool_calling_node
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import tool
from typing import TypedDict, Annotated
import operator

# --- Define a simple tool ---
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"The weather in {city} is sunny with a temperature of 30°C."

@tool
def tell_joke() -> str:
    """Tell a funny joke."""
    return "Why don't scientists trust atoms? Because they make up everything!"

tools = [get_weather, tell_joke]
tool_executor = ToolExecutor(tools)

# --- LLM model ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Tool-calling agent node ---
agent_node = create_tool_calling_node(llm, tool_executor)

# --- Agent memory ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# --- LangGraph state machine ---
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.set_entry_point("agent")
builder.set_finish_point("agent")
app = builder.compile()

# --- Interactive loop with memory ---
print("🤖 Agent is ready. Type 'exit' to stop.\n")

state = {"messages": []}

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("👋 Bye!")
        break

    state["messages"].append(HumanMessage(content=user_input))
    state = app.invoke(state)

    # Get the latest AI response
    ai_message = state["messages"][-1]
    print(f"Agent: {ai_message.content}\n")
