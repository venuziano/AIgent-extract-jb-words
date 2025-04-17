from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage, BaseMessage

class JobSchema(BaseModel):
    jobList: list[str] = Field(
        description="Job list",
        default_factory=list
    )
    numberWords: int = Field(
        description="Number of words to extract",
        default=10
    )
    messages: list[BaseMessage] = Field(default_factory=list)

# Model
model = ChatOpenAI(model="gpt-4o")

# Node
def tool_calling_llm(state: JobSchema):
    raw = state.jobList
    # normalize job_list as before…
    job_list = (
        [item.strip() for item in raw.split(",") if item.strip()]
        if isinstance(raw, str)
        else [str(item).strip() for item in raw]
    )

    # build a **list** of schema‑correct messages
    msgs: list[BaseMessage] = []
    if job_list:
        msgs.append(SystemMessage(content=f"tell me more about: {', '.join(job_list)}"))
    msgs.extend(state.messages)

    # now this will work
    response_messages = model.invoke(msgs)
    return {"messages": response_messages}

# Build graph
builder = StateGraph(JobSchema)

builder.add_node("tool_calling_llm", tool_calling_llm)

builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)

# Compile graph
graph = builder.compile()