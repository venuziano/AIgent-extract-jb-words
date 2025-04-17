from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field, PrivateAttr
from langchain.schema import BaseMessage, SystemMessage 

class JobSchema(BaseModel):
    jobList: list[str] = Field(
        ...,
        description="Job list",
    )
    numberWords: int = Field(
        description="Number of words to extract",
        default=10
    )
    keyWords: list[BaseMessage] = Field(default_factory=list)

# Model
model = ChatOpenAI(model="gpt-4o")

# Node
def call_llm(state: JobSchema):
    # (…build prompts…)
    prompts = []
    if state.jobList:
        prompts.append(
            SystemMessage(content=f"tell me more about: {', '.join(state.jobList)}")
        )
    prompts.extend(state.keyWords)

    response = model.invoke(prompts)
    
    return {"keyWords": response}

# Build graph
builder = StateGraph(JobSchema)

builder.add_node("call_llm", call_llm)

builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", END)

# Compile graph
graph = builder.compile()