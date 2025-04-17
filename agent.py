from typing import List
from playwright.sync_api import sync_playwright

from pydantic import BaseModel, Field, PrivateAttr
from langchain.schema import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# 1) Define your schema with only the real inputs public
class JobSchema(BaseModel):
    jobList:      List[str] = Field(
        ...,
        description="List of job‐posting URLs to scrape",
    )
    numberWords:  int        = Field(
        default=10,
        description="How many keywords to extract per description",
    )
    # private buffers, never treated as inputs
    descriptions: List[str] = Field(default_factory=list, exclude=True)
    responses: List[BaseMessage] = Field(default_factory=list, exclude=True)

# 2) Init your LLM
model = ChatOpenAI(model="o4-mini", verbose=True)

# 3) Scrape node: fetch each URL and pull out the JD text
def scrape_descriptions(state: JobSchema):
    """
    Scrape each URL in state.jobList for its job description,
    and store the raw text in state.descriptions.
    """
    descs: List[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        for url in state.jobList:
            page.goto(url, wait_until="networkidle")
            # adjust this selector to match the real description container
            job_div = page.query_selector("div.job-description, div#jobDescriptionText")
            if not job_div:
                job_div = page.query_selector("body")
            descs.append(job_div.inner_text().strip())
        browser.close()
    state.descriptions = descs
    print("🏷  scraped descriptions:", descs)
    return {"descriptions": descs}

# 4) LLM node: build prompts from the scraped text and invoke
def extract_keywords(state: JobSchema):
    """
    Build prompts asking the LLM to extract the top
    `state.numberWords` keywords from each scraped description.
    Store the LLM’s BaseMessage list in state.responses.
    """
    prompts: List[BaseMessage] = []
    print("🏷  scraped descriptions 2:", state.descriptions)

    for desc in state.descriptions:
        prompts.append(SystemMessage(
            content=(
                f"Extract the top {state.numberWords} keywords "
                f"from the following job description:\n\n{desc}"
            )
        ))
    print("🏷  prompts:", prompts)
    response = model.invoke(prompts)
    state.responses = response
    # print("🏷  response:", 'response')
    return {"responses": state.responses}

# 5) Final node: pull out the text and return it as the graph’s output
def emit_keywords(state: JobSchema):
    """
    Pull the .content field from each BaseMessage in state.responses
    and return it under the key "keywords" as the graph’s output.
    """
    return {"keywords": [m.content for m in state.responses]}

# 6) Wire it all up in a StateGraph
builder = StateGraph(JobSchema)

# wrap each step in a ToolNode so LangGraph knows these are “owned” tools
builder.add_node(
    "scrape", 
    scrape_descriptions
)
builder.add_node(
    "llm", 
    extract_keywords
)
builder.add_node(
    "emit", 
    emit_keywords
)

# define edges: START → scrape → llm → emit → END
builder.add_edge(START, "scrape")
builder.add_edge("scrape", "llm")
builder.add_edge("llm", "emit")
builder.add_edge("emit", END)
# builder.add_edge("llm", END)

graph = builder.compile()
