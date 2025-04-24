from typing import List
from playwright.sync_api import sync_playwright, TimeoutError
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
            try:
                page.goto(url, wait_until="domcontentloaded")
                # wait for our target container to appear (or timeout)
                page.wait_for_selector("div.job-description, div#jobDescriptionText",
                                       timeout=3000)
                job_div = page.query_selector("div.job-description, div#jobDescriptionText")
            except TimeoutError:
                print(f"⚠️  {url} timed out; using <body> instead")
                job_div = page.query_selector("body")

            text = job_div.inner_text().strip() if job_div else ""
            descs.append(text)
        browser.close()
    
    state.descriptions = descs
    
    return {"descriptions": descs}

# 4) LLM node: build prompts from the scraped text and invoke
def extract_keywords(state: JobSchema):
    """
    Build prompts asking the LLM to extract the top
    `state.numberWords` keywords from each scraped description.
    Store the LLM’s BaseMessage list in state.responses.
    """
    prompts: List[BaseMessage] = []

    for desc in state.descriptions:
        prompts.append(SystemMessage(
            content=(
                f"Extract the top {state.numberWords} keywords "
                f"from the following job description:\n\n{desc}"
            )
        ))
    
    response = model.invoke(prompts)
    state.responses = response

    return {"responses": state.responses}

# 5) Final node: pull out the text and return it as the graph’s output
def emit_keywords(state: JobSchema):
    """
    Pull the .content field from each BaseMessage in state.responses
    and return it under the key "keywords" as the graph’s output.
    """
    return {"keywords": state.responses.content}

# 6) Wire it all up in a StateGraph
builder = StateGraph(JobSchema)

# wrap each step in a ToolNode so LangGraph knows these are “owned” tools
builder.add_node(
    "scrape_job_url", 
    scrape_descriptions
)
builder.add_node(
    "extract_keywords", 
    extract_keywords
)
builder.add_node(
    "display_keywords", 
    emit_keywords
)

# define edges: START → scrape → llm → emit → END
builder.add_edge(START, "scrape_job_url")
builder.add_edge("scrape_job_url", "extract_keywords")
builder.add_edge("extract_keywords", "display_keywords")
builder.add_edge("display_keywords", END)

graph = builder.compile()
