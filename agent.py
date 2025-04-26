from typing import List
from playwright.sync_api import sync_playwright, TimeoutError
from pydantic import BaseModel, Field, PrivateAttr
from langchain.schema import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

class JobSchema(BaseModel):
    jobList: List[str] = Field(
        ...,
        description="List of job‐posting URLs to scrape",
    )
    numberWords: int = Field(
        default=10,
        description="How many keywords to extract per description",
    )
    descriptions: List[str] = Field(default_factory=list, exclude=True)
    responses: List[BaseMessage] = Field(default_factory=list, exclude=True)

model = ChatOpenAI(model="o4-mini", verbose=True)

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

builder = StateGraph(JobSchema)

builder.add_node(
    "scrape_job_url", 
    scrape_descriptions
)
builder.add_node(
    "extract_keywords", 
    extract_keywords
)

builder.add_edge(START, "scrape_job_url")
builder.add_edge("scrape_job_url", "extract_keywords")
builder.add_edge("extract_keywords", END)

graph = builder.compile()
