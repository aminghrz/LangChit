from typing import List, Dict, Optional, Literal, TypedDict, Annotated
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Send
from pydantic import BaseModel, Field
import operator

from web_tools import search_web_internal, fetch_url_content_internal


# Pydantic models for structured outputs
class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


# State definitions
class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    research_topic: str


class QueryGenerationState(TypedDict):
    search_query: List[str]


class WebSearchState(TypedDict):
    search_query: str
    id: int


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: List[str]
    research_loop_count: int
    number_of_ran_queries: int


# Prompts
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format: 
- Format your response as a JSON object with ALL two of these exact keys:
  - "rationale": Brief explanation of why these queries are relevant
  - "query": A list of search queries

Context: {research_topic}"""


web_searcher_instructions = """Analyze the web search results and content to gather comprehensive information on "{research_topic}".

Instructions:
- The current date is {current_date}.
- Synthesize the information from the search results and web page contents.
- Focus on extracting key facts, insights, and relevant details.
- Maintain accuracy and cite sources where appropriate.
- Create a coherent summary that addresses the research topic.

Research Topic: {research_topic}

Search Results:
{search_results}

Web Page Contents:
{web_contents}"""


reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Summaries:
{summaries}"""


answer_instructions = """Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step. 
- Generate a high-quality answer to the user's question based on the provided summaries.
- Include citations using markdown format [source title](url).
- Be comprehensive yet concise.
- Organize information logically with clear sections if needed.

User Context: {research_topic}

Research Summaries:
{summaries}

Sources:
{sources}"""


# Node functions
def generate_query(state: OverallState, llm: ChatOpenAI) -> Dict:
    """Generate initial search queries"""
    
    # Get research topic from messages
    research_topic = state.get("research_topic", "")
    if not research_topic and state.get("messages"):
        # Extract from last human message
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                research_topic = msg.content
                break
    
    prompt = query_writer_instructions.format(
        current_date=get_current_date(),
        research_topic=research_topic,
        number_queries=state.get("initial_search_query_count", 3)
    )
    
    structured_llm = llm.with_structured_output(SearchQueryList)
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "search_query": result.query,
        "research_topic": research_topic
    }


def continue_to_web_research(state: QueryGenerationState) -> List[Send]:
    """Send queries for parallel web research"""
    return [
        Send("web_research", {"search_query": search_query, "id": idx})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, llm: ChatOpenAI) -> Dict:
    """Perform web research for a single query"""
    query = state["search_query"]
    
    # Search web
    search_results = search_web_internal(query, num_results=3)
    
    # Extract URLs and fetch content
    urls = [result['href'] for result in search_results if 'href' in result and not result.get('error')]
    
    web_contents = []
    if urls:
        # Fetch content for up to 3 URLs
        content_text = fetch_url_content_internal(urls[:3])
        web_contents.append({"content": content_text})
    
    # Format search results for LLM
    search_results_text = "\n".join([
        f"- {r['title']}: {r['body'][:200]}..." 
        for r in search_results if not r.get('error')
    ])
    
    # Use LLM to synthesize the information
    prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=query,
        search_results=search_results_text,
        web_contents=web_contents[0]["content"] if web_contents else "No content fetched"
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Gather sources
    sources = []
    for result in search_results:
        if not result.get('error'):
            sources.append({
                "title": result['title'],
                "url": result['href'],
                "snippet": result['body']
            })
    
    return {
        "search_query": [query],
        "web_research_result": [response.content],
        "sources_gathered": sources
    }

def reflection(state: OverallState, llm: ChatOpenAI) -> Dict:
    """Reflect on research and identify gaps"""
    
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    
    # Compile summaries
    summaries = "\n\n---\n\n".join(state.get("web_research_result", []))
    
    prompt = reflection_instructions.format(
        research_topic=state.get("research_topic", ""),
        summaries=summaries
    )
    
    structured_llm = llm.with_structured_output(Reflection)
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state.get("search_query", []))
    }


def evaluate_research(state: ReflectionState) -> Literal["finalize_answer"] | List[Send]:
    """Decide whether to continue research or finalize"""
    max_loops = state.get("max_research_loops", 2)
    
    if state["is_sufficient"] or state["research_loop_count"] >= max_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + idx,
                }
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, llm: ChatOpenAI) -> Dict:
    """Generate final research report"""
    
    # Compile all summaries
    summaries = "\n\n---\n\n".join(state.get("web_research_result", []))
    
    # Format sources
    unique_sources = {}
    for source in state.get("sources_gathered", []):
        url = source.get("url")
        if url and url not in unique_sources:
            unique_sources[url] = source
    
    sources_text = "\n".join([
        f"- [{s['title']}]({s['url']})"
        for s in unique_sources.values()
    ])
    
    prompt = answer_instructions.format(
        current_date=get_current_date(),
        research_topic=state.get("research_topic", ""),
        summaries=summaries,
        sources=sources_text
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Add sources section to the response
    final_content = f"{response.content}\n\n## Sources\n{sources_text}"
    
    return {
        "messages": [AIMessage(content=final_content)],
        "sources_gathered": list(unique_sources.values())
    }


# Create the deep research graph
def create_deep_research_graph(llm: ChatOpenAI):
    """Create the deep research workflow graph"""
    
    workflow = StateGraph(OverallState)
    
    # Define node functions with LLM
    def generate_query_node(state):
        return generate_query(state, llm)
    
    def web_research_node(state):
        return web_research(state, llm)
    
    def reflection_node(state):
        return reflection(state, llm)
    
    def finalize_answer_node(state):
        return finalize_answer(state, llm)
    
    # Add nodes
    workflow.add_node("generate_query", generate_query_node)
    workflow.add_node("web_research", web_research_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("finalize_answer", finalize_answer_node)
    
    # Add edges
    workflow.add_edge(START, "generate_query")
    workflow.add_conditional_edges(
        "generate_query", 
        continue_to_web_research, 
        ["web_research"]
    )
    workflow.add_edge("web_research", "reflection")
    workflow.add_conditional_edges(
        "reflection",
        evaluate_research,
        ["web_research", "finalize_answer"]
    )
    workflow.add_edge("finalize_answer", END)
    
    return workflow.compile()


# Main tool function
@tool
def deep_research(
    topic: str,
    max_iterations: int = 5,
    initial_queries: int = 5
) -> str:
    """
    Perform deep research on a topic using iterative web search and content analysis.
    
    Args:
        topic: The research topic or question
        max_iterations: Maximum number of research iterations (default: 5)
        initial_queries: Number of initial search queries to generate (default: 5)
    
    Returns:
        A comprehensive research report with citations
    """
    # This will be configured when the tool is created
    return "Deep research will be performed on: " + topic


# Factory function to create configured tool
def create_deep_research_tool(model: str, api_key: str, base_url: Optional[str] = None, default_iterations: int = 5):
    """Create a configured deep research tool"""
    
    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0.7,
        model=model,
        api_key=api_key,
        base_url=base_url
    )
    
    # Create graph
    graph = create_deep_research_graph(llm)
    
    @tool
    def deep_research_configured(
        topic: str,
        max_iterations: Optional[int] = None,
        initial_queries: int = 5
    ) -> str:
        """
        Perform deep research on a topic using iterative web search and content analysis.
        
        Args:
            topic: The research topic or question
            max_iterations: Maximum number of research iterations (uses configured default if not specified)
            initial_queries: Number of initial search queries to generate (default: 3)
        
        Returns:
            A comprehensive research report with citations
        """
        iterations = max_iterations if max_iterations is not None else default_iterations
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=topic)],
            "search_query": [],
            "web_research_result": [],
            "sources_gathered": [],
            "initial_search_query_count": initial_queries,
            "max_research_loops": iterations,
            "research_loop_count": 0,
            "research_topic": topic
        }
        
        try:
            # Run the graph
            result = graph.invoke(initial_state)
            
            # Extract the final report from messages
            if result.get("messages"):
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        return msg.content
            
            return "Research could not be completed."
        
        except Exception as e:
            return f"Error during research: {str(e)}"
    
    return deep_research_configured