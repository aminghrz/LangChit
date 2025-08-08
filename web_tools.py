from typing import List, Dict, Literal
from datetime import datetime
from langchain_core.tools import tool
from ddgs import DDGS
from langchain_community.document_loaders import WebBaseLoader, SeleniumURLLoader
import requests
from bs4 import BeautifulSoup


def search_web_internal(query: str, num_results: int = 5, timelimit: Literal["d", "w", "m", "y"] = "w") -> List[Dict]:
    """Internal web search function that returns raw results"""
    try:
        results = DDGS().text(query, max_results=num_results, timelimit=timelimit)
        timestamp = datetime.now().isoformat()
        
        return [
            {
                "query": query,
                "title": result.get("title", ""),
                "href": result.get("href", ""),
                "body": result.get("body", ""),
                "timestamp": timestamp,
                "text": f"{result.get('title', '')} {result.get('body', '')}"
            }
            for result in results
        ]
    except Exception as e:
        return [{"query": query, "error": str(e), "timestamp": datetime.now().isoformat()}]


def needs_javascript(url: str) -> bool:
    """Check if a page likely needs JavaScript to render content"""
    try:
        response = requests.get(url, timeout=5, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        soup = BeautifulSoup(response.text, 'html.parser')

        # Check for signs that content is loaded via JavaScript
        if soup.find('div', id='root') or soup.find('div', id='app'):
            return True

        body_text = soup.body.get_text(strip=True) if soup.body else ""
        if len(body_text) < 100:
            return True

        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and any(framework in script.string for framework in ['React', 'Vue', 'Angular', '__NEXT_DATA__']):
                return True

        if soup.find(text=lambda t: t and 'loading' in t.lower()):
            return True

        return False
    except:
        return True


def fetch_url_content_internal(urls: List[str]) -> str:
    """Internal URL content fetcher that returns formatted content"""
    all_content = []

    for url in urls:
        try:
            if needs_javascript(url):
                loader = SeleniumURLLoader(
                    urls=[url],
                    headless=True,
                    arguments=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    ]
                )
                docs = loader.load()
                all_content.append(f"Content from {url} (dynamic):\n{docs[0].page_content}\n")
            else:
                loader = WebBaseLoader([url])
                docs = loader.load()
                all_content.append(f"Content from {url} (static):\n{docs[0].page_content}\n")
        except Exception as e:
            all_content.append(f"Error loading {url}: {str(e)}\n")

    return '\n'.join(all_content)


def create_search_web_tool(store, user_id, num_results: int = 5, search_method_rag: bool = True):
    """Factory function to create configured search_web tool"""
    
    @tool
    def search_web(query: str, timelimit: Literal["d", "w", "m", "y"] = "w") -> str:
        """Search the web for information about a topic.
        
        Args:
            query: The search query
            timelimit: Time limit - "d" (day), "w" (week), "m" (month), "y" (year)
        
        Returns:
            Search results from the web
        """
        results = search_web_internal(query, num_results, timelimit)
        
        # Store results in vector store
        namespace = ("web_search", user_id)
        for i, result in enumerate(results):
            if not result.get("error"):
                key = f"{query}_{result['timestamp']}_{i}"
                value = result
                store.put(
                    namespace=namespace,
                    key=key,
                    value=value,
                    index=["text"]
                )
        
        # Return based on search method
        if search_method_rag:
            final_results = store.search(namespace, query=query, limit=1)
        else:
            final_results = results
            
        return str(final_results)
    
    return search_web


@tool
def fetch_url_content(urls: List[str]) -> str:
    """This is a tool to retrieve contents of a list of URLs
    
    Args:
        urls: a list of URLs you want to extract their content from
    
    Returns:
        A large string which contains all texts in the given URLs.
    """
    return fetch_url_content_internal(urls)