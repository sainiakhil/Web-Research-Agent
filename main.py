import streamlit as st
import google.generativeai as genai
import os
import time
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from newsapi import NewsApiClient
from dotenv import load_dotenv
import re
from urllib.parse import urlparse


# --- Configuration & API Key Loading ---
# load_dotenv()
GOOGLE_API_KEY =  st.secrets["GOOGLE_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
GNEWS_API_KEY = st.secrets["GNEWS_API_KEY"]



# Max chars to extract from a webpage to keep context manageable
MAX_CHARS_PER_PAGE = 4000
# Max number of search results to process
MAX_RESULTS_TO_PROCESS = 3


# --- Real Tool Implementations ---

def tavily_web_search(query: str, api_key: str, max_results: int = 5) -> list:
    """Performs web search using Tavily API."""
    st.info(f" TAVILY SEARCH: Searching for '{query}'...")
    if not api_key:
        st.error("Tavily API key is missing.")
        return []
    try:
        tavily = TavilyClient(api_key=api_key)
        response = tavily.search(query=query, search_depth="basic", max_results=max_results) # basic is often enough
        st.success(f" TAVILY SEARCH: Found {len(response.get('results', []))} results.")
        # Return URL, title, and potentially snippets if needed later
        return [{"url": res["url"], "title": res["title"], "content": res.get("content", "")} for res in response.get("results", [])]
    except Exception as e:
        st.error(f"Tavily search failed: {e}")
        return []

def gnews_search(query: str, api_key: str, max_results: int = 5) -> list:
    """Fetches news articles using GNews API."""
    st.info(f" GNEWS SEARCH: Searching for news about '{query}'...")
    if not api_key:
        st.error("GNews API key is missing.")
        return []
    try:
        newsapi = NewsApiClient(api_key = api_key)
        news = newsapi.get_everything(q= query, language='en', sort_by='relevancy', page_size=max_results)
        articles = news.get('articles', [])
        st.success(f" GNEWS SEARCH: Found {len(news)} articles.")
        # Adapt GNews output to a consistent format
        return [{"url": article["url"], "title": article["title"], "content": article.get("description", "")} for article in articles]
    except Exception as e:
        st.error(f"GNews search failed: {e}")
        return []


def scrape_website_content(url: str) -> dict:
    """Scrapes text content from a given URL using Requests and BeautifulSoup."""
    st.info(f" SCRAPER: Attempting to scrape {url}...") # Restored Streamlit feedback
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    content = None
    status = "Failure: Unknown"
    error_message = ""

    try:
        response = requests.get(url, headers=headers, timeout=15) # Increased timeout slightly
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            status = f"Failure: Non-HTML content ({content_type})"
            st.warning(f" SCRAPER: Skipped {url} - Content type is not HTML ({content_type}).") # Restored
            return {"url": url, "status": status, "text": None, "error": status}


        soup = BeautifulSoup(response.content, 'lxml') # Use lxml parser

        # Remove script and style elements
        for script_or_style in soup(["script", "style", "nav", "footer", "header"]): # Added nav/footer/header removal
            script_or_style.decompose()

        # Try to find main content areas (common tags/attributes)
        main_content = soup.find('main') or \
                       soup.find('article') or \
                       soup.find('div', class_=re.compile(r'(main|content|post|body|article|story)', re.IGNORECASE)) or \
                       soup.body # Fallback to body

        if main_content:
            # Get text, separate paragraphs, remove excessive whitespace
            paragraphs = main_content.find_all('p', recursive=True) # Search recursively within main_content
            if paragraphs:
                 text_list = [p.get_text(strip=True) for p in paragraphs]
                 text = "\n".join(filter(None, text_list)) # Join non-empty paragraphs
            else: # If no <p> tags found in main area, get all text from it
                text = main_content.get_text(separator='\n', strip=True)

            # Clean up excessive newlines
            text = re.sub(r'\n\s*\n', '\n\n', text).strip()

            # Limit character count
            if len(text) > MAX_CHARS_PER_PAGE:
                 text = text[:MAX_CHARS_PER_PAGE] + "..."
                 st.warning(f" SCRAPER: Truncated content from {url} to {MAX_CHARS_PER_PAGE} chars.") # Restored

            content = text
            status = "Success"
            st.success(f" SCRAPER: Successfully scraped and extracted text from {url}.") # Restored

        else:
            status = "Failure: Could not find main content area."
            st.warning(f" SCRAPER: Failed to find main content area in {url}.") # Restored
            error_message = status

    # Catch network/HTTP errors
    except requests.exceptions.RequestException as e:
        status = f"Failure: Request Error ({type(e).__name__})"
        error_message = str(e)
        st.error(f" SCRAPER: Request failed for {url} - {error_message}") # Restored
    # Catch other unexpected errors during the process
    except Exception as e:
        status = f"Failure: Unexpected Error ({type(e).__name__})"
        error_message = str(e)
        st.error(f" SCRAPER: An unexpected error occurred processing {url} - {error_message}") # Restored


    return {"url": url, "status": status, "text": content, "error": error_message}


# --- Agent Core Logic ---

def run_research_agent(user_query: str, gemini_model, tavily_key: str, gnews_key: str):
    """
    Orchestrates the web research process using real tools.
    """
    st.write("--- Starting Research Process ---")
    start_time = time.time()

    # 1. Basic Query Analysis (Identify if it's primarily a news query)
    is_news_query = any(keyword in user_query.lower() for keyword in ["news", "latest", "update", "today", "recent"])
    st.write(f"Query identified as {'News-focused' if is_news_query else 'General Research'}.")

    # 2. Select Search Tool and Perform Search
    search_results = []
    if is_news_query:
        with st.spinner("Searching for recent news articles via GNews..."):
            search_results = gnews_search(user_query, gnews_key, max_results=5)
    else:
         with st.spinner("Performing general web search via Tavily..."):
            search_results = tavily_web_search(user_query, tavily_key, max_results=5)

    if not search_results:
        st.error("No search results found.")
        return "Could not find any relevant information through the selected search tool."

    st.subheader("Sources Found:")
    for i, res in enumerate(search_results):
        st.write(f"{i+1}. [{res['title']}]({res['url']})")
        if res.get('content'): # Display description/snippet if available
             st.caption(f"   Snippet: {res['content'][:150]}...")

    # 3. Content Extraction & Basic Filtering
    collected_data = []
    urls_processed = set()

    st.subheader(f"Processing up to {MAX_RESULTS_TO_PROCESS} Sources...")
    with st.spinner(f"Scraping and extracting content..."):
        processed_count = 0
        for result in search_results:
            if processed_count >= MAX_RESULTS_TO_PROCESS:
                st.write(f"Reached processing limit ({MAX_RESULTS_TO_PROCESS} sources).")
                break

            url = result["url"]
            # Basic check to avoid re-processing identical URLs sometimes returned by APIs
            parsed_url = urlparse(url)
            domain_path = f"{parsed_url.netloc}{parsed_url.path}".rstrip('/')
            if domain_path in urls_processed:
                st.write(f"Skipping duplicate URL: {url}")
                continue

            scraped_info = scrape_website_content(url)

            if scraped_info["status"] == "Success" and scraped_info["text"]:
                collected_data.append({
                    "url": url,
                    "title": result['title'],
                    "text_content": scraped_info["text"]
                    # No explicit relevance/reliability score here; rely on Gemini + source diversity
                })
                urls_processed.add(domain_path)
                processed_count += 1
            else:
                st.warning(f"Could not process content from {url} - {scraped_info.get('error', 'Scraping failed')}")
                # Optionally try the next result if one fails early


    if not collected_data:
        st.error("Failed to extract usable content from any of the found sources.")
        return "No relevant information could be extracted from the sources."

    # 4. Information Synthesis (Using Gemini)
    st.subheader("Synthesizing Information with Gemini...")
    with st.spinner("Compiling final report using Gemini..."):
        # Prepare context for Gemini
        context = ""
        for i, data in enumerate(collected_data):
            context += f"Source {i+1}:\n"
            context += f"URL: {data['url']}\n"
            context += f"Title: {data['title']}\n"
            context += f"Extracted Text (Truncated):\n{data['text_content']}\n\n"
            context += "---\n\n"

        prompt = f"""
        User Query: "{user_query}"

        Based *only* on the following text extracted from various web sources, please synthesize a comprehensive research report answering the user query.

        Extracted Information:
        {context}
        ---

        Instructions:
        1.  Carefully analyze the user query and the provided text from each source.
        2.  Construct a coherent report that directly addresses the query.
        3.  Use information from the sources provided. **Do not add external knowledge or information not present in the text extracts.**
        4.  When presenting information derived from a specific source, **cite the source URL** (e.g., "According to [URL], ...").
        5.  Combine insights logically. If sources conflict, state the conflicting information and attribute it to the respective sources (e.g., "Source [URL1] states X, while source [URL2] suggests Y.").
        6.  If the provided text is insufficient to fully answer the query, clearly state the limitations.
        7.  Format the report clearly using markdown (headings, lists, bold text). Aim for accuracy and conciseness based *solely* on the provided extracts.
        """

        try:
            # Ensure the model is configured before calling generate_content
            if not gemini_model:
                 raise ValueError("Gemini model not initialized.")
            response = gemini_model.generate_content(prompt)
            final_report = response.text
            st.success("Gemini finished generating the report.")
        except Exception as e:
            st.error(f"Error calling Gemini API: {e}")
            # Attempt to access candidate text even on error, if available
            try:
                 final_report = response.candidates[0].content.parts[0].text
                 st.warning("Gemini API reported an error, but some content might have been generated.")
            except (AttributeError, IndexError, Exception):
                 final_report = "An error occurred during Gemini synthesis, and no fallback content was available."


    end_time = time.time()
    st.write(f"--- Research Process Complete (Duration: {end_time - start_time:.2f} seconds) ---")
    return final_report

# --- Streamlit UI ---

st.set_page_config(page_title="Web Research Agent", layout="wide")
st.title(" Web Research Agent ")
st.caption("Powered by Tavily, GNews, Requests+BS4 & Google Gemini")



# User Query Input
user_query = st.text_area("Enter your research query:", height=100, key="query_input", placeholder="e.g., What are the latest advancements in renewable energy sources? or Latest news on the stock market")

# Run Button
if st.button("Start Research", key="start_button"):

    try:
        genai.configure(api_key = GOOGLE_API_KEY)
        # Using Flash as requested (latest version) - check Gemini docs for current best model name
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-001')
        st.success("Gemini configured successfully.")

        # Run the agent
        final_report = run_research_agent(user_query, gemini_model, TAVILY_API_KEY, GNEWS_API_KEY)

        # Display Final Report
        st.markdown("---") # Separator
        st.subheader(" Final Research Report")
        st.markdown(final_report)

    except Exception as e:
        st.error(f"An error occurred during setup or execution: {e}")


# Sidebar Information
st.sidebar.header("About")
st.sidebar.info(
    "This agent uses real APIs to perform web research:"
    "\n- **Tavily:** General web search."
    "\n- **GNews:** News article search."
    "\n- **Requests/BeautifulSoup:** Web page scraping."
    "\n- **Google Gemini:** Information synthesis."
)
st.sidebar.header("How it Works")
st.sidebar.markdown(
    """
    1.  **Query Input**: Enter your query & API keys.
    2.  **Search**: Detects if it's news-focused (uses GNews) or general (uses Tavily).
    3.  **Scrape**: Fetches content from top search result URLs using Requests & BeautifulSoup.
    4.  **Synthesize (Gemini)**: Sends extracted text to Google Gemini, asking it to generate a report answering your query, citing the source URLs.
    *Disclaimer: Web scraping can be unreliable and may fail on complex sites or sites with anti-scraping measures.*
    """
)

