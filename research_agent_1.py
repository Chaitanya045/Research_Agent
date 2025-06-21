import os
from openai import OpenAI
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from dotenv import load_dotenv
import asyncio
from duckduckgo_search import DDGS
import json
from datetime import datetime
import time
load_dotenv()
import logging

logger = logging.getLogger(__name__)
class AsyncRateLimiter:
    def __init__(self, max_calls, period_seconds):
        self.max_calls = max_calls
        self.period = period_seconds
        self.lock = asyncio.Lock()
        self.calls = []

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            self.calls = [t for t in self.calls if now - t < self.period]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0]) + 1  # Add 1 second buffer
                logger.info(f"Rate limit reached, waiting {sleep_time:.1f} seconds...")
                await asyncio.sleep(sleep_time)
                now = time.monotonic()
                self.calls = [t for t in self.calls if now - t < self.period]
            
            self.calls.append(time.monotonic())

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = "deepseek/deepseek-chat-v3-0324:free"
        # Rate limiter: 30 requests per minute (60 seconds)
        self.rate_limiter = AsyncRateLimiter(max_calls=30, period_seconds=60)
        # Context storage
        self.conversation_history = []
        self.max_context_length = 30  # Maximum number of messages to keep in context
    
    def add_to_context(self, role, content):
        """Add a message to the conversation context"""
        self.conversation_history.append({"role": role, "content": content})
        
        # Keep only the most recent messages to prevent context from growing too large
        if len(self.conversation_history) > self.max_context_length:
            self.conversation_history = self.conversation_history[-self.max_context_length:]
    
    def clear_context(self):
        """Clear the conversation history"""
        self.conversation_history = []
    
    def get_context(self):
        """Get the current conversation context"""
        return self.conversation_history.copy()
    
    def set_system_message(self, system_message):
        """Set or update the system message (will be the first message in context)"""
        # Remove existing system message if any
        self.conversation_history = [msg for msg in self.conversation_history if msg["role"] != "system"]
        # Add new system message at the beginning
        self.conversation_history.insert(0, {"role": "system", "content": system_message})
    
    async def chat(self, prompt, use_context=True, add_to_history=True):
        """
        Send a chat message with rate limiting and optional context
        
        Args:
            prompt: The user's message
            use_context: Whether to include conversation history in the request
            add_to_history: Whether to add this interaction to the conversation history
        """
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Prepare messages
        if use_context and self.conversation_history:
            messages = self.conversation_history.copy()
            messages.append({"role": "user", "content": prompt})
        else:
            messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            assistant_response = response.choices[0].message.content.strip()
            
            # Add to conversation history if requested
            if add_to_history:
                self.add_to_context("user", prompt)
                self.add_to_context("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error in chat request: {e}")
            raise
    
    def chat_sync(self, prompt, use_context=True, add_to_history=True):
        """
        Synchronous wrapper for the async chat method
        """
        return asyncio.run(self.chat(prompt, use_context, add_to_history))

class WebScraper: 
    def __init__(self):
        self.url = ""
        self.title = ''
        self.content = ''

    async def scrape_website(self, url):
        self.url = url
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            try:
                await page.goto(self.url, timeout=60000, wait_until='domcontentloaded')
                html_content = await page.content()
                soup = BeautifulSoup(html_content, 'html.parser')
                self.title = await page.title()

                if soup.body:
                    for irrelevant in soup.body(['script', 'img', 'style', 'input', 'nav', 'footer', 'header', 'aside']):
                        irrelevant.decompose()
                    self.content = soup.body.get_text(separator='\n',strip = True)
                else:
                    self.content = ''

            except Exception as e:
                print(f"Error scraping {self.url}: {e}")
                self.title = f"Scraping Error: {type(e).__name__}"
                self.webContent = ""

            finally:
                await browser.close()
            return {"content" : self.content, "title" : self.title}

class Search:

    def __init__(self):
        self.MAX_TOTAL_UNIQUE_RESULTS = 50

    def search(self, list_of_queries, max_results=10):
        unique_results = {}
        with DDGS() as ddgs:
            for query in list_of_queries:
                if len(unique_results) >= self.MAX_TOTAL_UNIQUE_RESULTS:
                    break
                try:
                    with DDGS() as ddgs:
                        results_for_current_query = ddgs.text(query, max_results=max_results)
                    for result_item in results_for_current_query:
                        href = result_item.get("href")

                        if href and href not in unique_results:
                            unique_results[href] = {"href": href}
                            
                            if len(unique_results) >= self.MAX_TOTAL_UNIQUE_RESULTS:
                                break
                except Exception as e:
                    print(f"Error searching for '{query}': {e}")
                time.sleep(1)
            final_unique_list = list(unique_results.values())
            print(f"Finished. Total unique results collected: {len(final_unique_list)}")
            return final_unique_list[:self.MAX_TOTAL_UNIQUE_RESULTS]

class SearchQueryGenerator:

    def __init__(self, llm):
        self.llm = llm
        self.search_queries = []
    
    async def extract(self, query):
        prompt = f"""
        You are a highly intelligent search query generator mainly focused on research for a given topic. Your task is to extract
        the most important keywords from the user's request and then generate a list of
        diverse and effective search queries (combinations of 1 to 5 words) that
        a search engine can use to find highly relevant information.

        Focus on variations that capture different facets of the original query.
        Ensure each generated query is concise and directly searchable.
        Return 3 to 5 queries as a comma-separated list. Do not include any other text or formatting.

        Example:
        Original Query: "What are the latest breakthroughs in renewable energy technology?"
        Output: renewable energy breakthroughs, latest renewable energy, renewable energy technology, green energy innovations, sustainable tech advancements

        Original Query: "Best practices for agile software development teams"
        Output: agile software development, agile team practices, scrum best practices, lean software development, agile methodologies

        Original Query: "How to train a dog to sit and stay?"
        Output: train dog sit, dog training stay, teaching dog commands, basic dog obedience, puppy sit stay

        Original Query: "{query}"
        Output:
        """
        result = self.llm.chat(prompt)
        
        keywords_line = result.strip().splitlines()[-1]
        self.search_queries = [word.strip() for word in keywords_line.split(",") if word.strip()]
        print(self.search_queries)
        return self.search_queries

class RelevanceChecker:
    def __init__(self, llm):
        self.llm = llm

    def is_relevant(self, query, content):
        prompt = f"""Is the following article relevant to the query: "{query}"?
Respond only with 'Yes' or 'No' with no explanation.

--- Article Content ---
{content[:2000]}"""  

        answer = self.llm.chat(prompt)
        return "yes" in answer.lower()
    

class JSONFormatter:
    def format(self, resources):
        return json.dumps(resources, indent=2)
    

class ResearchAgent:
    def __init__(self):
        self.llm = LLMClient()
        self.extractor = SearchQueryGenerator(self.llm)
        self.search = Search()
        self.scraper = WebScraper()
        self.checker = RelevanceChecker(self.llm)
        self.formatter = JSONFormatter()

    async def run(self, query):
        keywords = await self.extractor.extract(query)
        results = self.search.search(keywords)

        relevant_resources = []
        for result in results:
            content = await self.scraper.scrape_website(result["href"])
            if not content["content"]:
                continue
            if self.checker.is_relevant(query, content["content"]):
                summary = content["title"]
                relevant_resources.append({
                    "title": summary,
                    "url": result
                })

        return self.formatter.format(relevant_resources)
    
async def main():
    agent = ResearchAgent()
    query = input("Ask your research question: ")
    
    result = await agent.run(query)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"research_output_{timestamp}.json" 
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\nResearch completed. Results saved to: {filename}")

if __name__ == "__main__":
    asyncio.run(main())