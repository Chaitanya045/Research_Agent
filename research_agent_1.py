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

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('GROQ_API_KEY'),  
            base_url="https://api.groq.com/openai/v1/"
        )
        self.model = "deepseek-r1-distill-llama-70b"

    def chat(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

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