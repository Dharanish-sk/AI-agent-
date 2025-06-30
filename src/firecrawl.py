import os
from firecrawl import FirecrawlApp, ScrapeOptions
from dotenv import load_dotenv


load_dotenv()

class FirecrawlService:
    def __init__(self):
        api_key =os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY not set in environment variables")
        self.app = FirecrawlApp(api_key=api_key)
    def search_companies(self, query: str, limit: int = 5) :
        try:
            result = self.app.search(
                query = f"{query} company pricing",
                limit = limit,
                scrape_options={"formats": ["markdown"]}

            )
            # print("🔍 search result type:", type(result), "\nResult:", result)
            print("search_results sample:", result.data[:1])

            return result.data
        except Exception as e:
            print(f"Error occurred while searching companies: {e}")
            return []
        
    def scrape_company_pages(self, url: str) -> str:
        try:
            result = self.app.scrape_url(
                url=url,
                scrape_options={"formats": ["markdown"]}
            )
            return result.get("markdown", "")
        except Exception as e:
            print(f"Error occurred while scraping company pages: {e}")
            return ""
 