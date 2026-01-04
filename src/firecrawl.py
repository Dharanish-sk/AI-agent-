import os
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

load_dotenv()

class FirecrawlService:
    def __init__(self):
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY not set in environment variables")
        self.app = FirecrawlApp(api_key=api_key)
    
    def search_companies(self, query: str, num_results: int = 5):  # Changed: limit -> num_results
        """Search for companies with Firecrawl."""
        try:
            result = self.app.search(
                query=query,  # Removed the "company pricing" suffix - let caller control
                limit=num_results,
                scrape_options={"formats": ["markdown"]}
            )
            
            print(f"🔍 Found {len(result.data) if result and result.data else 0} search results")
            
            # Return the full result object, not just result.data
            return result  # CHANGED: was return result.data
            
        except Exception as e:
            print(f"❌ Error searching: {e}")
            return None  # CHANGED: return None instead of []
    
    def scrape_company_pages(self, url: str):
        """Scrape a company page and return the result object."""
        try:
            result = self.app.scrape_url(
                url=url,
                scrape_options={"formats": ["markdown"]}
            )
            
            # Return the full result object
            return result  # CHANGED: was return result.get("markdown", "")
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return None  # CHANGED: return None instead of ""
