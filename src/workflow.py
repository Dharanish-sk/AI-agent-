from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import tiktoken
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .models import ResearchState, CompanyInfo, CompanyAnalysis
from .firecrawl import FirecrawlService
from .prompts import DeveloperToolsPrompts


@dataclass
class TokenBudget:
    """Token budget configuration for different operations."""
    scrape_per_url: int = 3000
    analysis_input: int = 8000
    extraction_input: int = 10000
    final_analysis: int = 15000
    model_max: int = 128000  # gpt-4o-mini context window


class TokenCounter:
    """Utility for counting tokens accurately."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for newer models
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, max_tokens: int, overlap: int = 200) -> List[str]:
        """Split text into chunks respecting token boundaries."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunks.append(self.encoding.decode(chunk_tokens))
            start = end - overlap  # Overlap to maintain context
        
        return chunks


class Workflow:
    """Agentic workflow for developer tools research with token-aware processing."""
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini", 
        temperature: float = 0.1,
        token_budget: Optional[TokenBudget] = None
    ):
        self.firecrawl = FirecrawlService()
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.prompts = DeveloperToolsPrompts()
        self.token_counter = TokenCounter(model)
        self.token_budget = token_budget or TokenBudget()
        self.workflow = self._build_workflow()
        
        # Telemetry for debugging
        self.metrics = {
            "tokens_processed": 0,
            "api_calls": 0,
            "scraped_urls": 0,
            "failed_operations": 0
        }
    
    def _build_workflow(self) -> StateGraph:
        """Build true agentic LangGraph workflow with map-reduce pattern."""
        graph = StateGraph(ResearchState)
        
        # Phase 1: Discovery
        graph.add_node("extract_tools", self._extract_tools_step)
        
        # Phase 2: Parallel research (MAP)
        graph.add_node("map_research", self._map_research_step)
        
        # Phase 3: Analysis aggregation (REDUCE)
        graph.add_node("reduce_analysis", self._reduce_analysis_step)
        
        # Phase 4: Final synthesis
        graph.add_node("synthesize", self._synthesize_step)
        
        # Build graph
        graph.set_entry_point("extract_tools")
        graph.add_edge("extract_tools", "map_research")
        graph.add_edge("map_research", "reduce_analysis")
        graph.add_edge("reduce_analysis", "synthesize")
        graph.add_edge("synthesize", END)
        
        return graph.compile()
    
    def _log_tokens(self, operation: str, tokens: int) -> None:
        """Log token usage for observability."""
        self.metrics["tokens_processed"] += tokens
        print(f"📊 {operation}: {tokens:,} tokens (Total: {self.metrics['tokens_processed']:,})")
    
    def _extract_tools_step(self, state: ResearchState) -> Dict[str, Any]:
        """Phase 1: Extract tool names from search results with chunking."""
        print(f"\n{'='*60}")
        print(f"🔍 PHASE 1: Tool Extraction")
        print(f"{'='*60}")
        print(f"Query: {state.query}\n")
        
        article_query = f"{state.query} tools comparison best alternatives"
        search_results = self.firecrawl.search_companies(article_query, num_results=5)
        
        if not search_results or not search_results.data:
            print("⚠️ No search results found")
            return {"extracted_tools": [], "extraction_metadata": {"sources": 0}}
        
        # Scrape with token awareness
        scraped_content = self._scrape_with_token_limit(
            search_results.data,
            max_total_tokens=self.token_budget.extraction_input
        )
        
        if not scraped_content:
            print("⚠️ No content scraped from search results")
            return {"extracted_tools": [], "extraction_metadata": {"sources": 0}}
        
        # Extract tools with chunking if needed
        tool_names = self._llm_extract_tools_chunked(state.query, scraped_content)
        
        metadata = {
            "sources": len(search_results.data),
            "tools_found": len(tool_names),
            "content_tokens": self.token_counter.count(scraped_content)
        }
        
        print(f"\n✅ Extracted {len(tool_names)} tools from {metadata['sources']} sources")
        print(f"Tools: {', '.join(tool_names[:5])}")
        if len(tool_names) > 5:
            print(f"... and {len(tool_names) - 5} more")
        
        return {
            "extracted_tools": tool_names,
            "extraction_metadata": metadata
        }
    
    def _scrape_with_token_limit(
        self, 
        results: List[Dict], 
        max_total_tokens: int
    ) -> str:
        """Scrape URLs while respecting token budget."""
        all_content = []
        tokens_used = 0
        
        for i, result in enumerate(results):
            url = result.get("url")
            if not url:
                continue
            
            # Check if we're approaching budget
            if tokens_used >= max_total_tokens * 0.9:  # 90% threshold
                print(f"⚠️ Token budget nearly exhausted, stopping at {i+1}/{len(results)} URLs")
                break
            
            try:
                scraped = self.firecrawl.scrape_company_pages(url)
                if not scraped or not scraped.markdown:
                    continue
                
                # Calculate remaining budget
                remaining_tokens = max_total_tokens - tokens_used
                per_url_limit = min(self.token_budget.scrape_per_url, remaining_tokens)
                
                # Chunk content to fit budget
                content = scraped.markdown
                content_tokens = self.token_counter.count(content)
                
                if content_tokens > per_url_limit:
                    # Truncate intelligently
                    content = self._intelligent_truncate(content, per_url_limit)
                    content_tokens = self.token_counter.count(content)
                    print(f"  ✂️ Truncated {url} to {content_tokens} tokens")
                
                all_content.append(f"# Source {i+1}: {url}\n\n{content}")
                tokens_used += content_tokens
                self.metrics["scraped_urls"] += 1
                
                print(f"  ✅ Scraped {url}: {content_tokens:,} tokens")
                
            except Exception as e:
                print(f"  ❌ Failed to scrape {url}: {e}")
                self.metrics["failed_operations"] += 1
                continue
        
        combined = "\n\n---\n\n".join(all_content)
        self._log_tokens("Scraping", tokens_used)
        
        return combined
    
    def _intelligent_truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text intelligently, preserving structure."""
        # Try to split by paragraphs first
        paragraphs = text.split('\n\n')
        result = []
        tokens_used = 0
        
        for para in paragraphs:
            para_tokens = self.token_counter.count(para)
            if tokens_used + para_tokens > max_tokens:
                # If we have room for at least half the paragraph, include it
                if tokens_used < max_tokens * 0.8:
                    # Take what we can
                    remaining = max_tokens - tokens_used
                    truncated = self.token_counter.chunk_text(para, remaining)[0]
                    result.append(truncated + "...")
                break
            result.append(para)
            tokens_used += para_tokens
        
        return '\n\n'.join(result)
    
    def _llm_extract_tools_chunked(self, query: str, content: str) -> List[str]:
        """Extract tools with chunking support for large content."""
        content_tokens = self.token_counter.count(content)
        
        # If content fits in budget, process directly
        if content_tokens <= self.token_budget.extraction_input:
            return self._llm_extract_tools(query, content)
        
        # Otherwise, chunk and process with map-reduce
        print(f"📚 Content too large ({content_tokens} tokens), using chunked extraction")
        
        chunks = self.token_counter.chunk_text(
            content, 
            self.token_budget.extraction_input - 1000  # Leave room for prompt
        )
        
        print(f"  Split into {len(chunks)} chunks")
        
        # MAP: Extract from each chunk
        all_tools = []
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}...")
            tools = self._llm_extract_tools(query, chunk)
            all_tools.extend(tools)
        
        # REDUCE: Deduplicate and rank
        unique_tools = list(dict.fromkeys(all_tools))  # Preserve order, remove dupes
        print(f"  Found {len(unique_tools)} unique tools across all chunks")
        
        return unique_tools
    
    def _llm_extract_tools(self, query: str, content: str) -> List[str]:
        """Use LLM to extract tool names from content."""
        messages = [
            SystemMessage(content=self.prompts.TOOL_EXTRACTION_SYSTEM),
            HumanMessage(content=self.prompts.tool_extraction_user(query, content))
        ]
        
        try:
            self.metrics["api_calls"] += 1
            response = self.llm.invoke(messages)
            
            # Count tokens in request/response
            request_tokens = sum(self.token_counter.count(m.content) for m in messages)
            response_tokens = self.token_counter.count(response.content)
            self._log_tokens("Tool Extraction", request_tokens + response_tokens)
            
            tool_names = [
                name.strip()
                for name in response.content.strip().split("\n")
                if name.strip() and not name.strip().startswith('#')
            ]
            return tool_names
        except Exception as e:
            print(f"❌ Tool extraction failed: {e}")
            self.metrics["failed_operations"] += 1
            return []
    
    def _map_research_step(self, state: ResearchState) -> Dict[str, Any]:
        """Phase 2 (MAP): Research each tool in parallel (conceptually)."""
        print(f"\n{'='*60}")
        print(f"🔬 PHASE 2: Parallel Tool Research (MAP)")
        print(f"{'='*60}\n")
        
        extracted_tools = getattr(state, "extracted_tools", [])
        
        if not extracted_tools:
            print("⚠️ No extracted tools, using fallback search")
            tool_names = self._fallback_tool_search(state.query, num_results=5)
        else:
            # Take top N tools
            tool_names = extracted_tools[:8]
        
        print(f"Researching {len(tool_names)} tools: {', '.join(tool_names)}\n")
        
        # MAP: Research each tool independently
        research_tasks = []
        for i, tool_name in enumerate(tool_names, 1):
            print(f"[{i}/{len(tool_names)}] Researching: {tool_name}")
            task_result = self._research_single_tool_with_budget(tool_name)
            research_tasks.append(task_result)
        
        # Filter out failed tasks
        companies = [task for task in research_tasks if task is not None]
        
        print(f"\n✅ Successfully researched {len(companies)}/{len(tool_names)} tools")
        
        return {"companies": companies, "research_metadata": {"attempted": len(tool_names)}}
    
    def _research_single_tool_with_budget(self, tool_name: str) -> Optional[CompanyInfo]:
        """Research a single tool with strict token budget."""
        try:
            # Search for official site
            search_results = self.firecrawl.search_companies(
                f"{tool_name} official site", 
                num_results=1
            )
            
            if not search_results or not search_results.data:
                print(f"  ⚠️ No results found")
                return None
            
            result = search_results.data[0]
            url = result.get("url", "")
            
            # Create base company info
            company = CompanyInfo(
                name=tool_name,
                description=result.get("markdown", "")[:500],
                website=url,
                tech_stack=[],
                competitors=[]
            )
            
            # Scrape with token limit
            scraped = self.firecrawl.scrape_company_pages(url)
            if scraped and scraped.markdown:
                # Ensure content fits in analysis budget
                content = scraped.markdown
                content_tokens = self.token_counter.count(content)
                
                if content_tokens > self.token_budget.analysis_input:
                    print(f"  ✂️ Truncating content from {content_tokens} to {self.token_budget.analysis_input} tokens")
                    content = self._intelligent_truncate(content, self.token_budget.analysis_input)
                
                analysis = self._analyze_company_content(company.name, content)
                self._enrich_company_info(company, analysis)
                print(f"  ✅ Analysis complete")
            else:
                print(f"  ⚠️ Could not scrape content")
            
            return company
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            self.metrics["failed_operations"] += 1
            return None
    
    def _analyze_company_content(self, company_name: str, content: str) -> CompanyAnalysis:
        """Analyze company content using structured LLM output."""
        structured_llm = self.llm.with_structured_output(CompanyAnalysis)
        
        messages = [
            SystemMessage(content=self.prompts.TOOL_ANALYSIS_SYSTEM),
            HumanMessage(content=self.prompts.tool_analysis_user(company_name, content))
        ]
        
        try:
            self.metrics["api_calls"] += 1
            
            # Log tokens
            request_tokens = sum(self.token_counter.count(m.content) for m in messages)
            self._log_tokens(f"Analysis ({company_name})", request_tokens)
            
            analysis = structured_llm.invoke(messages)
            return analysis
        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
            self.metrics["failed_operations"] += 1
            return CompanyAnalysis(
                pricing_model="Unknown",
                is_open_source=None,
                tech_stack=[],
                description="Analysis failed",
                api_available=None,
                language_support=[],
                integration_capabilities=[],
            )
    
    def _enrich_company_info(self, company: CompanyInfo, analysis: CompanyAnalysis) -> None:
        """Enrich company info with analysis results."""
        company.pricing_model = analysis.pricing_model
        company.is_open_source = analysis.is_open_source
        company.tech_stack = analysis.tech_stack
        company.description = analysis.description
        company.api_available = analysis.api_available
        company.language_support = analysis.language_support
        company.integration_capabilities = analysis.integration_capabilities
    
    def _reduce_analysis_step(self, state: ResearchState) -> Dict[str, Any]:
        """Phase 3 (REDUCE): Aggregate and deduplicate company data."""
        print(f"\n{'='*60}")
        print(f"📊 PHASE 3: Analysis Aggregation (REDUCE)")
        print(f"{'='*60}\n")
        
        companies = state.companies
        
        if not companies:
            print("⚠️ No companies to aggregate")
            return {"aggregated_companies": [], "aggregation_metadata": {}}
        
        # Deduplicate by website
        unique_companies = {}
        for company in companies:
            key = company.website.lower().strip('/')
            if key not in unique_companies:
                unique_companies[key] = company
            else:
                # Merge data if we have duplicate entries
                existing = unique_companies[key]
                # Keep the more complete description
                if len(company.description) > len(existing.description):
                    existing.description = company.description
                # Merge tech stacks
                existing.tech_stack = list(set(existing.tech_stack + company.tech_stack))
        
        aggregated = list(unique_companies.values())
        
        metadata = {
            "original_count": len(companies),
            "deduplicated_count": len(aggregated),
            "duplicates_removed": len(companies) - len(aggregated)
        }
        
        print(f"✅ Aggregated {metadata['original_count']} → {metadata['deduplicated_count']} unique companies")
        if metadata['duplicates_removed'] > 0:
            print(f"  Removed {metadata['duplicates_removed']} duplicates")
        
        return {
            "aggregated_companies": aggregated,
            "aggregation_metadata": metadata
        }
    
    def _synthesize_step(self, state: ResearchState) -> Dict[str, Any]:
        """Phase 4: Synthesize final recommendations with token management."""
        print(f"\n{'='*60}")
        print(f"🎯 PHASE 4: Final Synthesis")
        print(f"{'='*60}\n")
        
        companies = getattr(state, "aggregated_companies", state.companies)
        
        if not companies:
            return {
                "analysis": "No companies found to analyze. Please try a different query.",
                "synthesis_metadata": {"success": False}
            }
        
        # Serialize company data with token awareness
        company_summaries = []
        total_tokens = 0
        
        for company in companies:
            summary = self._create_company_summary(company)
            summary_tokens = self.token_counter.count(summary)
            
            # Check if we're approaching budget
            if total_tokens + summary_tokens > self.token_budget.final_analysis:
                print(f"⚠️ Token budget reached, including {len(company_summaries)}/{len(companies)} companies")
                break
            
            company_summaries.append(summary)
            total_tokens += summary_tokens
        
        company_data = "\n\n".join(company_summaries)
        
        print(f"📝 Synthesizing recommendations from {len(company_summaries)} companies")
        print(f"   Input tokens: {total_tokens:,}")
        
        messages = [
            SystemMessage(content=self.prompts.RECOMMENDATIONS_SYSTEM),
            HumanMessage(content=self.prompts.recommendations_user(state.query, company_data))
        ]
        
        try:
            self.metrics["api_calls"] += 1
            response = self.llm.invoke(messages)
            
            # Log final token usage
            request_tokens = sum(self.token_counter.count(m.content) for m in messages)
            response_tokens = self.token_counter.count(response.content)
            self._log_tokens("Final Synthesis", request_tokens + response_tokens)
            
            print("✅ Recommendations generated successfully")
            
            return {
                "analysis": response.content,
                "synthesis_metadata": {
                    "success": True,
                    "companies_included": len(company_summaries),
                    "input_tokens": total_tokens
                }
            }
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
            self.metrics["failed_operations"] += 1
            return {
                "analysis": "Failed to generate recommendations. Please try again.",
                "synthesis_metadata": {"success": False}
            }
    
    def _create_company_summary(self, company: CompanyInfo) -> str:
        """Create a concise company summary for final analysis."""
        return f"""**{company.name}**
Website: {company.website}
Description: {company.description[:300]}...
Pricing: {company.pricing_model}
Open Source: {company.is_open_source}
API Available: {company.api_available}
Tech Stack: {', '.join(company.tech_stack[:5])}
Languages: {', '.join(company.language_support[:5])}
Integrations: {', '.join(company.integration_capabilities[:5])}"""
    
    def _fallback_tool_search(self, query: str, num_results: int = 5) -> List[str]:
        """Fallback method to find tools via direct search."""
        search_results = self.firecrawl.search_companies(query, num_results=num_results)
        
        if not search_results or not search_results.data:
            return []
        
        return [
            result.get("metadata", {}).get("title", f"Tool {i}")
            for i, result in enumerate(search_results.data, 1)
        ]
    
    def run(self, query: str) -> ResearchState:
        """Execute the complete agentic workflow with full observability."""
        print(f"\n{'#'*60}")
        print(f"🚀 STARTING AGENTIC WORKFLOW")
        print(f"{'#'*60}")
        print(f"Query: {query}")
        print(f"Model: {self.llm.model_name}")
        print(f"Token Budget: {self.token_budget.model_max:,}")
        print(f"{'#'*60}\n")
        
        # Reset metrics
        self.metrics = {
            "tokens_processed": 0,
            "api_calls": 0,
            "scraped_urls": 0,
            "failed_operations": 0
        }
        
        initial_state = ResearchState(query=query)
        
        try:
            final_state = self.workflow.invoke(initial_state)
            result = ResearchState(**final_state)
            
            self._print_final_summary(result)
            
            return result
        except Exception as e:
            print(f"\n{'#'*60}")
            print(f"❌ WORKFLOW FAILED")
            print(f"{'#'*60}")
            print(f"Error: {e}")
            print(f"Metrics at failure: {self.metrics}")
            print(f"{'#'*60}\n")
            raise
    
    def _print_final_summary(self, result: ResearchState) -> None:
        """Print comprehensive workflow summary."""
        print(f"\n{'#'*60}")
        print(f"✅ WORKFLOW COMPLETED SUCCESSFULLY")
        print(f"{'#'*60}")
        
        print(f"\n📊 METRICS:")
        print(f"  Total Tokens: {self.metrics['tokens_processed']:,}")
        print(f"  API Calls: {self.metrics['api_calls']}")
        print(f"  URLs Scraped: {self.metrics['scraped_urls']}")
        print(f"  Failed Operations: {self.metrics['failed_operations']}")
        
        if hasattr(result, 'extraction_metadata'):
            print(f"\n🔍 EXTRACTION:")
            print(f"  Sources: {result.extraction_metadata.get('sources', 0)}")
            print(f"  Tools Found: {result.extraction_metadata.get('tools_found', 0)}")
        
        if hasattr(result, 'aggregation_metadata'):
            print(f"\n📊 AGGREGATION:")
            print(f"  Companies Researched: {result.aggregation_metadata.get('original_count', 0)}")
            print(f"  Unique Companies: {result.aggregation_metadata.get('deduplicated_count', 0)}")
        
        if hasattr(result, 'synthesis_metadata'):
            print(f"\n🎯 SYNTHESIS:")
            print(f"  Success: {result.synthesis_metadata.get('success', False)}")
            print(f"  Companies Included: {result.synthesis_metadata.get('companies_included', 0)}")
        
        efficiency = (1 - self.metrics['failed_operations'] / max(1, self.metrics['api_calls'])) * 100
        print(f"\n⚡ EFFICIENCY: {efficiency:.1f}%")
        
 
