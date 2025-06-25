#!/usr/bin/env python3
"""
ShadowForge OS v5.1 - Advanced Web Scraping Engine
Intelligent content extraction and business intelligence gathering
"""

import asyncio
import aiohttp
import json
import logging
import time
import re
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, parse_qs
from pathlib import Path
import hashlib
import random

# Safe imports with fallbacks
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    import requests_html
    REQUESTS_HTML_AVAILABLE = True
except ImportError:
    REQUESTS_HTML_AVAILABLE = False

@dataclass
class ScrapingTarget:
    """Target website for scraping."""
    url: str
    name: str
    selectors: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    rate_limit: float = 1.0  # seconds between requests
    use_selenium: bool = False
    requires_login: bool = False
    login_credentials: Optional[Dict[str, str]] = None
    priority: str = "normal"
    categories: List[str] = field(default_factory=list)

@dataclass
class ScrapedContent:
    """Scraped content with metadata."""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    quality_score: float = 0.0
    content_type: str = "unknown"
    word_count: int = 0
    sentiment_score: float = 0.0

class AdvancedWebScrapingEngine:
    """
    Advanced web scraping engine with intelligent content extraction.
    
    Features:
    - Multi-method scraping (requests, selenium, requests-html)
    - Intelligent content extraction and parsing
    - Rate limiting and respectful crawling
    - Anti-detection measures
    - Content quality assessment
    - Business intelligence gathering
    - Trend monitoring and analysis
    - Automated content categorization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.WebScrapingEngine")
        self.session = None
        self.selenium_driver = None
        self.is_initialized = False
        
        # Scraping configuration
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0"
        ]
        
        # Content extraction patterns
        self.content_selectors = {
            "title": ["h1", "title", ".title", "#title", "[data-title]"],
            "content": ["article", ".content", ".post-content", ".entry-content", "main", ".article-body"],
            "description": ["meta[name='description']", ".description", ".summary", ".excerpt"],
            "author": [".author", ".byline", "[data-author]", ".post-author"],
            "date": [".date", ".published", "[datetime]", ".post-date", "time"],
            "price": [".price", ".cost", "[data-price]", ".amount", ".value"],
            "rating": [".rating", ".score", ".stars", "[data-rating]"],
            "tags": [".tags", ".categories", ".keywords", ".labels"]
        }
        
        # Business intelligence targets
        self.business_targets = {
            "competitors": {
                "selectors": {
                    "products": ".product, .item, .listing",
                    "prices": ".price, .cost, .amount",
                    "features": ".features, .specs, .description",
                    "reviews": ".review, .rating, .feedback"
                }
            },
            "trends": {
                "selectors": {
                    "trending": ".trending, .popular, .hot",
                    "metrics": ".count, .views, .shares, .likes",
                    "hashtags": ".hashtag, .tag, [data-hashtag]"
                }
            },
            "news": {
                "selectors": {
                    "headlines": "h1, h2, h3, .headline, .title",
                    "content": "article, .article-body, .content",
                    "timestamp": "time, .date, .published"
                }
            }
        }
        
        # Performance tracking
        self.scraping_stats = {
            "total_requests": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "total_content_extracted": 0,
            "average_response_time": 0.0,
            "blocked_requests": 0
        }
        
        # Content storage
        self.scraped_content = []
        self.content_cache = {}
        self.failed_urls = set()
        
        # Rate limiting
        self.last_request_time = {}
        self.request_delays = {}
        
    async def initialize(self):
        """Initialize the web scraping engine."""
        self.logger.info("Initializing Advanced Web Scraping Engine...")
        
        try:
            # Initialize HTTP session
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "User-Agent": random.choice(self.user_agents),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1"
                }
            )
            
            # Initialize Selenium if available
            if SELENIUM_AVAILABLE:
                await self._initialize_selenium()
            
            self.is_initialized = True
            self.logger.info("Advanced Web Scraping Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Web Scraping Engine: {e}")
            raise
    
    async def _initialize_selenium(self):
        """Initialize Selenium WebDriver."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(f"--user-agent={random.choice(self.user_agents)}")
            
            self.selenium_driver = webdriver.Chrome(options=chrome_options)
            self.logger.info("Selenium WebDriver initialized")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize Selenium: {e}")
            self.selenium_driver = None
    
    async def scrape_url(self, url: str, target_config: Optional[ScrapingTarget] = None) -> ScrapedContent:
        """
        Scrape content from a single URL.
        """
        if not self.is_initialized:
            await self.initialize()
        
        self.logger.info(f"Scraping URL: {url}")
        
        # Apply rate limiting
        await self._apply_rate_limit(url)
        
        try:
            # Determine scraping method
            if target_config and target_config.use_selenium and self.selenium_driver:
                content = await self._scrape_with_selenium(url, target_config)
            else:
                content = await self._scrape_with_requests(url, target_config)
            
            # Extract and process content
            processed_content = await self._process_scraped_content(content, url)
            
            # Update statistics
            self.scraping_stats["successful_scrapes"] += 1
            self.scraping_stats["total_content_extracted"] += len(processed_content.content)
            
            # Cache content
            self.content_cache[url] = processed_content
            self.scraped_content.append(processed_content)
            
            return processed_content
            
        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            self.scraping_stats["failed_scrapes"] += 1
            self.failed_urls.add(url)
            
            # Return empty content with error info
            return ScrapedContent(
                url=url,
                title="Scraping Failed",
                content=f"Failed to scrape content: {str(e)}",
                metadata={"error": str(e), "status": "failed"}
            )
    
    async def _scrape_with_requests(self, url: str, target_config: Optional[ScrapingTarget]) -> str:
        """Scrape using aiohttp requests."""
        headers = {}
        if target_config and target_config.headers:
            headers.update(target_config.headers)
        
        # Rotate user agent
        headers["User-Agent"] = random.choice(self.user_agents)
        
        start_time = time.time()
        
        async with self.session.get(url, headers=headers) as response:
            self.scraping_stats["total_requests"] += 1
            
            if response.status == 200:
                content = await response.text()
                response_time = time.time() - start_time
                self._update_response_time(response_time)
                return content
            elif response.status == 429:
                self.scraping_stats["blocked_requests"] += 1
                self.logger.warning(f"Rate limited on {url}")
                await asyncio.sleep(random.uniform(5, 15))
                raise Exception(f"Rate limited: {response.status}")
            else:
                raise Exception(f"HTTP {response.status}: {response.reason}")
    
    async def _scrape_with_selenium(self, url: str, target_config: ScrapingTarget) -> str:
        """Scrape using Selenium WebDriver."""
        if not self.selenium_driver:
            raise Exception("Selenium WebDriver not available")
        
        try:
            self.selenium_driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.selenium_driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for dynamic content if needed
            await asyncio.sleep(2)
            
            return self.selenium_driver.page_source
            
        except Exception as e:
            raise Exception(f"Selenium scraping failed: {e}")
    
    async def _process_scraped_content(self, html_content: str, url: str) -> ScrapedContent:
        """Process and extract meaningful content from HTML."""
        if not BS4_AVAILABLE:
            # Fallback processing without BeautifulSoup
            return ScrapedContent(
                url=url,
                title=self._extract_title_fallback(html_content),
                content=self._extract_text_fallback(html_content),
                metadata={"method": "fallback"}
            )
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract basic information
        title = self._extract_element(soup, self.content_selectors["title"])
        content = self._extract_element(soup, self.content_selectors["content"])
        description = self._extract_element(soup, self.content_selectors["description"])
        
        # Extract metadata
        metadata = {
            "author": self._extract_element(soup, self.content_selectors["author"]),
            "date": self._extract_element(soup, self.content_selectors["date"]),
            "description": description,
            "keywords": self._extract_keywords(soup),
            "language": soup.get("lang", "en"),
            "canonical_url": self._extract_canonical_url(soup)
        }
        
        # Extract links and images
        links = [urljoin(url, a.get("href", "")) for a in soup.find_all("a", href=True)]
        images = [urljoin(url, img.get("src", "")) for img in soup.find_all("img", src=True)]
        
        # Clean and process content
        clean_content = self._clean_content(content)
        word_count = len(clean_content.split())
        
        # Assess content quality
        quality_score = self._assess_content_quality(clean_content, title, metadata)
        
        # Determine content type
        content_type = self._classify_content_type(clean_content, title, url)
        
        return ScrapedContent(
            url=url,
            title=title,
            content=clean_content,
            metadata=metadata,
            links=links[:50],  # Limit links
            images=images[:20],  # Limit images
            quality_score=quality_score,
            content_type=content_type,
            word_count=word_count
        )
    
    def _extract_element(self, soup, selectors: List[str]) -> str:
        """Extract element using multiple selectors."""
        for selector in selectors:
            try:
                if selector.startswith("[") or selector.startswith(".") or selector.startswith("#"):
                    # CSS selector
                    elements = soup.select(selector)
                    if elements:
                        return self._get_element_text(elements[0])
                else:
                    # Tag name
                    element = soup.find(selector)
                    if element:
                        return self._get_element_text(element)
            except:
                continue
        return ""
    
    def _get_element_text(self, element) -> str:
        """Get text from element, handling different element types."""
        if element.name == "meta":
            return element.get("content", "")
        elif element.name in ["input", "textarea"]:
            return element.get("value", "")
        else:
            return element.get_text(strip=True)
    
    def _extract_keywords(self, soup) -> List[str]:
        """Extract keywords from meta tags."""
        keywords = []
        
        # Meta keywords
        meta_keywords = soup.find("meta", attrs={"name": "keywords"})
        if meta_keywords:
            keywords.extend([k.strip() for k in meta_keywords.get("content", "").split(",")])
        
        # Extract from title and headings
        title = soup.find("title")
        if title:
            keywords.extend(title.get_text().split())
        
        for heading in soup.find_all(["h1", "h2", "h3"]):
            keywords.extend(heading.get_text().split())
        
        return list(set(keywords))[:20]  # Limit and deduplicate
    
    def _extract_canonical_url(self, soup) -> str:
        """Extract canonical URL."""
        canonical = soup.find("link", attrs={"rel": "canonical"})
        return canonical.get("href", "") if canonical else ""
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        if not content:
            return ""
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common noise
        noise_patterns = [
            r'cookie\s+policy',
            r'privacy\s+policy',
            r'terms\s+of\s+service',
            r'subscribe\s+to\s+newsletter',
            r'follow\s+us\s+on',
            r'share\s+this\s+article'
        ]
        
        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _assess_content_quality(self, content: str, title: str, metadata: Dict[str, Any]) -> float:
        """Assess the quality of extracted content."""
        score = 0.0
        
        # Content length
        if len(content) > 500:
            score += 0.3
        elif len(content) > 100:
            score += 0.2
        
        # Title quality
        if title and len(title) > 10:
            score += 0.2
        
        # Metadata richness
        if metadata.get("author"):
            score += 0.1
        if metadata.get("date"):
            score += 0.1
        if metadata.get("description"):
            score += 0.1
        
        # Content structure
        if "." in content:  # Has sentences
            score += 0.1
        
        # Language detection (basic)
        english_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
        content_lower = content.lower()
        english_count = sum(1 for word in english_words if word in content_lower)
        if english_count >= 3:
            score += 0.1
        
        return min(1.0, score)
    
    def _classify_content_type(self, content: str, title: str, url: str) -> str:
        """Classify the type of content."""
        content_lower = content.lower()
        title_lower = title.lower() if title else ""
        url_lower = url.lower()
        
        # News/Article
        if any(keyword in content_lower for keyword in ["published", "reporter", "breaking", "news"]):
            return "news"
        
        # Product/E-commerce
        if any(keyword in content_lower for keyword in ["price", "buy", "cart", "shipping", "product"]):
            return "product"
        
        # Blog/Opinion
        if any(keyword in content_lower for keyword in ["opinion", "think", "believe", "blog", "post"]):
            return "blog"
        
        # Technical/Tutorial
        if any(keyword in content_lower for keyword in ["tutorial", "guide", "how to", "step", "install"]):
            return "tutorial"
        
        # Social Media
        if any(keyword in url_lower for keyword in ["twitter", "facebook", "instagram", "linkedin", "tiktok"]):
            return "social"
        
        return "general"
    
    def _extract_title_fallback(self, html_content: str) -> str:
        """Fallback title extraction without BeautifulSoup."""
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        return title_match.group(1).strip() if title_match else "No Title"
    
    def _extract_text_fallback(self, html_content: str) -> str:
        """Fallback text extraction without BeautifulSoup."""
        # Remove script and style elements
        clean_html = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        clean_html = re.sub(r'<style[^>]*>.*?</style>', '', clean_html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', clean_html)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    async def _apply_rate_limit(self, url: str):
        """Apply rate limiting for respectful crawling."""
        domain = urlparse(url).netloc
        
        if domain in self.last_request_time:
            time_since_last = time.time() - self.last_request_time[domain]
            delay = self.request_delays.get(domain, 1.0)
            
            if time_since_last < delay:
                sleep_time = delay - time_since_last
                await asyncio.sleep(sleep_time)
        
        self.last_request_time[domain] = time.time()
    
    def _update_response_time(self, response_time: float):
        """Update average response time statistics."""
        current_avg = self.scraping_stats["average_response_time"]
        total_requests = self.scraping_stats["total_requests"]
        
        if total_requests == 1:
            self.scraping_stats["average_response_time"] = response_time
        else:
            self.scraping_stats["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
    
    async def scrape_multiple_urls(self, urls: List[str], concurrent_limit: int = 5) -> List[ScrapedContent]:
        """Scrape multiple URLs concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def scrape_with_semaphore(url: str) -> ScrapedContent:
            async with semaphore:
                return await self.scrape_url(url)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, ScrapedContent):
                valid_results.append(result)
            else:
                self.logger.error(f"Scraping task failed: {result}")
        
        return valid_results
    
    async def gather_business_intelligence(self, target_domain: str, categories: List[str] = None) -> Dict[str, Any]:
        """Gather business intelligence from target domain."""
        self.logger.info(f"Gathering business intelligence for {target_domain}")
        
        intelligence = {
            "domain": target_domain,
            "timestamp": datetime.now().isoformat(),
            "categories": categories or ["general"],
            "findings": {}
        }
        
        try:
            # Scrape main page
            main_content = await self.scrape_url(f"https://{target_domain}")
            
            # Extract business information
            intelligence["findings"]["main_page"] = {
                "title": main_content.title,
                "content_quality": main_content.quality_score,
                "word_count": main_content.word_count,
                "content_type": main_content.content_type,
                "links_count": len(main_content.links),
                "images_count": len(main_content.images)
            }
            
            # Analyze content for business insights
            content_analysis = await self._analyze_business_content(main_content.content)
            intelligence["findings"]["content_analysis"] = content_analysis
            
            # Check for specific business indicators
            business_indicators = await self._extract_business_indicators(main_content)
            intelligence["findings"]["business_indicators"] = business_indicators
            
            return intelligence
            
        except Exception as e:
            self.logger.error(f"Failed to gather business intelligence: {e}")
            intelligence["error"] = str(e)
            return intelligence
    
    async def _analyze_business_content(self, content: str) -> Dict[str, Any]:
        """Analyze content for business insights."""
        analysis = {
            "keywords": [],
            "topics": [],
            "sentiment": "neutral",
            "business_type": "unknown",
            "products_mentioned": [],
            "pricing_info": []
        }
        
        content_lower = content.lower()
        
        # Extract business keywords
        business_keywords = [
            "product", "service", "pricing", "solution", "company", "business",
            "customer", "client", "support", "contact", "about", "team"
        ]
        
        found_keywords = [kw for kw in business_keywords if kw in content_lower]
        analysis["keywords"] = found_keywords
        
        # Extract pricing information
        price_patterns = [
            r'\$[\d,]+\.?\d*',
            r'[\d,]+\s*dollars?',
            r'free\s+trial',
            r'starting\s+at\s+\$[\d,]+',
            r'per\s+month',
            r'per\s+year'
        ]
        
        pricing_info = []
        for pattern in price_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            pricing_info.extend(matches)
        
        analysis["pricing_info"] = pricing_info[:10]  # Limit results
        
        # Determine business type
        if any(word in content_lower for word in ["ecommerce", "shop", "store", "buy", "cart"]):
            analysis["business_type"] = "ecommerce"
        elif any(word in content_lower for word in ["saas", "software", "app", "platform"]):
            analysis["business_type"] = "saas"
        elif any(word in content_lower for word in ["consulting", "agency", "services"]):
            analysis["business_type"] = "services"
        elif any(word in content_lower for word in ["blog", "news", "media", "content"]):
            analysis["business_type"] = "content"
        
        return analysis
    
    async def _extract_business_indicators(self, content: ScrapedContent) -> Dict[str, Any]:
        """Extract key business indicators from content."""
        indicators = {
            "has_contact_info": False,
            "has_pricing": False,
            "has_testimonials": False,
            "has_social_media": False,
            "has_blog": False,
            "technology_stack": [],
            "social_links": []
        }
        
        content_text = content.content.lower()
        
        # Check for contact information
        if any(term in content_text for term in ["contact", "email", "phone", "address"]):
            indicators["has_contact_info"] = True
        
        # Check for pricing
        if any(term in content_text for term in ["price", "pricing", "cost", "$", "free", "paid"]):
            indicators["has_pricing"] = True
        
        # Check for testimonials
        if any(term in content_text for term in ["testimonial", "review", "customer", "client"]):
            indicators["has_testimonials"] = True
        
        # Extract social media links
        social_platforms = ["facebook", "twitter", "linkedin", "instagram", "youtube", "tiktok"]
        social_links = []
        
        for link in content.links:
            link_lower = link.lower()
            for platform in social_platforms:
                if platform in link_lower:
                    social_links.append(link)
                    indicators["has_social_media"] = True
        
        indicators["social_links"] = social_links
        
        # Check for blog
        if any(term in content_text for term in ["blog", "article", "post", "news"]):
            indicators["has_blog"] = True
        
        return indicators
    
    async def monitor_trends(self, keywords: List[str], sources: List[str] = None) -> Dict[str, Any]:
        """Monitor trends across multiple sources."""
        if not sources:
            sources = [
                "https://trends.google.com",
                "https://twitter.com",
                "https://reddit.com"
            ]
        
        trend_data = {
            "keywords": keywords,
            "timestamp": datetime.now().isoformat(),
            "sources": sources,
            "trends": {}
        }
        
        for keyword in keywords:
            trend_data["trends"][keyword] = {
                "mentions": 0,
                "sentiment": "neutral",
                "sources": [],
                "growth": 0.0
            }
        
        # This would be expanded with actual trend monitoring logic
        # For now, return structure
        return trend_data
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scraping metrics."""
        return {
            "scraping_stats": self.scraping_stats,
            "content_stored": len(self.scraped_content),
            "failed_urls": len(self.failed_urls),
            "cache_size": len(self.content_cache),
            "average_content_quality": sum(c.quality_score for c in self.scraped_content) / max(1, len(self.scraped_content)),
            "content_types": self._get_content_type_distribution(),
            "initialized": self.is_initialized
        }
    
    def _get_content_type_distribution(self) -> Dict[str, int]:
        """Get distribution of content types."""
        distribution = {}
        for content in self.scraped_content:
            content_type = content.content_type
            distribution[content_type] = distribution.get(content_type, 0) + 1
        return distribution
    
    async def deploy(self, target: str):
        """Deploy web scraping engine to target environment."""
        self.logger.info(f"Deploying Web Scraping Engine to {target}")
        
        if not self.is_initialized:
            await self.initialize()
        
        # Adjust configuration based on target
        if target == "production":
            # More conservative settings for production
            self.request_delays = {domain: 2.0 for domain in self.request_delays}
        elif target == "development":
            # Faster scraping for development
            self.request_delays = {domain: 0.5 for domain in self.request_delays}
        
        self.logger.info(f"Web Scraping Engine deployed to {target}")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        
        if self.selenium_driver:
            self.selenium_driver.quit()
        
        self.logger.info("Web Scraping Engine cleanup complete")

# Convenience functions
async def quick_scrape(url: str) -> ScrapedContent:
    """Quick scraping for simple use cases."""
    engine = AdvancedWebScrapingEngine()
    try:
        result = await engine.scrape_url(url)
        return result
    finally:
        await engine.cleanup()

async def business_intelligence_scan(domain: str) -> Dict[str, Any]:
    """Quick business intelligence scan."""
    engine = AdvancedWebScrapingEngine()
    try:
        result = await engine.gather_business_intelligence(domain)
        return result
    finally:
        await engine.cleanup()