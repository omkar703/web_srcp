import os
import time
import threading
import logging
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache
from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import re

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "660e0e2836b845b2a62b5fa69d847ddb")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 3600))  # 1 hour

# Cache
cache = TTLCache(maxsize=100, ttl=UPDATE_INTERVAL)

# Keywords
AI_KEYWORDS = {
    'artificial intelligence', 'ai', 'machine learning', 'ml', 
    'deep learning', 'neural network', 'llm', 'nlp',
    'computer vision', 'generative ai', 'chatgpt', 'gpt',
    'openai', 'huggingface', 'transformer', 'pytorch',
    'tensorflow', 'langchain', 'llama', 'stable diffusion'
}

@dataclass
class AITrendConfig:
    news_count: int = 20
    github_count: int = 20
    hf_count: int = 20
    min_stars: int = 100
    min_downloads: int = 1000

class ConfigUpdateRequest(BaseModel):
    news: Optional[int] = None
    github: Optional[int] = None
    hf: Optional[int] = None
    min_stars: Optional[int] = None
    min_downloads: Optional[int] = None

class AITrendTracker:
    def __init__(self):
        self.last_updated = None
        self.cached_data = {'news': [], 'github_repos': [], 'hf_models': []}
        self.config = AITrendConfig()
        self.lock = threading.Lock()
        self.is_refreshing = False

    def refresh_data(self, force=False):
        with self.lock:
            if self.is_refreshing and not force:
                logger.info("Refresh already in progress")
                return False

            self.is_refreshing = True
            try:
                logger.info("Starting data refresh...")
                tasks = {
                    'news': self.get_ai_news,
                    'github_repos': self.get_github_trending,
                    'hf_models': self.get_huggingface_trends
                }

                results = {}
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_key = {executor.submit(task): key for key, task in tasks.items()}
                    for future in as_completed(future_to_key):
                        key = future_to_key[future]
                        try:
                            results[key] = future.result()
                        except Exception as e:
                            logger.error(f"Error fetching {key}: {str(e)}")
                            results[key] = []

                self.cached_data.update(results)
                self.last_updated = datetime.utcnow()
                logger.info("Refresh completed")
                return True
            except Exception as e:
                logger.error(f"Refresh failed: {str(e)}")
                return False
            finally:
                self.is_refreshing = False

    def needs_refresh(self):
        if not self.last_updated:
            return True
        return (datetime.utcnow() - self.last_updated).total_seconds() > UPDATE_INTERVAL

    def get_ai_news(self) -> List[Dict]:
        if not NEWS_API_KEY:
            logger.warning("NewsAPI key not configured")
            return []

        try:
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q=AI OR 'artificial intelligence' OR 'machine learning' OR 'deep learning'&"
                f"sortBy=publishedAt&pageSize=100&language=en&apiKey={NEWS_API_KEY}"
            )
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            articles = response.json().get('articles', [])

            processed = []
            for article in articles:
                if len(processed) >= self.config.news_count:
                    break
                if self.is_ai_related(article):
                    processed.append({
                        'title': article.get('title'),
                        'source': article.get('source', {}).get('name'),
                        'published_at': article.get('publishedAt'),
                        'url': article.get('url'),
                        'summary': self.summarize_article(article.get('content', '')),
                        'keywords': self.extract_keywords(article)
                    })
            return processed
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []

    def is_ai_related(self, article: Dict) -> bool:
        text = ' '.join([
            article.get('title', '').lower(),
            article.get('description', '').lower(),
            article.get('content', '').lower()
        ])
        return any(re.search(rf'\b{keyword}\b', text) for keyword in AI_KEYWORDS)

    def extract_keywords(self, article: Dict) -> List[str]:
        text = ' '.join([
            article.get('title', ''),
            article.get('description', ''),
            article.get('content', '')
        ]).lower()
        return [keyword for keyword in AI_KEYWORDS if keyword in text]

    def summarize_article(self, content: str) -> str:
        if not content:
            return "No summary available"
        return content[:200] + "..."

    def get_github_trending(self) -> List[Dict]:
        if self.config.github_count == 0:
            return []

        base_url = "https://github.com/trending"
        headers = {"User-Agent": "AI-Trend-Tracker/1.0"}
        repos = []

        try:
            for page in range(1, 4):
                if len(repos) >= self.config.github_count:
                    break

                response = requests.get(
                    f"{base_url}?since=weekly&spoken_language_code=en&page={page}", 
                    headers=headers
                )
                soup = BeautifulSoup(response.text, 'html.parser')
                repos += self._parse_github_page(soup)
                time.sleep(1)

            repos = [
                r for r in repos 
                if r['stars'] >= self.config.min_stars and r['is_ai_related']
            ][:self.config.github_count]

            return repos
        except Exception as e:
            logger.error(f"GitHub scrape error: {str(e)}")
            return []

    def _parse_github_page(self, soup) -> List[Dict]:
        result = []

        for article in soup.find_all('article'):
            try:
                title = article.find('h2').text.strip()
                owner, name = [s.strip() for s in title.split('/')]
                desc = article.find('p').text.strip() if article.find('p') else ""

                stars_elem = article.find('a', href=lambda x: x and 'stargazers' in x)
                stars = int(stars_elem.text.strip().replace(',', '')) if stars_elem else 0

                details = self.get_github_repo_details(owner, name)
                is_ai = self.is_ai_related_repo(desc, details.get('topics', []))

                result.append({
                    'name': name,
                    'owner': owner,
                    'url': f"https://github.com/{owner}/{name}",
                    'description': desc,
                    'stars': details.get('stargazers_count', stars),
                    'forks': details.get('forks_count', 0),
                    'topics': details.get('topics', []),
                    'language': details.get('language', ''),
                    'last_updated': details.get('updated_at', None),
                    'is_ai_related': is_ai
                })
            except Exception as e:
                logger.warning(f"Repo parse error: {str(e)}")
        return result

    def is_ai_related_repo(self, description: str, topics: List[str]) -> bool:
        text = description.lower()
        return any(keyword in text for keyword in AI_KEYWORDS) or any(t.lower() in AI_KEYWORDS for t in topics)

    def get_github_repo_details(self, owner: str, repo: str) -> Dict:
        cache_key = f"github_{owner}_{repo}"
        if cache_key in cache:
            return cache[cache_key]

        url = f"https://api.github.com/repos/{owner}/{repo}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

        try:
            response = requests.get(url, headers=headers)
            data = response.json()
            cache[cache_key] = data
            return data
        except Exception:
            return {}

    def get_huggingface_trends(self) -> List[Dict]:
        if self.config.hf_count == 0:
            return []

        url = "https://huggingface.co/api/models?sort=downloads&direction=-1"
        try:
            response = requests.get(url)
            models = response.json()
            ai_models = []

            for model in models:
                if len(ai_models) >= self.config.hf_count:
                    break

                if model.get('downloads', 0) >= self.config.min_downloads:
                    ai_models.append({
                        'model_id': model['modelId'],
                        'downloads': model.get('downloads', 0),
                        'url': f"https://huggingface.co/{model['modelId']}",
                        'pipeline_tag': model.get('pipeline_tag', ''),
                        'last_modified': model.get('lastModified'),
                        'library_name': model.get('library_name', ''),
                        'likes': model.get('likes', 0),
                        'tags': model.get('tags', [])
                    })
            return ai_models
        except Exception as e:
            logger.error(f"HF fetch error: {str(e)}")
            return []

# FastAPI setup
app = FastAPI()
tracker = AITrendTracker()
threading.Thread(target=tracker.refresh_data, kwargs={'force': True}).start()

@app.get("/api/ai-trends")
async def get_trends(
    news: int = Query(20, gt=0, le=100),
    github: int = Query(20, gt=0, le=100),
    hf: int = Query(20, gt=0, le=100)
):
    try:
        tracker.config.news_count = news
        tracker.config.github_count = github
        tracker.config.hf_count = hf

        if tracker.needs_refresh():
            threading.Thread(target=tracker.refresh_data).start()

        return {
            'last_updated': tracker.last_updated.isoformat() if tracker.last_updated else None,
            'news': tracker.cached_data['news'][:news],
            'github_repos': tracker.cached_data['github_repos'][:github],
            'hf_models': tracker.cached_data['hf_models'][:hf],
            'is_refreshing': tracker.is_refreshing
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refresh")
async def force_refresh(config: ConfigUpdateRequest = Body(None)):
    try:
        if config:
            for field, value in config.dict(exclude_unset=True).items():
                if hasattr(tracker.config, field):
                    setattr(tracker.config, field, value)

        threading.Thread(target=tracker.refresh_data, kwargs={'force': True}).start()
        return {
            "status": "refresh_started",
            "last_updated": tracker.last_updated.isoformat() if tracker.last_updated else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    return tracker.config.__dict__

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "last_updated": tracker.last_updated.isoformat() if tracker.last_updated else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
