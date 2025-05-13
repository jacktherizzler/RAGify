# This file will contain functions for fetching and parsing content from web URLs.

from newspaper import Article

def fetch_url_content(url: str) -> dict:
    """Fetches and parses content from a given URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            "title": article.title,
            "text": article.text,
            "publish_date": article.publish_date,
            "authors": article.authors,
            "url": url
        }
    except Exception as e:
        print(f"Error fetching or parsing URL '{url}': {e}")
        return {
            "title": "",
            "text": "",
            "publish_date": None,
            "authors": [],
            "url": url
        }