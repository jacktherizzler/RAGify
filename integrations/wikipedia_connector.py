# This file will contain functions for interacting with Wikipedia.

import wikipedia

def fetch_wikipedia_content(query: str, num_results: int = 1) -> list:
    """Fetches content from Wikipedia based on a query."""
    results = []
    try:
        page_titles = wikipedia.search(query, results=num_results)
        for title in page_titles:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                results.append({"title": page.title, "summary": page.summary, "url": page.url})
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation pages, e.g., by taking the first option or logging
                print(f"Disambiguation error for '{title}': {e.options[:2]}") # Log first 2 options
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        results.append({"title": page.title, "summary": page.summary, "url": page.url})
                    except Exception as inner_e:
                        print(f"Could not fetch content for disambiguated page '{e.options[0]}': {inner_e}")
            except wikipedia.exceptions.PageError:
                print(f"PageError for '{title}': Page does not exist.")
            except Exception as e:
                print(f"An error occurred while fetching page '{title}': {e}")
    except Exception as e:
        print(f"An error occurred during Wikipedia search for '{query}': {e}")
    return results