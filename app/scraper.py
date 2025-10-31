# file: scraper.py

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging
from typing import Set, Dict

# Import components from our project
from app.database import AsyncSessionLocal
from app.config import settings
from app import models

# --- 1. CONFIGURATION AND LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def get_or_create(db_session: AsyncSession, model, **kwargs):
    """
    A generic, reusable function to get an object or create it if it doesn't exist.
    This is a core pattern in database programming.
    """
    result = await db_session.execute(select(model).filter_by(**kwargs))
    instance = result.scalar_one_or_none()
    if instance:
        return instance
    else:
        logger.info(f"Creating new {model.__name__} with attributes: {kwargs}")
        instance = model(**kwargs)
        db_session.add(instance)
        await db_session.flush() # Flush to get the ID
        return instance

async def scrape_and_populate():
    """
    Orchestrates the scraping and database population process asynchronously.
    """
    total_quotes_added = 0
    seen_quotes: Set[str] = set()

    async with aiohttp.ClientSession() as http_session:
        async with AsyncSessionLocal() as db_session:
            url = settings.base_url
            page_num = 1

            while url and page_num <= settings.max_pages:
                try:
                    logger.info(f"Fetching page {page_num}: {url}")
                    async with http_session.get(url, timeout=10) as response:
                        response.raise_for_status()
                        html = await response.text()
                    
                    soup = BeautifulSoup(html, "html.parser")
                    quote_divs = soup.find_all("div", class_="quote")
                    if not quote_divs:
                        break

                    for div in quote_divs:
                        text = div.find("span", class_="text").text.strip()
                        if text in seen_quotes:
                            continue
                        seen_quotes.add(text)
                        
                        author_name = div.find("small", class_="author").text.strip()
                        tag_names = {t.text for t in div.find_all("a", class_="tag")}
                        
                        # Use the generic "get or create" helper for author and tags
                        author = await get_or_create(db_session, models.Author, name=author_name)
                        
                        tags = [await get_or_create(db_session, models.Tag, name=tag_name) for tag_name in tag_names]

                        # Create the Quote object and add it to the session
                        new_quote = models.Quote(
                            text=text,
                            author_id=author.id,
                            topic=settings.topic,
                            scraped_url=url,
                            tags=tags # Directly assign the list of Tag objects
                        )
                        db_session.add(new_quote)
                        total_quotes_added += 1

                    # Commit all changes for this page in a single transaction
                    await db_session.commit()
                    logger.info(f"Committed {len(quote_divs)} processed quotes from page {page_num}.")
                    
                    # Pagination
                    next_btn = soup.find("li", class_="next")
                    url = (settings.base_url + next_btn.find("a")["href"]) if (next_btn and next_btn.find("a")) else None
                    page_num += 1

                except Exception as e:
                    logger.error(f"An error occurred on page {page_num}: {e}", exc_info=True)
                    await db_session.rollback() # Rollback the failed transaction
                    break

    logger.info(f"✅ Scraping complete! Total new quotes added: {total_quotes_added}")

if __name__ == "__main__":
    asyncio.run(scrape_and_populate())