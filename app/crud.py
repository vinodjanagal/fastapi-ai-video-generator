# file: app/crud.py

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List, Optional
import logging
from sqlalchemy import select, func
from app import models, schemas



logger = logging.getLogger(__name__)

# --- UTILITY CRUD FUNCTIONS (GET OR CREATE PATTERN) ---

async def get_or_create_author(db: AsyncSession, name: str) -> models.Author:
    """
    Retrieves an author by name. If the author does not exist, they are staged for creation.
    This function does NOT commit the transaction.
    """
    if len(name) > 255:
        raise ValueError("Author name exceeds 255 characters.")
        
    result = await db.execute(select(models.Author).where(models.Author.name == name))
    author = result.scalar_one_or_none()
    
    if not author:
        logger.info(f"Staging new author for creation: {name}")
        author = models.Author(name=name)
        db.add(author)
        await db.flush()
        await db.refresh(author)
    return author

async def get_or_create_tag(db: AsyncSession, name: str) -> models.Tag:
    """
    Retrieves a tag by name. If the tag does not exist, it is staged for creation.
    This function does NOT commit the transaction.
    """
    if len(name) > 200:
        raise ValueError("Tag name exceeds 200 characters.")
        
    result = await db.execute(select(models.Tag).where(models.Tag.name == name))
    tag = result.scalar_one_or_none()
    
    if not tag:
        logger.info(f"Staging new tag for creation: {name}")
        tag = models.Tag(name=name)
        db.add(tag)
        await db.flush()
        await db.refresh(tag)
    return tag

# --- CORE CRUD ORCHESTRATORS ---
# In app/crud.py

async def create_quote(db: AsyncSession, quote: schemas.QuoteCreate) -> models.Quote:
    """
    Creates a new quote and handles its relationships with author and tags.
    This function uses a two-phase approach within a single transaction to be async-safe.
    """
    
    # --- PHASE 1: Get or Create all related objects ---
    
    # Get or create the author.
    author = await get_or_create_author(db, name=quote.author_name)
    
    # Get or create all necessary tags.
    tags_to_link = []
    for tag_name in set(quote.tag_names):
        tag = await get_or_create_tag(db, name=tag_name)
        tags_to_link.append(tag)

    # Idempotency Check: See if this exact quote already exists.
    # We do this *after* ensuring the author exists.
    existing_quote_q = await db.execute(
        select(models.Quote).where(
            models.Quote.text == quote.text,
            models.Quote.author_id == author.id
        )
    )
    existing_quote = existing_quote_q.scalar_one_or_none()
    if existing_quote:
        logger.info(f"Returning existing quote ID {existing_quote.id} instead of creating a duplicate.")
        # Eagerly load the relationships for the response.
        await db.refresh(existing_quote, ["author", "tags"])
        return existing_quote

    # --- PHASE 2: Create the main object and link relationships ---
    
    # If it's a new quote, create the ORM object.
    db_quote = models.Quote(
        text=quote.text,
        topic=quote.topic,
        scraped_url=quote.scraped_url,
        author_id=author.id  # Assign the foreign key directly.
    )
    
    # Now, associate the persistent tag objects with the new quote.
    # Because the tags are already in the session (either fetched or newly created and flushed),
    # this operation is now safe and will not trigger a lazy load.
    db_quote.tags.extend(tags_to_link)
    
    # Add the final quote object to the session.
    db.add(db_quote)
    
    # We let the endpoint handle the final commit. The session now contains
    # the new quote and all its relationship links, ready to be saved.
    
    logger.info(f"Prepared new quote '{db_quote.text[:30]}...' for commit.")
    
    # We must flush to get the ID and refresh to load the relationships for the return value.
    await db.flush()
    await db.refresh(db_quote, ["author", "tags"])
    
    return db_quote


# --- STANDARD READ FUNCTIONS ---

async def get_quote(db: AsyncSession, quote_id: int) -> Optional[models.Quote]:
    """Fetches a single quote by its ID, eagerly loading its relationships."""
    query = (
        select(models.Quote)
        .where(models.Quote.id == quote_id)
        .options(selectinload(models.Quote.author), selectinload(models.Quote.tags))
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()

async def get_quotes(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[models.Quote]:
    """Fetches a list of quotes, eagerly loading relationships."""
    query = (
        select(models.Quote)
        .offset(skip)
        .limit(limit)
        .options(selectinload(models.Quote.author), selectinload(models.Quote.tags))
    )
    result = await db.execute(query)
    return result.scalars().unique().all()

async def get_author(db: AsyncSession, author_id: int) -> Optional[models.Author]:
    """Fetches a single author by their ID."""
    result = await db.execute(select(models.Author).where(models.Author.id == author_id))
    return result.scalar_one_or_none()

async def get_authors(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[models.Author]:
    """Fetches a list of authors with pagination."""
    result = await db.execute(select(models.Author).offset(skip).limit(limit))
    return result.scalars().all()


async def update_quote(
    db: AsyncSession, 
    quote_id: int, 
    quote_update: schemas.QuoteUpdate # Use your excellent optional schema
) -> Optional[models.Quote]:
    """
    Performs an intelligent partial update on an existing quote.
    Only the fields provided in the `quote_update` payload will be changed.
    """
    # 1. Fetch the existing quote from the DB. If it doesn't exist, we can't update it.
    db_quote = await get_quote(db=db, quote_id=quote_id)
    if not db_quote:
        logger.warning(f"Attempted to update a non-existent quote with ID: {quote_id}")
        return None

    # 2. Convert the Pydantic model to a dictionary, EXCLUDING any fields that the user
    # did not send. This is the key to a partial update.
    update_data = quote_update.model_dump(exclude_unset=True)
    logger.info(f"Received update data for quote {quote_id}: {update_data}")

    # 3. Iterate through the data the user actually sent and update the ORM object.
    for key, value in update_data.items():
        if key == "author_name":
            # Handle the author relationship
            author = await get_or_create_author(db, name=value)
            db_quote.author = author
        elif key == "tag_names":
            # Handle the many-to-many tags relationship
            db_quote.tags.clear() # Clear the old tags
            tags_to_link = [await get_or_create_tag(db, name=tag_name) for tag_name in set(value)]
            db_quote.tags.extend(tags_to_link)
        else:
            # Handle simple attributes like text, topic, etc.
            setattr(db_quote, key, value)
    
    # 4. Mark the object as modified and prepare it for commit.
    db.add(db_quote)
    await db.flush()
    await db.refresh(db_quote, ["author", "tags"]) # Refresh to load all relationships correctly

    logger.info(f"Staged quote ID {quote_id} for a partial update.")
    return db_quote

async def delete_quote(db: AsyncSession, quote_id: int) -> Optional[models.Quote]:
    """
    Finds a quote by ID and stages it for deletion.
    Does NOT commit the transaction.
    """
    quote_to_delete = await get_quote(db, quote_id=quote_id)
    if not quote_to_delete:
        return None
    
    await db.delete(quote_to_delete)
    logger.info(f"Staged quote ID {quote_id} for deletion.")
    return quote_to_delete

async def get_top_authors_by_quote_count(db: AsyncSession, limit: int = 5) -> List[models.Author]:

    """
    Finds the authors with the most quotes.

    Args:
        db (AsyncSession): The database session.
        limit (int): The number of top authors to return.

    Returns:
        A list of tuples, where each tuple contains (Author Object, quote_count).
    """

    logger.info(f"Fetching top {limit} authors by quote count.")

    query = (
        select(models.Author, func.count(models.Quote.id).label("quote count"))
        .join(models.Quote, models.Author.id == models.Quote.author_id)
        .group_by(models.Author.id)
        .order_by(func.count(models.Quote.id).desc())
        .limit(limit)
    )

    result = await db.execute(query)
    return result.all()

# ==============================================================================
# --- NEW: VIDEO CRUD FUNCTIONS ---
# ==============================================================================

async def get_video(db: AsyncSession, video_id: int) -> Optional[models.Video]:
    """
    Fetches a single video by its ID, eagerly loading the related quote and author.
    This is crucial for the background task which needs this information.
    """
    logger.info(f"Fetching video record with ID: {video_id}")
    query = (
        select(models.Video)
        .where(models.Video.id == video_id)
        .options(selectinload(models.Video.quote).selectinload(models.Quote.author)) # Eager load quote -> author
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()

async def create_video_record(db: AsyncSession, quote_id: int) -> models.Video:
    """
    Creates a new Video record with a PENDING status to act as a 'job ticket'.
    This function stages the new record but does NOT commit it.
    """
    logger.info(f"Staging new video record for quote_id: {quote_id}")
    new_video = models.Video(quote_id=quote_id)
    db.add(new_video)
    # The endpoint will handle the commit to finalize the creation.
    return new_video

# ++++++++++++++++++++++ ADD THIS NEW FUNCTION +++++++++++++++++++++++++++
async def update_video_record(
    db: AsyncSession,
    video: models.Video,
    status: models.VideoStatus,
    video_path: Optional[str] = None,
    error_message: Optional[str] = None
) -> models.Video:
    """
    Updates a video record with a new status and optional video_path or error_message.
    This consolidated function is used by the background worker for all state transitions.
    It stages the changes but does NOT commit them.
    """
    logger.info(f"Staging update for video ID {video.id}: status -> {status.value}")
    
    video.status = status
    
    # Logic for COMPLETED state
    if status == models.VideoStatus.COMPLETED:
        video.video_path = video_path
        video.error_message = None # Ensure error is cleared on success

    # Logic for FAILED state
    elif status == models.VideoStatus.FAILED:
        video.error_message = error_message
        video.video_path = None # Ensure path is cleared on failure
        
    # Logic for other states (like PROCESSING)
    else:
        # No other fields need to be changed for PENDING or PROCESSING
        pass
        
    db.add(video)
    return video