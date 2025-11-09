from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from typing import List,Optional
import logging
from contextlib import asynccontextmanager
from app.database import engine, get_db, lifespan
from app import models, schemas, crud
from sqlalchemy.ext.asyncio import AsyncSession
from textblob import TextBlob
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import uuid # To generate unique filenames
#from . import audio_generator # Import your new audio module
from . import audio_generator_gtts as audio_generator
from pathlib import Path
from . import video_generator
from app.tasks import process_video_generation
from fastapi.responses import JSONResponse
from app.tasks import process_video_generation, process_video_generation_speecht5, process_video_generation_animatediff
from app.schemas import SemanticVideoRequest
from app.tasks import process_semantic_video_generation
from arq import create_pool
from arq.connections import RedisSettings




# --- APPLICATION SETUP ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

redis = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    - On startup: Initializes the audio models.
    - On shutdown: Disposes of the database engine.
    """
    global redis

    logger.info("Application starting up...")
    
    # Load the heavy AI/ML models once at the beginning.
    await audio_generator.initialize_tts_models()

    logger.info("Initializing ARQ Redis connection pool")
    redis = await create_pool(RedisSettings())
    logger.info("ARQ Redis pool initialized")
    
    logger.info("Application startup complete.")
    yield # The application runs here
    
    logger.info("Application shutting down...")
    logger.info("Closing ARQ Redis connection pool")
    await redis.close()
    logger.info("ARQ Redis pool closed.")

    await engine.dispose()
    logger.info("Database engine disposed.")

app = FastAPI(
    title="AI Quote & Video Generator API",
    description="An API to manage quotes, authors, and generate quote-based videos.",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- API ENDPOINTS ---

@app.post("/quotes/", response_model=schemas.QuoteRead, status_code=status.HTTP_201_CREATED)
async def create_quote_endpoint(quote: schemas.QuoteCreate, db: AsyncSession = Depends(get_db)):
    """
    Creates a new quote, along with its author and tags if they don't exist.
    This entire operation is a single atomic transaction.
    """
    try:
        logger.info(f"Received request to create quote: {quote.text[:30]}...")
        # The CRUD function prepares the new quote and its relationships
        db_quote = await crud.create_quote(db, quote)
        # This endpoint is responsible for the final commit
        await db.commit()
        return db_quote
    except Exception as e:
        # If any part of the process fails, roll back everything
        await db.rollback()
        logger.error(f"Quote creation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Internal server error during quote creation."
        )

@app.get("/quotes/", response_model=List[schemas.QuoteRead])
async def read_quotes_endpoint(skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)):
    """Retrieves a paginated list of all quotes."""
    logger.info(f"Fetching quotes with skip={skip} and limit={limit}")
    quotes = await crud.get_quotes(db=db, skip=skip, limit=limit)
    return quotes

@app.get("/quotes/{quote_id}", response_model=schemas.QuoteRead)
async def read_quote_endpoint(quote_id: int, db: AsyncSession = Depends(get_db)):
    """Retrieves a single quote by its ID."""
    logger.info(f"Fetching quote with id: {quote_id}")
    db_quote = await crud.get_quote(db=db, quote_id=quote_id)
    if db_quote is None:
        logger.warning(f"Quote with id {quote_id} not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quote not found")
    return db_quote


@app.put("/quotes/{quote_id}", response_model=schemas.QuoteRead, summary="Update an existing quote")
async def update_quote_endpoint(
    quote_id: int, 
    quote: schemas.QuoteUpdate, # Using your excellent partial update schema
    db: AsyncSession = Depends(get_db)
):
    """
    Updates a quote's fields. Only the fields provided in the request body will be updated.
    """
    try:
        updated_quote = await crud.update_quote(db, quote_id=quote_id, quote_update=quote)
        
        if updated_quote is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quote not found")
        
        await db.commit()
        await db.refresh(updated_quote)
        return updated_quote
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to update quote {quote_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    

@app.delete("/quotes/{quote_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a quote")
async def delete_quote_endpoint(quote_id: int, db: AsyncSession = Depends(get_db)):
    """Deletes a quote by its ID."""
    try:
        deleted_quote = await crud.delete_quote(db, quote_id=quote_id)
        if deleted_quote is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quote not found")
        
        await db.commit()
        return None # 204 response has no body
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to delete quote {quote_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    

# Replace the old PUT endpoint with this PATCH endpoint.
@app.patch("/quotes/{quote_id}", response_model=schemas.QuoteRead, summary="Partially update an existing quote")
async def partial_update_quote_endpoint(
    quote_id: int, 
    quote_update: schemas.QuoteUpdate, # Use your optional update schema
    db: AsyncSession = Depends(get_db)
):
    """
    Partially updates a quote. Only the fields provided in the request body will be changed.
    For example, to update only the topic, send:
    {
        "topic": "A Brand New Topic"
    }
    """
    try:
        updated_quote = await crud.update_quote(db, quote_id=quote_id, quote_update=quote_update)
        
        if updated_quote is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quote not found")
        
        await db.commit()
        await db.refresh(updated_quote) # One final refresh after commit
        return updated_quote
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to patch quote {quote_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    

@app.get("/authors/top", response_model=List[schemas.AuthorWithQuoteCount], summary= "Get Top authors by Quote Count")
async def get_top_authors_endpoint(k: int, db: AsyncSession= Depends(get_db)):
    """
    Retrieves the top K authors based on the number of quotes they have.
    """

    try: 
        top_authors_rows = await crud.get_top_authors_by_quote_count(db, limit=k)

        response = []

        for row in top_authors_rows:
            author_obj = row[0]
            quote_count_val = row[1]

            response.append(
                schemas.AuthorWithQuoteCount(
                    id = author_obj.id,
                    name=author_obj.name,
                    quote_count= quote_count_val
                )
            )

        return response
    
    except Exception as e:
        logger.error(f"Failed to fetch top authors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    polarity: float
    subjectivity: float


@app.post("/quotes/analyze-sentiment", response_model=SentimentResponse, summary= "Analyze the sentiment of a text")
async def analyze_sentiment_endpoint(request: SentimentRequest):
    """
    Analyzes the sentiment of a given text string and returns its polarity and subjectivity.
    - **Polarity**: A float between -1.0 (very negative) and 1.0 (very positive).
    - **Subjectivity**: A float between 0.0 (very objective) and 1.0 (very subjective).
    """

    logger.info(f"Analyzing sentiment for text: {request.text[:50]}")
    try:
            blob = TextBlob(request.text)
            sentiment = blob.sentiment
            return SentimentResponse(
                text= request.text,
                polarity= sentiment.polarity,
                subjectivity= sentiment.subjectivity
            )
    
    except Exception as e:
        logger.error(f"sentiment analysis failed: {e}, exc_info=True")
        raise HTTPException(status_code=500, detail= "Internal server error during sentiment analysis")


@app.post("/generate-audio/", response_model=schemas.AudioCreateResponse, summary="Generate Audio from Text")
async def generate_audio_endpoint(request: schemas.AudioCreateRequest):
    """
    Generates a WAV audio file from text using a pre-trained TTS model.
    Returns a URL to the generated audio file.
    """
    try:
        logger.info(f"Received audio generation request for text: '{request.text[:50]}...'")
        
        # Generate a unique filename to prevent file collisions.
        unique_filename = f"quote_{uuid.uuid4()}.wav"
        
        # Call the audio generation function from the dedicated module.
        audio_path_str = await audio_generator.generate_audio_from_text(
            text=request.text,
            output_filename=unique_filename
        )
        
        # Convert the local file system path to a web-accessible URL.
        # Path(...).as_posix() ensures forward slashes are used, which is required for URLs.
        audio_url = "/" + Path(audio_path_str).as_posix()
        
        return schemas.AudioCreateResponse(audio_url=audio_url, text=request.text)

    except ValueError as e: # Catches "Text cannot be empty"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e: # Catches "TTS models unavailable"
        logger.error(f"Audio generation failed: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred during audio generation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.")
    



# This is the replacement for your old synchronous endpoint.

@app.post("/generate-video/", response_model=schemas.VideoAcceptedResponse, status_code=status.HTTP_202_ACCEPTED, summary="Submit a video generation job")
async def generate_video_endpoint(
    request: schemas.VideoGenerateRequest,
    # background_tasks: BackgroundTasks, # << DELETE THIS LINE
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts a video generation request and starts the process in the background using ARQ.
    This endpoint returns immediately with a job ID.
    """
    try:
        logger.info(f"Received request to generate video for quote_id: {request.quote_id}")
        db_quote = await crud.get_quote(db=db, quote_id=request.quote_id)
        if not db_quote:
            raise HTTPException(status_code=status.HTTP_4_NOT_FOUND, detail="Quote not found.")

        # This part stays the same
        new_video = await crud.create_video_record(db, quote_id=db_quote.id)
        await db.commit()
        await db.refresh(new_video)
        logger.info(f"Created pending video record with ID: {new_video.id}")

        # --- REPLACE THE ENTIRE OLD if/else BLOCK WITH THIS ---
        logger.info(f"Enqueuing ARQ job 'generate_video_task' for video_id: {new_video.id}")
        await redis.enqueue_job(
            "generate_video_task",  # The string name of our new worker function
            new_video.id,           # The first argument (video_id)
            request.style           # The second argument (style)
        )
        # ----------------------------------------------------

        return {
            "video_id": new_video.id,
            "status_url": f"/videos/{new_video.id}",
            "message": "Video generation has been accepted and is processing in the background."
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to initiate video generation for quote {request.quote_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create video generation job.")
    
# --- ADDED: NEW ENDPOINT TO CHECK JOB STATUS ---
@app.get("/videos/{video_id}", response_model=schemas.VideoStatusResponse, summary="Get Video Generation Status")
async def get_video_status_endpoint(video_id: int, db: AsyncSession = Depends(get_db)):
    """
    Retrieves the status and result of a video generation task.
    The client will poll this endpoint until the status is 'COMPLETED' or 'FAILED'.
    """
    logger.info(f"Checking status for video_id: {video_id}")
    db_video = await crud.get_video(db, video_id=video_id)
    if not db_video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video record not found.")



    response_data = {
        "id": db_video.id,
        "status": db_video.status,
        "progress": db_video.progress,
        "video_url": None, # Start with null
        "error_message": db_video.error_message
    }

    if db_video.status == models.VideoStatus.COMPLETED and db_video.video_path:
        response_data["video_url"] = "/" + Path(db_video.video_path).as_posix()

    return JSONResponse(content=response_data)


class AnimateDiffGenerateRequest(BaseModel):
    prompt: str
    quote_id: int
    voice: str = "atlas" # Add voice with a default

@app.post("/generate-animatediff-video/", response_model=schemas.VideoAcceptedResponse, status_code=202)
async def generate_animatediff_endpoint(
    request: AnimateDiffGenerateRequest,
    db: AsyncSession = Depends(get_db)
):
    logger.info(f"Received AnimateDiff request for quote_id: {request.quote_id}")

    quote = await crud.get_quote(db, quote_id=request.quote_id)
    if not quote:
        raise HTTPException(status_code=404, detail=f"Quote with id {request.quote_id} not found")

    try:
        new_video = await crud.create_video_record(db, quote_id=request.quote_id)
        await db.commit()
        await db.refresh(new_video)

        # ✅ Instead of BackgroundTasks → Use ARQ for safety
        logger.info(f"Enqueuing AnimateDiff job to ARQ for video_id: {new_video.id}")
        await redis.enqueue_job(
            "process_video_generation_animatediff",
            new_video.id,
            request.prompt,
            request.voice
        )

        return {
            "video_id": new_video.id,
            "status_url": f"/videos/{new_video.id}",
            "message": "AnimateDiff video generation started using ARQ (long process)."
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to enqueue AnimateDiff: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to start AnimateDiff job.")


    
@app.post("/test-audio-job/", status_code=status.HTTP_202_ACCEPTED)
async def test_audio_job_endpoint():
    """
    A simple endpoint to test enqueuing a job for our ARQ worker.
    This will call the `generate_audio_task` function in worker.py.
    """
    logger.info("Received request to test ARQ audio job.")
    
    quote_text_for_test = "Success is not final, failure is not fatal: it is the courage to continue that counts."
    output_filename = f"static/audio/test_job_{uuid.uuid4()}.wav"

    # This is the key command to send a job to the worker.
    await redis.enqueue_job(
        "generate_audio_task",         # The name of the function in worker.py
        quote_text_for_test,           # The first argument (quote_text)
        output_filename                # The second argument (audio_file_path)
    )

    logger.info(f"Successfully enqueued audio job. Output will be at: {output_filename}")
    return {"message": "Job enqueued successfully.", "output_path": output_filename}



@app.post(
    "/video/semantic/",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create and enqueue a semantic AI video generation job",
    response_description="Confirmation that the video generation job has been enqueued."
)
async def generate_semantic_video(
    request: SemanticVideoRequest,
    db: AsyncSession = Depends(get_db)
    
):
    """
    Enqueues a job to generate a semantic video from either an existing quote_id or new text.
    This operation is a single atomic database transaction.
    """
    try:
        target_quote: Optional[models.Quote] = None

        # --- STEP 1: RESOLVE THE QUOTE ---
        if request.quote_id:
            logger.info(f"Received video request for existing quote_id={request.quote_id}")
            target_quote = await crud.get_quote(db, quote_id=request.quote_id)
            if not target_quote:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Quote with id {request.quote_id} not found."
                )
        elif request.quote_text and request.author_name:
            logger.info(f"Received video request for new quote text: '{request.quote_text[:50]}...'")
            quote_to_create = schemas.QuoteCreate(
                text=request.quote_text,
                author_name=request.author_name
            )
            # This function STAGES the quote for creation but does NOT commit.
            target_quote = await crud.create_quote(db, quote=quote_to_create)

        if not target_quote:
            raise HTTPException(status_code=400, detail="Could not find or create a target quote.")

        # --- STEP 2: CREATE THE VIDEO RECORD ---
        # This function STAGES the video record for creation but does NOT commit.
        new_video = await crud.create_video_record(db=db, quote_id=target_quote.id)
        
        # --- STEP 3: COMMIT THE TRANSACTION ---
        # This single commit will atomically save the new quote, the new author (if any),
        # AND the new video record all at once.
        await db.commit()
        
        # --- STEP 4: REFRESH THE OBJECT ---
        # NOW the new_video object is persistent and has an ID. This call will succeed.
        await db.refresh(new_video)

        # --- STEP 5: ENQUEUE THE JOB ---
        logger.info(f"Enqueuing semantic pipeline for video_id={new_video.id} linked to quote_id={target_quote.id}")
        await redis.enqueue_job("generate_semantic_video_task", new_video.id)

        return {
            "message": "Semantic video generation has been enqueued.",
            "video_id": new_video.id,
            "quote_id": target_quote.id
        }
    except Exception as e:
        # If anything fails before the commit, this rollback clears the session.
        await db.rollback()
        logger.error(f"Failed to create semantic video job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")