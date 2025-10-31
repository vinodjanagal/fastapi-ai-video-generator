from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from typing import List
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



# --- APPLICATION SETUP ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    - On startup: Initializes the audio models.
    - On shutdown: Disposes of the database engine.
    """
    logger.info("Application starting up...")
    
    # Load the heavy AI/ML models once at the beginning.
    await audio_generator.initialize_tts_models()
    
    logger.info("Application startup complete.")
    yield # The application runs here
    
    logger.info("Application shutting down...")
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
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts a video generation request and starts the process in the background.
    This endpoint returns immediately with a job ID.
    """
    try:
        logger.info(f"Received request to generate video for quote_id: {request.quote_id}")
        db_quote = await crud.get_quote(db=db, quote_id=request.quote_id)
        if not db_quote:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quote not found.")

        # Stage 1: Create the 'job ticket' in the database.
        new_video = await crud.create_video_record(db, quote_id=db_quote.id)
        await db.commit()
        await db.refresh(new_video)
        logger.info(f"Created pending video record with ID: {new_video.id}")

        
        style_parts = request.style.split(':')
        voice = style_parts[0]
        video_style = style_parts[1] if len(style_parts) > 1 else "dark_gradient" # Default video style

        speecht5_voices = {"atlas", "nova", "echo", "breeze"}
        
        if voice in speecht5_voices:
            logger.info(f"Enqueuing SpeechT5 task with voice: {voice} and video style: {video_style}")
            background_tasks.add_task(
                process_video_generation_speecht5,
                video_id=new_video.id,
                voice_name=voice,
                video_style=video_style
            )
        else:
            # Fallback to the original gTTS engine, using the whole style string
            logger.info(f"Enqueuing default gTTS task with style: {request.style}")
            background_tasks.add_task(
                process_video_generation,
                video_id=new_video.id,
                style=request.style
            )
        # ----------------------------

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
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts a job to generate a short, animated video using AnimateDiff.
    This is a VERY slow, resource-intensive process.
    """
    logger.info(f"Received AnimateDiff request for quote_id: {request.quote_id}")

    quote = await crud.get_quote(db, quote_id=request.quote_id)
    if not quote:
        raise HTTPException(status_code=404, detail=f"Quote with id {request.quote_id} not found")

    try:
        new_video = await crud.create_video_record(db, quote_id=request.quote_id)
        await db.commit()
        await db.refresh(new_video)
        
        logger.info(f"Enqueuing AnimateDiff background task for video_id: {new_video.id}")
        background_tasks.add_task(
            process_video_generation_animatediff,
            video_id=new_video.id,
            prompt=request.prompt,
            voice_name=request.voice
        )

        return {
            "video_id": new_video.id,
            "status_url": f"/videos/{new_video.id}",
            "message": "AnimateDiff video generation accepted. This will be very slow."
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to initiate AnimateDiff job for quote {request.quote_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create AnimateDiff job.")