# In app/schemas.py

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional

# --- CORRECTED IMPORT ---
# We need to import the entire 'models' module to be able to reference 'models.VideoStatus'.
# The old import 'from .models import VideoStatus' is also fine, but then you would just use 'VideoStatus' directly.
# Importing 'models' is a common and clear pattern.
from . import models


# --- Author Schemas (No changes needed) ---
class AuthorBase(BaseModel):
    name: str = Field(..., max_length=255)

class AuthorCreate(AuthorBase):
    pass

class AuthorRead(AuthorBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class AuthorWithQuoteCount(AuthorRead):
    quote_count: int

# --- Tag Schemas (No changes needed) ---
class TagBase(BaseModel):
    name: str = Field(..., max_length=200)

class TagCreate(TagBase):
    pass

class TagRead(TagBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

# --- Quote Schemas (No changes needed) ---
class QuoteBase(BaseModel):
    text: str = Field(..., max_length=2000)
    topic: Optional[str] = Field(None, max_length=50)
    scraped_url: Optional[str] = Field(None, max_length=500)

class QuoteCreate(QuoteBase):
    author_name: str
    tag_names: List[str] = []

class QuoteUpdate(BaseModel):
    text: Optional[str] = None
    topic: Optional[str] = None
    scraped_url: Optional[str] = None
    author_name: Optional[str] = None
    tag_names: Optional[List[str]] = None

class QuoteRead(QuoteBase):
    id: int
    author: AuthorRead
    tags: List[TagRead] = []
    model_config = ConfigDict(from_attributes=True)

# --- Audio Schemas (No changes needed) ---
class AudioCreateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Text to convert to speech.")

class AudioCreateResponse(BaseModel):
    audio_url: str
    text: str

# --- Video Schemas ---

# --- CONSOLIDATED AND CORRECTED ---
# This is now the one and only definition for a video generation request.
# It combines all the fields from your three previous versions.
class VideoGenerateRequest(BaseModel):
    quote_id: int = Field(..., gt=0, description="The ID of the quote to generate a video for.")
    style: Optional[str] = Field(default="dark_gradient", description="e.g., 'dark_gradient', 'yellow_punch', 'blue_calm'")
    background_type: Optional[str] = Field(default="gradient", description="'gradient', 'image_url', or 'video_id'")
    background_value: Optional[str] = Field(default=None, description="Color hex codes, image URL, or video filename")

# --- KEPT FOR REFERENCE ---
# This schema was used for your old synchronous endpoint. We won't use it for the new
# '/generate-video/' endpoint, but it's okay to keep it for now.
class VideoGenerateResponse(BaseModel):
    video_url: str
    quote: QuoteRead

# --- NEW SCHEMAS FOR ASYNC FLOW ---

# This is the response the user gets immediately after submitting a job.
class VideoAcceptedResponse(BaseModel):    
    video_id: int
    status_url: str
    message: str

# This is the response for the status check endpoint.
class VideoStatusResponse(BaseModel):
    id: int
    status: models.VideoStatus # Now correctly references the imported 'models' module
    video_url: Optional[str] = None
    error_message: Optional[str] = None
    
    # --- ADDED THIS LINE ---
    # This is CRITICAL. It tells Pydantic to read data from ORM model attributes
    # (e.g., db_video.status) not just from dict keys. Without this, you'd get an error
    # when returning a database object from your status endpoint.
    model_config = ConfigDict(from_attributes=True)