from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from . import models

# --- Author Schemas ---
class AuthorBase(BaseModel):
    name: str = Field(..., max_length=255)

class AuthorCreate(AuthorBase):
    pass

class AuthorRead(AuthorBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class AuthorWithQuoteCount(AuthorRead):
    quote_count: int

# --- Tag Schemas ---
class TagBase(BaseModel):
    name: str = Field(..., max_length=200)

class TagCreate(TagBase):
    pass

class TagRead(TagBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

# --- Quote Schemas ---
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

# --- Audio Schemas ---
class AudioCreateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class AudioCreateResponse(BaseModel):
    audio_url: str
    text: str

# --- Video Schemas ---
class VideoGenerateRequest(BaseModel):
    quote_id: int = Field(..., gt=0)
    style: Optional[str] = Field(default="dark_gradient", description="Format: voice:style")

class SemanticVideoRequest(BaseModel):
    quote_id: int = Field(..., gt=0)

class VideoAcceptedResponse(BaseModel):
    video_id: int
    status_url: str
    message: str

class VideoStatusResponse(BaseModel):
    id: int
    status: models.VideoStatus
    video_url: Optional[str] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)
