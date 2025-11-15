from pydantic import BaseModel, ConfigDict, Field, model_validator
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


class SemanticVideoRequest(BaseModel):
    # The user can provide a quote_id. It's optional.
    quote_id: Optional[int] = Field(None, description="The ID of an existing quote to generate a video for.")
    
    # Or, the user can provide new text and an author.
    quote_text: Optional[str] = Field(None, description="The text of a new quote to create and generate a video for.")
    author_name: Optional[str] = Field(None, description="The name of the author for the new quote.")

    # Allow the user to guide the overall visual style
    style_hint: Optional[str] = None # e.g., "vintage film", "anime", "documentary"
    
    # Allow specifying a negative prompt to avoid unwanted elements
    negative_prompt: Optional[str] = None # e.g., "hands", "text", "watermark"
    
    # Allow choosing the voice for narration
    voice: str = "nova" # Default to a high-quality voice

    # This is a powerful Pydantic feature. It's a custom validator that checks the whole model at once.
    @model_validator(mode='before')
    def check_exclusive_fields(cls, values):
        # Get the values for our key fields
        id_present = 'quote_id' in values and values['quote_id'] is not None
        text_present = 'quote_text' in values and values['quote_text'] is not None

        # RULE 1: If both are present, it's an error.
        if id_present and text_present:
            raise ValueError("Provide either 'quote_id' or 'quote_text', but not both.")
        
        # RULE 2: If neither is present, it's an error.
        if not id_present and not text_present:
            raise ValueError("You must provide either 'quote_id' or 'quote_text'.")
        
        # RULE 3: If text is present, the author must also be present.
        if text_present and ('author_name' not in values or not values['author_name']):
            raise ValueError("'author_name' is required when providing 'quote_text'.")
        
        # If all rules pass, return the values.
        return values