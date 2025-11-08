from sqlalchemy import Column, Integer, String, Text, ForeignKey, Table, DateTime,Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
# app/models/quote_video.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
import enum
#from app.db.base_class import Base  # adapt to your project



# --- Junction Table Definition ---
# Explicitly associate the junction table with the declarative Base's metadata.
quote_tags_table = Table(
    "quote_tags",
    Base.metadata,
    Column("quote_id", ForeignKey("quotes.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)

# --- ORM Model Definitions ---

class Author(Base):
    __tablename__ = "authors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    bio = Column(Text)

    # Best Practice: Use lazy="selectin" for all relationships in an async environment
    # to prevent await-related errors from lazy loading.
    quotes = relationship("Quote", back_populates="author", cascade="all, delete-orphan", lazy="selectin")

class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, unique=True, index=True)

    quotes = relationship("Quote", secondary=quote_tags_table, back_populates="tags", lazy="selectin")

class Quote(Base):
    __tablename__ = "quotes"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    author_id = Column(Integer, ForeignKey("authors.id"), nullable=False)
    topic = Column(String(50), index=True)
    scraped_url = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    author = relationship("Author", back_populates="quotes", lazy="selectin")
    tags = relationship("Tag", secondary=quote_tags_table, back_populates="quotes", lazy="selectin")
    videos = relationship("Video", back_populates="quote", cascade="all, delete-orphan", lazy="selectin")


class VideoStatus(str, enum.Enum):
    PENDING= "PENDING"
    PROCESSING= "PROCESSING"
    COMPLETED= "COMPLETED"
    FAILED= "FAILED"

class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    quote_id = Column(Integer, ForeignKey("quotes.id", ondelete="CASCADE"))
    video_path = Column(String(500), nullable=True)
    status= Column(Enum(VideoStatus), default= VideoStatus.PENDING, nullable=False)
    error_message= Column(Text, nullable=True)
    duration_seconds = Column(Integer)
    views = Column(Integer, default=0)
    mood_score = Column(Float)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    progress = Column(Float, default=0.0, nullable=False)
    
    # Add lazy="selectin"
    quote = relationship("Quote", back_populates="videos", lazy="selectin")