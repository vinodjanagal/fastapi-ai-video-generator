from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import MetaData
from app.config import settings
import logging

logger = logging.getLogger(__name__)

metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

Base = declarative_base(metadata=metadata)

engine = create_async_engine(
    settings.database_url.get_secret_value(),
    echo=False,
    future=True,
    pool_recycle=1800,  # Recycle connections every 30 mins
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting app lifespan - connecting to DB")
    yield
    logger.info("Ending app lifespan - disposing DB engine")
    await engine.dispose()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session