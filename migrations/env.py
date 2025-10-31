import asyncio
from logging.config import fileConfig
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import pool
from alembic import context
import sys
import pathlib
import logging
from app.database import Base
from app.models import quote_tags_table, Author, Tag, Quote, Video

# Project path
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]  # Adjust to parents[1] for migrations/env.py
sys.path.append(str(BASE_DIR))

 # Use settings.database_url
from app.database import Base
from app.config import settings
from app import models  

logger = logging.getLogger("alembic.env")

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.database_url.get_secret_value())

target_metadata = Base.metadata

async def run_migrations_online():
    try:
        connectable = create_async_engine(
            config.get_main_option("sqlalchemy.url"),
            poolclass=pool.NullPool,
        )
        async with connectable.connect() as connection:
            await connection.run_sync(
                lambda sync_conn: context.configure(
                    connection=sync_conn,
                    target_metadata=target_metadata,
                )
            )
            async with context.begin_transaction():
                await connection.run_sync(lambda sync_conn: context.run_migrations())
        logger.info("✅ Online migrations completed")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        await connectable.dispose()

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
        logger.info("✅ Offline migrations completed")

if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())