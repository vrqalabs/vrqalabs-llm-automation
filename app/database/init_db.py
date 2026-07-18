from app.database.base import Base
from app.database.session import engine

# Import all models here
from app.database.models import LLMRequest


def init_db():
    """
    Create all database tables.
    """

    Base.metadata.create_all(bind=engine)
