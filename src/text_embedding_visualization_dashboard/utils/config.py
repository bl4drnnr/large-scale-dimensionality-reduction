from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Configs(BaseModel):
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8800"))


cfg = Configs()
