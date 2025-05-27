from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Configs(BaseModel):
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8800"))
    
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")

    HPC_HOST: str = os.getenv("HPC_HOST", "")
    HPC_USERNAME: str = os.getenv("HPC_USERNAME", "")
    HPC_SSH_KEY: str = os.getenv("HPC_SSH_KEY", "")
    HPC_WORK_DIR: str = os.getenv("HPC_WORK_DIR", "")
    HPC_WORK_FOLDER: str = os.getenv("HPC_WORK_FOLDER", "")

    S3_BUCKET_URL: str = os.getenv("S3_BUCKET_URL", "")


cfg = Configs()
