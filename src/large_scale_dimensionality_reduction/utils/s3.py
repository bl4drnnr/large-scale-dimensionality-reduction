import boto3
from botocore.exceptions import ClientError
from large_scale_dimensionality_reduction.utils import setup_logger, cfg
import io
import pandas as pd
from datetime import datetime
from typing import Optional

logger = setup_logger("s3-logger")


class S3Client:
    def __init__(self):
        """Initialize S3 client with credentials from environment variables."""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=cfg.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=cfg.AWS_SECRET_ACCESS_KEY,
            region_name=cfg.AWS_REGION
        )
        self.bucket_name = cfg.S3_BUCKET_NAME
        
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable is not set")
        
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ValueError(f"Bucket {self.bucket_name} does not exist")
            elif error_code == '403':
                raise ValueError(f"Access denied to bucket {self.bucket_name}")
            else:
                raise

    def upload_dataframe(self, df: pd.DataFrame, filename: str, prefix: Optional[str] = None) -> str:
        """
        Upload a pandas DataFrame to S3 as a CSV file.
        
        Args:
            df: The pandas DataFrame to upload
            filename: The name of the file (without path)
            prefix: Optional prefix (folder) in the S3 bucket
            
        Returns:
            str: The S3 key (path) where the file was uploaded
        """
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{prefix}/{timestamp}_{filename}" if prefix else f"{timestamp}_{filename}"
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=csv_buffer.getvalue()
            )
            logger.info(f"Successfully uploaded {filename} to S3 bucket {self.bucket_name}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Error uploading {filename} to S3: {str(e)}")
            raise

    def download_dataframe(self, s3_key: str) -> pd.DataFrame:
        """
        Download a CSV file from S3 and return it as a pandas DataFrame.
        
        Args:
            s3_key: The S3 key (path) of the file to download
            
        Returns:
            pd.DataFrame: The downloaded data as a DataFrame
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            df = pd.read_csv(io.BytesIO(response['Body'].read()))
            logger.info(f"Successfully downloaded {s3_key} from S3 bucket {self.bucket_name}")
            return df
            
        except ClientError as e:
            logger.error(f"Error downloading {s3_key} from S3: {str(e)}")
            raise 