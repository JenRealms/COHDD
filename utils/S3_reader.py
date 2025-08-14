import boto3
from botocore.exceptions import ClientError
import logging
from typing import Optional, Tuple, Type
from pydantic import BaseModel

class S3ReaderArgs(BaseModel):
    s3_url: str

class S3_reader(BaseModel):
    name: str = "Read and get a file from S3 with the given S3 URL"
    description: str = "Read a file from S3 with the given S3 URL and return the content of the file."
    args_schema: Type[BaseModel] = S3ReaderArgs

    def read_file_from_s3(self, s3_url: str) -> Tuple[bool, Optional[str]]:
        """
        Read a file from S3 with the given S3 URL and return the content of the file.
        """
        try:
            file_path = s3_url.replace('s3://', '')
            bucket_name = file_path.split('/')[0]
            key = '/'.join(file_path.split('/')[1:])
            s3_client = boto3.client('s3')
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            content = response['Body'].read().decode('utf-8')
            return True, content
        except ClientError as e:
            logging.error(f"AWS Error reading {file_path}: {str(e)}")
            return False, None
        except Exception as e:
            logging.error(f"Unexpected error reading {file_path}: {str(e)}")
            return False, None

    def _run(self, s3_url: str):
        success, content = self.read_file_from_s3(s3_url)
        if not success:
            raise RuntimeError(f"Failed to read file from S3: {s3_url}")
        return content