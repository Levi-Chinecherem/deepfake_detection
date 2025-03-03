# src/s3_utils.py
import boto3
from botocore.exceptions import ClientError
import io
import os
from dotenv import load_dotenv  # Added for .env loading

# Load environment variables from .env in the root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# S3 credentials from .env
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

def get_s3_client():
    """
    Initialize and return an S3 client using credentials from .env.
    
    Returns:
        boto3.client: S3 client object.
    """
    try:
        return boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    except Exception as e:
        raise Exception(f"Failed to initialize S3 client: {e}")

def extract_bucket_name(s3_path):
    """
    Extract the bucket name from an S3 path (e.g., 's3://my-bucket/path' -> 'my-bucket').
    
    Args:
        s3_path (str): Full S3 path.
    
    Returns:
        str: Bucket name.
    """
    return s3_path.replace("s3://", "").split("/")[0]

def extract_key(s3_path):
    """
    Extract the key (path after bucket) from an S3 path (e.g., 's3://my-bucket/path' -> 'path').
    
    Args:
        s3_path (str): Full S3 path.
    
    Returns:
        str: Key (path within bucket).
    """
    parts = s3_path.replace("s3://", "").split("/", 1)
    return parts[1] if len(parts) > 1 else ""

def stream_from_s3(s3_path):
    """
    Stream a file from S3 into memory.
    
    Args:
        s3_path (str): S3 path (e.g., 's3://my-bucket/dataset/frames/train/real/image1.jpg').
    
    Returns:
        io.BytesIO: In-memory file stream.
    """
    s3_client = get_s3_client()
    bucket = extract_bucket_name(s3_path)
    key = extract_key(s3_path)
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return io.BytesIO(response['Body'].read())
    except ClientError as e:
        raise Exception(f"Failed to stream {s3_path}: {e}")

def create_s3_folder(s3_path):
    """
    Create an S3 folder if it doesn’t exist (S3 uses empty objects for folders).
    
    Args:
        s3_path (str): S3 folder path (e.g., 's3://my-bucket/outputs/models/').
    """
    s3_client = get_s3_client()
    bucket = extract_bucket_name(s3_path)
    key = extract_key(s3_path)
    
    # Ensure key ends with '/' for folder
    if not key.endswith('/'):
        key += '/'
    
    try:
        # Check if folder exists by listing objects
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
        if 'Contents' not in response:
            # Folder doesn’t exist, create it
            s3_client.put_object(Bucket=bucket, Key=key)
            print(f"Created S3 folder: {s3_path}")
    except ClientError as e:
        raise Exception(f"Failed to create folder {s3_path}: {e}")

def upload_to_s3(data, s3_path, is_bytes=False):
    """
    Upload data to S3 (file or bytes).
    
    Args:
        data: Data to upload (file path or bytes).
        s3_path (str): Destination S3 path (e.g., 's3://my-bucket/outputs/models/model.pth').
        is_bytes (bool): True if data is bytes, False if it’s a file path.
    """
    s3_client = get_s3_client()
    bucket = extract_bucket_name(s3_path)
    key = extract_key(s3_path)
    
    try:
        if is_bytes:
            s3_client.put_object(Bucket=bucket, Key=key, Body=data)
        else:
            with open(data, 'rb') as f:
                s3_client.upload_fileobj(f, bucket, key)
        print(f"Uploaded to {s3_path}")
    except ClientError as e:
        raise Exception(f"Failed to upload to {s3_path}: {e}")
    except FileNotFoundError:
        raise Exception(f"Local file not found: {data}")

def list_s3_files(s3_path):
    """
    List all files in an S3 directory.
    
    Args:
        s3_path (str): S3 directory path (e.g., 's3://my-bucket/dataset/frames/train/real/').
    
    Returns:
        list: List of S3 file paths.
    """
    s3_client = get_s3_client()
    bucket = extract_bucket_name(s3_path)
    prefix = extract_key(s3_path)
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' not in response:
            return []
        return [f"s3://{bucket}/{obj['Key']}" for obj in response['Contents'] if not obj['Key'].endswith('/')]
    except ClientError as e:
        raise Exception(f"Failed to list files in {s3_path}: {e}")

if __name__ == "__main__":
    # Example usage (for testing)
    import yaml
    with open("../config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Get bucket from .env instead of config
    BUCKET = os.getenv('S3_BUCKET')
    
    # Test creating output folders
    for subdir in config['s3']['subdirs'].values():
        create_s3_folder(f"{BUCKET}/{config['s3']['outputs_path']}{subdir}")
    
    # Test streaming a sample file (replace with a valid path)
    sample_path = f"{BUCKET}/dataset/frames/train/real/sample.jpg"
    stream = stream_from_s3(sample_path)
    print(f"Streamed {sample_path}, size: {stream.getbuffer().nbytes} bytes")