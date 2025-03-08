# src/s3_utils.py
import boto3
from botocore.exceptions import ClientError
import io
import os
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
LOCAL_SERVER_URL = os.getenv('LOCAL_SERVER_URL', 'http://your-ngrok-url.ngrok.io')

def get_s3_client():
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
    return s3_path.replace("s3://", "").split("/")[0]

def extract_key(s3_path):
    parts = s3_path.replace("s3://", "").split("/", 1)
    return parts[1] if len(parts) > 1 else ""

def stream_from_s3(s3_path):
    s3_client = get_s3_client()
    bucket = extract_bucket_name(s3_path)
    key = extract_key(s3_path)
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return io.BytesIO(response['Body'].read())
    except ClientError as e:
        raise Exception(f"Failed to stream {s3_path}: {e}")

def stream_from_local_server(local_path):
    url = f"{LOCAL_SERVER_URL}/{local_path}"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return io.BytesIO(response.content)
    except requests.RequestException as e:
        raise Exception(f"Failed to stream {url}: {e}")

def create_s3_folder(s3_path):
    s3_client = get_s3_client()
    bucket = extract_bucket_name(s3_path)
    key = extract_key(s3_path)
    if not key.endswith('/'):
        key += '/'
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
        if 'Contents' not in response:
            s3_client.put_object(Bucket=bucket, Key=key)
            print(f"Created S3 folder: {s3_path}")
    except ClientError as e:
        raise Exception(f"Failed to create folder {s3_path}: {e}")

def upload_to_s3(data, s3_path, is_bytes=False):
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

def list_local_server_files(local_dir):
    url = f"{LOCAL_SERVER_URL}/list/{local_dir}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        files = response.json()
        return [f"{local_dir}{f}" for f in files if not f.endswith('/')]
    except requests.RequestException as e:
        raise Exception(f"Failed to list files in {url}: {e}")

if __name__ == "__main__":
    import yaml
    with open("../config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    bucket = os.getenv('S3_BUCKET')
    for subdir in config['s3']['subdirs'].values():
        create_s3_folder(f"s3://{bucket}/{config['s3']['outputs_path']}{subdir}")
    
    sample_path = "dataset/frames/train/real/sample.jpg"
    stream = stream_from_local_server(sample_path)
    print(f"Streamed {sample_path}, size: {stream.getbuffer().nbytes} bytes")