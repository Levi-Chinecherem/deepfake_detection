# web_interface/detection_app/s3_utils.py
import boto3
from botocore.exceptions import ClientError
import io

def get_s3_client():
    try:
        return boto3.client('s3')
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