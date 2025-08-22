"""
S3 upload utilities for plot images.

Saves plots to a temporary PNG file, uploads to S3, and returns a pre-signed URL.
"""

import os
import tempfile
import uuid
from pathlib import Path

from .logging_config import setup_logging

logger = setup_logging("s3_utils")

# Attempt to import boto3/botocore; provide fallbacks if unavailable so linters don't fail
try:
    import boto3  # type: ignore
    from botocore.client import Config as BotoConfig  # type: ignore
    from botocore.exceptions import BotoCoreError, NoCredentialsError  # type: ignore
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore
    BotoConfig = None  # type: ignore

    class BotoCoreError(Exception):
        pass

    class NoCredentialsError(Exception):
        pass


class S3UploadError(Exception):
    """Raised when an S3 upload or URL generation fails."""


def _get_s3_client(region: str | None = None, session_token: str | None = None):
    """Create and return an S3 client using environment variables."""
    if boto3 is None or BotoConfig is None:  # type: ignore
        raise S3UploadError(
            "boto3/botocore not installed. Please install boto3 to enable S3 uploads."
        )

    access_key = os.environ.get("AWS_ACCESS_KEY")
    secret_key = os.environ.get("AWS_SECRET_KEY")
    region = region or os.environ.get("AWS_REGION", "us-west-1")
    session_token = session_token or os.environ.get("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise S3UploadError(
            "Missing AWS credentials. Ensure AWS_ACCESS_KEY and AWS_SECRET_KEY are set."
        )

    session = boto3.session.Session(  # type: ignore
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
        region_name=region,
    )
    return session.client("s3", config=BotoConfig(signature_version="s3v4"))  # type: ignore


def upload_file_and_get_presigned_url(
    file_path: Path,
    *,
    content_type: str = "image/png",
    bucket_env_var: str = "AWS_S3_BUCKET",
    prefix: str = "plots/",
    expires_in_seconds: int = 600,
) -> str:
    """
    Upload a local file to S3 and return a pre-signed URL.

    Args:
        file_path: Path to the local file to upload.
        content_type: MIME type to set on the S3 object.
        bucket_env_var: Environment variable name that holds the S3 bucket.
        prefix: Key prefix inside the bucket.
        expires_in_seconds: Pre-signed URL expiration in seconds.

    Returns:
        Pre-signed URL string.
    """
    bucket = os.environ.get(bucket_env_var)
    if not bucket:
        raise S3UploadError(
            f"Missing S3 bucket name. Please set {bucket_env_var} environment variable."
        )

    # Start with a client (may be wrong region), discover correct region, then rebuild client
    s3 = _get_s3_client()
    try:
        loc = s3.get_bucket_location(Bucket=bucket)["LocationConstraint"]
        bucket_region = loc or "us-east-1"
        s3 = _get_s3_client(region=bucket_region)
    except Exception:
        # If discovery fails, continue with current region; presign may still work when region matches env
        pass

    unique_id = uuid.uuid4().hex
    key = f"{prefix}{unique_id}.png"

    try:
        s3.upload_file(
            str(file_path),
            bucket,
            key,
            ExtraArgs={"ContentType": content_type, "ACL": "private"},
        )
    except (BotoCoreError, NoCredentialsError) as e:
        logger.error("Failed to upload file to S3: %s", e)
        raise S3UploadError(f"Failed to upload file to S3: {e}") from e

    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in_seconds,
        )
        return url
    except (BotoCoreError, NoCredentialsError) as e:
        logger.error("Failed to generate pre-signed URL: %s", e)
        raise S3UploadError(f"Failed to generate pre-signed URL: {e}") from e


def save_bytes_to_temp_png(data: bytes) -> Path:
    """Persist bytes to a temporary PNG file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    return tmp_path
