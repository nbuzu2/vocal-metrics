import json
import os

import boto3
from pydantic import BaseModel


class DatabaseSettings(BaseModel):
    """
    Represents the database settings.
    """
    host: str
    port: int = 3306
    database: str
    username: str
    password: str


def _parse_aws_secret() -> dict:
    """
    Fetches and parses the database credentials from AWS Secrets Manager.
    
    Returns:
        dict: A dictionary containing the database credentials.
    Raises:
        ValueError: If the DB_SECRET_NAME environment variable is not set.
        botocore.exceptions.ClientError: If the secret cannot be retrieved.
    """
    secret_name = os.getenv("DB_SECRET_NAME")
    if not secret_name:
        raise ValueError("DB_SECRET_NAME is not set.")

    region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    client = boto3.client("secretsmanager", region_name=region_name)

    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


def load_database_settings() -> DatabaseSettings:
    """
    Loads the database settings from environment variables and AWS Secrets Manager.
    host, port, and database name are loaded from environment variables, 
    while username and password are fetched from AWS Secrets Manager.
    
    Returns:
        DatabaseSettings: An instance of DatabaseSettings with the loaded configuration.
    """
    secret = _parse_aws_secret()

    return DatabaseSettings(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME"),
        username=secret["username"],
        password=secret["password"],
    )
