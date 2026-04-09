from pydantic import BaseModel

from auth.passwords import verify_password
from utils.db import get_connection


class AuthenticatedUser(BaseModel):
    id: int
    email: str
    full_name: str | None = None


def get_user_by_email(email: str) -> dict | None:
    """Fetches a user from the database by their email address.
    Args:
        email (str): The email address of the user to fetch.
    
    Returns:
        dict | None: A dictionary containing the user's data if found, or None if no user with the given email exists.
    """
    normalized_email = email.strip().lower()

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, email, password_hash, full_name, is_active
                FROM users
                WHERE email = %s
                LIMIT 1
                """,
                (normalized_email,),
            )
            return cursor.fetchone()


def authenticate_user(email: str, password: str) -> AuthenticatedUser | None:
    """Authenticates a user by their email and password.
    Args:
        email (str): The email address of the user to authenticate.
        password (str): The plaintext password provided by the user.
        
    Returns:
        AuthenticatedUser | None: An AuthenticatedUser object if authentication is successful, or None if authentication fails.
    """
    if not email or not password:
        return None

    user = get_user_by_email(email)
    if not user:
        return None

    if not user["is_active"]:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    return AuthenticatedUser(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
    )
