from passlib.context import CryptContext


pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
)


def hash_password(password: str) -> str:
    """Returns the hashed version of the given password using bcrypt algorithm."""
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Returns True if the given password matches the hash, False otherwise."""
    return pwd_context.verify(password, password_hash)
