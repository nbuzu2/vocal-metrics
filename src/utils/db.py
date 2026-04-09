from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor

from auth.config import load_database_settings

@contextmanager
def get_connection():
    """
    Open a MySQL connection and manage commit, rollback, and close automatically.
    Usage:
        with get_connection() as connection:
            # Use the connection here
    Yields:
        pymysql.connections.Connection: An active MySQL connection.
    Raises:
        Exception: Any exception raised within the context will trigger a rollback and be re-raised.
    """
    settings = load_database_settings()

    connection = pymysql.connect(
        host=settings.host,
        port=settings.port,
        user=settings.username,
        password=settings.password,
        database=settings.database,
        cursorclass=DictCursor,
        autocommit=False,
    )

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
