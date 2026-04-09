# Utils

This folder contains shared infrastructure helpers used across the project.

## Files

- `db.py`: Opens MySQL connections using the settings loaded by `auth.config`.

## Design notes

- `db.py` is intentionally shared instead of being hard-wired to the auth module.
- Other persistence-related modules can reuse this connection helper in the future.
- The connection helper manages commit, rollback, and close through a context manager.

## Example

```python
from utils.db import get_connection

with get_connection() as connection:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
```
