# Auth

This folder contains the authentication domain for the app.

## Files

- `config.py`: Loads database settings from environment variables and, when needed, AWS Secrets Manager.
- `passwords.py`: Hashes and verifies user passwords through `passlib`.
- `auth_service.py`: Retrieves users from the database and validates login credentials.

## Responsibilities

- Authentication is currently email/password based.
- User records are stored in MySQL.
- Passwords must be stored as hashes in `users.password_hash`, never as plain text.

## Expected database fields

The login flow currently expects a `users` table with at least:

- `id`
- `email`
- `password_hash`
- `full_name`
- `is_active`

## Notes

- In local development, `DB_USER` and `DB_PASSWORD` can come from environment variables.
- In EC2, credentials can instead come from AWS Secrets Manager through an instance role.
