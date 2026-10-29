from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from auth import db

PBKDF2_ITERATIONS = 200_000
TOKEN_LIFETIME_SECONDS = 30 * 24 * 60 * 60


@dataclass(frozen=True)
class AuthResult:
    success: bool
    username: str | None = None
    message: str = ""
    token: str | None = None


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt_bytes = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERATIONS
    )
    return digest.hex(), salt_bytes.hex()


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    candidate, _ = hash_password(password, bytes.fromhex(salt))
    return hmac.compare_digest(candidate, password_hash)


def _create_user(
    username: str, password_hash: str, salt: str, path: str | Path | None
) -> bool:
    try:
        with db.connect(path) as connection:
            connection.execute(
                "INSERT INTO users(username, password_hash, salt, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, password_hash, salt, time.time()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def signup(
    username: str, password: str, path: str | Path | None = None
) -> AuthResult:
    normalized = username.strip()
    if len(normalized) < 3:
        return AuthResult(False, message="Username must be at least 3 characters.")
    if len(password) < 6:
        return AuthResult(False, message="Password must be at least 6 characters.")
    password_hash, salt = hash_password(password)
    if not _create_user(normalized, password_hash, salt, path):
        return AuthResult(False, message="That username is already in use.")
    return AuthResult(True, username=normalized, message="Account created.")


def _issue_token(user_id: int, path: str | Path | None) -> str:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with db.connect(path) as connection:
        connection.execute(
            "INSERT INTO tokens(token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, time.time() + TOKEN_LIFETIME_SECONDS),
        )
    return token


def login(
    username: str,
    password: str,
    remember: bool = False,
    path: str | Path | None = None,
) -> AuthResult:
    with db.connect(path) as connection:
        user = connection.execute(
            "SELECT id, username, password_hash, salt FROM users "
            "WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()
    if user is None or not verify_password(
        password, user["password_hash"], user["salt"]
    ):
        return AuthResult(False, message="Username or password is incorrect.")
    token = _issue_token(user["id"], path) if remember else None
    return AuthResult(True, username=user["username"], message="Welcome back.", token=token)


def validate_token(
    token: str, path: str | Path | None = None, now: float | None = None
) -> str | None:
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    current_time = time.time() if now is None else now
    with db.connect(path) as connection:
        row = connection.execute(
            """
            SELECT users.username, tokens.expires_at
            FROM tokens
            JOIN users ON users.id = tokens.user_id
            WHERE tokens.token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] <= current_time:
            connection.execute("DELETE FROM tokens WHERE token_hash = ?", (token_hash,))
            return None
        return str(row["username"])


def revoke_token(token: str, path: str | Path | None = None) -> None:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with db.connect(path) as connection:
        connection.execute("DELETE FROM tokens WHERE token_hash = ?", (token_hash,))


def ensure_demo_user(path: str | Path | None = None) -> str:
    username = "demo_fan"
    with db.connect(path) as connection:
        existing = connection.execute(
            "SELECT username FROM users WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()
    if existing is None:
        password_hash, salt = hash_password(secrets.token_urlsafe(24))
        _create_user(username, password_hash, salt, path)
    return username
