from __future__ import annotations

import hashlib
import sqlite3

from auth import db
from auth.service import (
    TOKEN_LIFETIME_SECONDS,
    hash_password,
    login,
    signup,
    validate_token,
    verify_password,
)


def test_password_hash_verifies_correct_and_rejects_wrong() -> None:
    password_hash, salt = hash_password("supporter123")

    assert verify_password("supporter123", password_hash, salt)
    assert not verify_password("incorrect", password_hash, salt)
    assert password_hash != "supporter123"


def test_signup_validation_and_login(tmp_path) -> None:
    database = tmp_path / "auth.sqlite3"
    db.initialize(database)

    assert not signup("ab", "long-enough", database).success
    assert not signup("supporter", "short", database).success
    assert signup("supporter", "safe-pass", database).success
    assert not signup("supporter", "other-pass", database).success
    assert login("supporter", "safe-pass", path=database).success
    assert not login("supporter", "bad-pass", path=database).success


def test_expired_tokens_are_rejected_and_deleted(tmp_path) -> None:
    database = tmp_path / "auth.sqlite3"
    db.initialize(database)
    signup("supporter", "safe-pass", database)
    result = login("supporter", "safe-pass", remember=True, path=database)
    assert result.token is not None

    expired_at = 1_000.0
    token_hash = hashlib.sha256(result.token.encode("utf-8")).hexdigest()
    with db.connect(database) as connection:
        connection.execute(
            "UPDATE tokens SET expires_at = ? WHERE token_hash = ?",
            (expired_at, token_hash),
        )

    assert validate_token(result.token, database, now=expired_at) is None
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM tokens WHERE token_hash = ?", (token_hash,)
        ).fetchone()[0] == 0


def test_remember_token_is_valid_until_expiry(tmp_path) -> None:
    database = tmp_path / "auth.sqlite3"
    db.initialize(database)
    signup("supporter", "safe-pass", database)
    result = login("supporter", "safe-pass", remember=True, path=database)

    assert result.token is not None
    assert validate_token(result.token, database) == "supporter"
    assert TOKEN_LIFETIME_SECONDS == 30 * 24 * 60 * 60
