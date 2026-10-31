from __future__ import annotations

from pathlib import Path
from typing import Any

from auth import db


def initialize(path: str | Path | None = None) -> None:
    with db.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_sales (
                user_id INTEGER NOT NULL REFERENCES users(id),
                date TEXT NOT NULL,
                team TEXT NOT NULL,
                product TEXT NOT NULL,
                units REAL NOT NULL CHECK (units >= 0),
                price REAL NOT NULL DEFAULT 0 CHECK (price >= 0),
                PRIMARY KEY (user_id, date, team, product)
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(user_sales)")
        }
        if "price" not in columns:
            connection.execute(
                "ALTER TABLE user_sales ADD COLUMN price REAL NOT NULL DEFAULT 0"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS user_sales_user_id_idx ON user_sales(user_id)"
        )


def user_id(username: str, path: str | Path | None = None) -> int | None:
    with db.connect(path) as connection:
        row = connection.execute(
            "SELECT id FROM users WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()
    return int(row["id"]) if row is not None else None


def records(user_id_value: int, path: str | Path | None = None) -> list[dict[str, Any]]:
    with db.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT date, team, product, units, price
            FROM user_sales
            WHERE user_id = ?
            ORDER BY date, team, product
            """,
            (user_id_value,),
        ).fetchall()
    return [dict(row) for row in rows]


def replace_records(
    user_id_value: int,
    rows: list[dict[str, Any]],
    path: str | Path | None = None,
) -> None:
    with db.connect(path) as connection:
        connection.execute("DELETE FROM user_sales WHERE user_id = ?", (user_id_value,))
        connection.executemany(
            """
            INSERT INTO user_sales(user_id, date, team, product, units, price)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, date, team, product)
            DO UPDATE SET units = excluded.units, price = excluded.price
            """,
            [
                (
                    user_id_value,
                    row["date"],
                    row["team"],
                    row["product"],
                    row["units"],
                    row.get("price", 0),
                )
                for row in rows
            ],
        )


def upsert_record(
    user_id_value: int,
    row: dict[str, Any],
    path: str | Path | None = None,
) -> None:
    with db.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO user_sales(user_id, date, team, product, units, price)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, date, team, product)
            DO UPDATE SET units = excluded.units, price = excluded.price
            """,
            (
                user_id_value,
                row["date"],
                row["team"],
                row["product"],
                row["units"],
                row.get("price", 0),
            ),
        )


def reset(user_id_value: int, path: str | Path | None = None) -> None:
    with db.connect(path) as connection:
        connection.execute("DELETE FROM user_sales WHERE user_id = ?", (user_id_value,))


def record_count(user_id_value: int, path: str | Path | None = None) -> int:
    with db.connect(path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM user_sales WHERE user_id = ?",
            (user_id_value,),
        ).fetchone()
    return int(row["count"])
