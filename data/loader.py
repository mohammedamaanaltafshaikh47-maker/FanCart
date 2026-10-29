from __future__ import annotations

import io
import json
import urllib.request
import zipfile
import pandas as pd

from data.synthetic import generate_matches

CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_json.zip"


def _match_from_info(info: dict[str, object]) -> dict[str, object] | None:
    dates = info.get("dates")
    teams = info.get("teams")
    outcome = info.get("outcome")
    if not isinstance(dates, list) or not dates or not isinstance(teams, list) or len(teams) < 2:
        return None
    if not isinstance(outcome, dict) or not outcome.get("winner"):
        return None
    margin_data = outcome.get("by", {})
    margin = None
    if isinstance(margin_data, dict) and margin_data:
        margin_type, margin_value = next(iter(margin_data.items()))
        margin = f"{margin_value} {margin_type}"
    return {
        "date": pd.to_datetime(dates[0]),
        "team1": str(teams[0]),
        "team2": str(teams[1]),
        "venue": str(info.get("venue", "Unknown venue")),
        "winner": str(outcome["winner"]),
        "margin": margin,
        "season": str(info.get("season", pd.to_datetime(dates[0]).year)),
    }


def _read_info(archive: zipfile.ZipFile, name: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    text = ""
    with archive.open(name) as member:
        while len(text) <= 256_000:
            chunk = member.read(8192)
            if not chunk:
                break
            text += chunk.decode("utf-8")
            info_key = text.find('"info"')
            if info_key < 0:
                continue
            colon = text.find(":", info_key + 6)
            if colon < 0:
                continue
            value_start = colon + 1
            while value_start < len(text) and text[value_start].isspace():
                value_start += 1
            try:
                info, _ = decoder.raw_decode(text, value_start)
            except json.JSONDecodeError:
                continue
            if isinstance(info, dict):
                return info
            break
    raise ValueError(f"Unable to read match info from {name}.")


def _load_cricsheet() -> pd.DataFrame:
    request = urllib.request.Request(
        CRICSHEET_URL, headers={"User-Agent": "FanCart lightweight forecaster"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = [
            name for name in archive.namelist()
            if name.lower().endswith(".json") and not name.startswith("__MACOSX/")
        ]
        parsed: list[dict[str, object]] = []
        for name in names:
            info = _read_info(archive, name)
            row = _match_from_info(info)
            if row is not None:
                parsed.append(row)
    frame = pd.DataFrame(parsed)
    if frame.empty:
        raise ValueError("Cricsheet archive contained no usable IPL results.")
    frame["date"] = pd.to_datetime(frame["date"])
    seasons = frame.groupby("season")["date"].max().sort_values().tail(3).index
    frame = frame[frame["season"].isin(seasons)].copy()
    frame["simulated"] = False
    return frame.sort_values("date").reset_index(drop=True)


def load_match_data() -> tuple[pd.DataFrame, str]:
    try:
        frame = _load_cricsheet()
        if frame.empty:
            raise ValueError("No match data was loaded.")
        return frame, "Cricsheet"
    except Exception:
        sample = generate_matches()
        sample["simulated"] = True
        return sample, "generated sample"
