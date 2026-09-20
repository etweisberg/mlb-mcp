"""
Client for the FanGraphs major league leaderboard JSON API.

pybaseball's FanGraphs support scrapes ``/leaders-legacy.aspx``, which FanGraphs
retired: every request now comes back as HTTP 403. This module talks to the
JSON leaderboard API that backs the current site instead, and shapes the result
to match what pybaseball used to return so callers stay unchanged.

Note on the user agent: FanGraphs rejects requests that claim to be a browser,
so the honest client string below must be preserved.
"""

import re
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from mlb_stats_mcp.utils.logging_config import setup_logging

logger = setup_logging("fangraphs_api")

LEADERS_URL = "https://www.fangraphs.com/api/leaders/major-league/data"
USER_AGENT = "mlb-stats-mcp (+https://github.com/etweisberg/mlb-mcp)"
REQUEST_TIMEOUT = 60

# "8" is the leaderboard's full stat set, matching pybaseball's ALL columns.
_ALL_STAT_COLUMNS = "8"

# Team rows reuse the player schema, so a representative player's identity
# leaks into every row. Drop those columns rather than report them as team data.
_PLAYER_ONLY_COLUMNS = (
    "Bats",
    "Throws",
    "Name",
    "PlayerName",
    "PlayerNameRoute",
    "playerid",
    "xMLBAMID",
    "positionDB",
    "position",
    "playerTeamId",
)

_ANCHOR_TEXT = re.compile(r"<[^>]+>")

_SORT_COLUMNS = {
    "bat": ["WAR", "OPS"],
    "pit": ["WAR", "W"],
    "fld": ["Defense"],
}

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Return a module-level session so connections are reused across calls."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return _session


def _strip_html(value: Any) -> Any:
    """Reduce a FanGraphs anchor cell to its text, leaving other values alone."""
    if isinstance(value, str) and "<" in value:
        return _ANCHOR_TEXT.sub("", value).strip()
    return value


def _sort(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Sort descending by the first available column, mirroring pybaseball."""
    known = [column for column in columns if column in df.columns]
    if not known:
        return df
    return df.sort_values(known, ascending=False).reset_index(drop=True)


def _postprocess(df: pd.DataFrame, stats: str, team_data: bool) -> pd.DataFrame:
    """Shape a raw leaderboard payload into the frame pybaseball used to return."""
    if "PlayerName" in df.columns:
        df["Name"] = df["PlayerName"]
    if "TeamNameAbb" in df.columns:
        df["Team"] = df["TeamNameAbb"]

    for column in ("Name", "Team"):
        if column in df.columns:
            df[column] = df[column].map(_strip_html)

    if team_data:
        if "teamid" in df.columns:
            df.insert(0, "teamIDfg", df["teamid"])
        df = df.drop(columns=[c for c in _PLAYER_ONLY_COLUMNS if c in df.columns])
    elif "playerid" in df.columns:
        df.insert(0, "IDfg", df["playerid"])

    # Lead with the identifying columns; the stat columns keep FanGraphs' order.
    leading = [c for c in ("teamIDfg", "IDfg", "Name", "Team", "Season") if c in df.columns]
    df = df[leading + [c for c in df.columns if c not in leading]]

    return _sort(df, _SORT_COLUMNS.get(stats, []))


def fetch_leaders(
    stats: str,
    start_season: int,
    end_season: Optional[int] = None,
    league: str = "all",
    qual: Optional[int] = None,
    ind: int = 1,
    team_data: bool = False,
    max_results: int = 1000000,
) -> pd.DataFrame:
    """
    Fetch a FanGraphs leaderboard as a DataFrame.

    Args:
        stats: Leaderboard category - "bat", "pit", or "fld"
        start_season: First season to retrieve data from
        end_season: Final season to retrieve data from. If None, start_season
        league: Either "all", "nl", "al", or "mnl"
        qual: Minimum playing time to be included. If None, FanGraphs' default
        ind: 1 for individual season level, 0 for aggregate data
        team_data: True to aggregate rows by team rather than by player
        max_results: Maximum number of rows to request

    Returns:
        DataFrame of leaderboard rows

    Raises:
        Exception: If the leaderboard cannot be retrieved or parsed
    """
    if start_season is None:
        raise ValueError("start_season is required to query FanGraphs")

    params: Dict[str, Any] = {
        "pos": "all",
        "stats": stats,
        "lg": (league or "all").lower(),
        "qual": "y" if qual is None else qual,
        "type": _ALL_STAT_COLUMNS,
        "season": end_season or start_season,
        "season1": start_season,
        "month": 0,
        "ind": ind,
        "team": "0,ts" if team_data else "",
        "rost": 0,
        "age": "",
        "filter": "",
        "players": "",
        "pageitems": max_results,
        "pagenum": 1,
    }

    logger.debug(f"Requesting FanGraphs {stats} leaderboard with {params}")

    try:
        response = _get_session().get(LEADERS_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as e:
        raise Exception(f"Error accessing '{LEADERS_URL}': {e!s}") from e
    except ValueError as e:
        raise Exception(f"FanGraphs returned a non-JSON response from '{LEADERS_URL}'") from e

    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not rows:
        return pd.DataFrame()

    return _postprocess(pd.DataFrame(rows), stats, team_data)


def pitching_stats(
    start_season: int,
    end_season: Optional[int] = None,
    league: str = "all",
    qual: Optional[int] = None,
    ind: int = 1,
) -> pd.DataFrame:
    """Get season-level pitching data for individual players."""
    return fetch_leaders("pit", start_season, end_season, league, qual, ind)


def team_batting(
    start_season: int,
    end_season: Optional[int] = None,
    league: str = "all",
    ind: int = 1,
) -> pd.DataFrame:
    """Get season-level batting data aggregated by team."""
    return fetch_leaders("bat", start_season, end_season, league, qual=0, ind=ind, team_data=True)


def team_pitching(
    start_season: int,
    end_season: Optional[int] = None,
    league: str = "all",
    ind: int = 1,
) -> pd.DataFrame:
    """Get season-level pitching data aggregated by team."""
    return fetch_leaders("pit", start_season, end_season, league, qual=0, ind=ind, team_data=True)


def team_fielding(
    start_season: int,
    end_season: Optional[int] = None,
    league: str = "all",
    ind: int = 1,
) -> pd.DataFrame:
    """Get season-level fielding data aggregated by team."""
    return fetch_leaders("fld", start_season, end_season, league, qual=0, ind=ind, team_data=True)
