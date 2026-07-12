"""
Supplemental pybaseball tool implementations
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from pybaseball import (
    get_splits,
    pitching_stats,
    pitching_stats_bref,
    pitching_stats_range,
    playerid_lookup,
    playerid_reverse_lookup,
    schedule_and_record,
    standings,
    team_batting,
    team_fielding,
    team_pitching,
    top_prospects,
)

from mlb_stats_mcp.utils.logging_config import setup_logging

logger = setup_logging("pybaseball_supp_tools")


def _convert_dataframe_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert a pandas DataFrame to a dictionary for JSON serialization.

    Args:
        df: The pandas DataFrame to convert

    Returns:
        Dictionary representation of the DataFrame
    """

    if df is None or df.empty:
        return {"data": [], "count": 0}

    try:
        df_clean = df.copy()
        df_clean = df_clean.reset_index()
        df_clean = df_clean.where(pd.notnull(df_clean), None)
        df_clean = df_clean.replace("", None)
        records = df_clean.to_dict(orient="records")
        return {"data": records, "count": len(records), "columns": df.columns.tolist()}

    except Exception as e:
        logger.error(f"Error in DF processing - {e}")
        return {
            "data": [],
            "count": 0,
            "error": f"DataFrame serialization failed: {e!s}",
        }


async def get_pitching_stats_bref(season: Optional[int] = None) -> Dict[str, Any]:
    """
    Get pitching stats from Baseball Reference for a given season.

    Args:
        season: The season to get data for. If None, pulls data for current year.

    Returns:
        Dictionary containing pitching stats from Baseball Reference

    Raises:
        Exception: If there's an error retrieving pitching stats
    """
    try:
        logger.debug(f"Retrieving Baseball Reference pitching stats for season: {season}")

        df = pitching_stats_bref(season)

        if len(df) == 0:
            raise Exception("No pitching stats data found")

        logger.debug(f"Retrieved {len(df)} pitching stats records from Baseball Reference")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving pitching stats from Baseball Reference: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_pitching_stats_range(
    start_dt: str,
    end_dt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get pitching stats from Baseball Reference for a date range.

    Args:
        start_dt: Beginning of date range (YYYY-MM-DD)
        end_dt: End of date range (YYYY-MM-DD). If None, returns data for start_dt only.

    Returns:
        Dictionary containing pitching stats for the date range

    Raises:
        Exception: If there's an error retrieving pitching stats
    """
    try:
        logger.debug(f"Retrieving pitching stats range from {start_dt} to {end_dt or start_dt}")

        df = pitching_stats_range(start_dt, end_dt)

        if len(df) == 0:
            raise Exception("No pitching stats data found for date range")

        logger.debug(f"Retrieved {len(df)} pitching stats records for date range")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving pitching stats range: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_pitching_stats(
    start_season: int,
    end_season: Optional[int] = None,
    league: str = "all",
    qual: Optional[int] = None,
    ind: int = 1,
) -> Dict[str, Any]:
    """
    Get season-level pitching data from FanGraphs.

    Args:
        start_season: First season to retrieve data from
        end_season: Final season to retrieve data from.
        If None, returns only start_season.
        league: Either "all", "nl", "al", or "mnl"
        qual: Minimum number of plate appearances to be included
        ind: 1 for individual season level, 0 for aggregate data

    Returns:
        Dictionary containing pitching stats from FanGraphs

    Raises:
        Exception: If there's an error retrieving pitching stats
    """
    try:
        logger.debug(
            f"Retrieving FanGraphs pitching stats for seasons {start_season} to "
            f"{end_season or start_season}, league: {league}, qual: {qual}, ind: {ind}"
        )

        try:
            df = pitching_stats(
                start_season=start_season,
                end_season=end_season or start_season,
                league=league.upper(),
                qual=qual,
                ind=ind,
            )
        except Exception as e:
            logger.error(
                f"[BREAKPOINT] error: {e} with args: "
                f"({start_season}, {end_season}, {league}, {qual}, {ind})"
            )

        if len(df) == 0:
            raise Exception("No pitching stats data found")

        logger.debug(f"Retrieved {len(df)} pitching stats records from FanGraphs")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving pitching stats from FanGraphs: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_playerid_lookup(
    last: str,
    first: Optional[str] = None,
    fuzzy: bool = False,
) -> Dict[str, Any]:
    """
    Look up a player's IDs by name.

    Args:
        last: Player's last name (case insensitive)
        first: Player's first name (case insensitive)
        fuzzy: Search for inexact name matches

    Returns:
        Dictionary containing player ID data

    Raises:
        Exception: If there's an error looking up player
    """
    try:
        logger.debug(f"Looking up player: {last}, {first}, fuzzy: {fuzzy}")

        df = playerid_lookup(last, first, fuzzy=fuzzy)

        logger.debug(f"Completed lookup df: {df}")
        if len(df) == 0:
            raise Exception(f"No player found matching {first} {last}")

        logger.debug(f"Found {len(df)} players matching search criteria")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error looking up player {first} {last}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def reverse_lookup_player(
    player_ids: List[int],
    key_type: str = "mlbam",
) -> Dict[str, Any]:
    """
    Find player names and IDs given a list of player IDs.

    Args:
        player_ids: List of player IDs
        key_type: Type of ID ('mlbam', 'retro', 'bbref', 'fangraphs')

    Returns:
        Dictionary containing player names and cross-referenced IDs

    Raises:
        Exception: If there's an error in reverse lookup
    """
    try:
        logger.debug(f"Reverse looking up {len(player_ids)} players with key_type: {key_type}")

        df = playerid_reverse_lookup(player_ids, key_type)

        if len(df) == 0:
            raise Exception("No players found for provided IDs")

        logger.debug(f"Found {len(df)} players in reverse lookup")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error in reverse player lookup: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_schedule_and_record(season: int, team: str) -> Dict[str, Any]:
    """
    Get a team's game-level results for a given season.

    Args:
        season: The season for which you want team record data
        team: Team abbreviation (e.g. "PHI", "BOS", "LAD")

    Returns:
        Dictionary containing team's schedule and record data

    Raises:
        Exception: If there's an error retrieving schedule data
    """
    try:
        logger.debug(f"Retrieving schedule and record for {team} in {season}")

        df = schedule_and_record(season, team)

        if len(df) == 0:
            raise Exception(f"No schedule data found for {team} in {season}")

        logger.debug(f"Retrieved {len(df)} games for {team} in {season}")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving schedule for {team} in {season}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_player_splits(
    playerid: str,
    year: Optional[int] = None,
    player_info: bool = False,
    pitching_splits: bool = False,
) -> Dict[str, Any]:
    """
    Look up a player's split stats from baseball-reference.

    Args:
        playerid: Player's bbref playerid (e.g. 'troutmi01')
        year: Year to get split stats for. If None, returns career splits.
        player_info: If True, returns both splits and player info
        pitching_splits: If True, returns pitching splits; otherwise batting splits

    Returns:
        Dictionary containing split stats data

    Raises:
        Exception: If there's an error retrieving split stats
    """
    try:
        logger.debug(
            f"Retrieving splits for player {playerid}, year: {year}, "
            f"pitching: {pitching_splits}, info: {player_info}"
        )

        result = get_splits(playerid, year, player_info, pitching_splits)

        if player_info:
            df, info_dict = result
            splits_data = _convert_dataframe_to_dict(df)
            splits_data["player_info"] = info_dict
            return splits_data
        else:
            df = result
            return _convert_dataframe_to_dict(df)

    except Exception as e:
        error_msg = f"Error retrieving splits for player {playerid}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_standings(season: Optional[int] = None) -> Dict[str, Any]:
    """
    Get division standings for a given season.

    Args:
        season: Season to get standings for. Defaults to current year if None.

    Returns:
        Dictionary containing division standings

    Raises:
        Exception: If there's an error retrieving standings
    """
    try:
        logger.debug(f"Retrieving standings for season: {season}")

        tables = standings(season)

        divisions = [
            "AL East",
            "AL Central",
            "AL West",
            "NL East",
            "NL Central",
            "NL West",
        ]

        # Convert each table and combine results
        all_data = {}
        for i, table in enumerate(tables):
            table_dict = _convert_dataframe_to_dict(table)
            all_data[divisions[i]] = table_dict
        return {"data": all_data, "divisions": divisions}
    except Exception as e:
        error_msg = f"Error retrieving standings for season {season}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_team_batting(
    start_season: int,
    end_season: Optional[int] = None,
    league: str = "all",
    ind: int = 1,
) -> Dict[str, Any]:
    """
    Get team-level batting stats.

    Args:
        start_season: First season for team batting data
        end_season: Last season for team batting data.
        If None, returns only start_season.
        league: Either "all", "nl", "al", or "mnl"
        ind: 1 for individual season level, 0 for aggregate data

    Returns:
        Dictionary containing team batting stats

    Raises:
        Exception: If there's an error retrieving team batting stats
    """
    try:
        logger.debug(
            f"Retrieving team batting stats for seasons {start_season} to "
            f"{end_season or start_season}, league: {league}, ind: {ind}"
        )

        df = team_batting(start_season, end_season, league, ind)

        if len(df) == 0:
            raise Exception("No team batting data found")

        logger.debug(f"Retrieved {len(df)} team batting records")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving team batting stats: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_team_fielding(
    start_season: int,
    end_season: Optional[int] = None,
    league: str = "all",
    ind: int = 1,
) -> Dict[str, Any]:
    """
    Get team-level fielding stats.

    Args:
        start_season: First season for team fielding data
        end_season: Last season for team fielding data.
        If None, returns only start_season.
        league: Either "all", "nl", "al", or "mnl"
        ind: 1 for individual season level, 0 for aggregate data

    Returns:
        Dictionary containing team fielding stats

    Raises:
        Exception: If there's an error retrieving team fielding stats
    """
    try:
        logger.debug(
            f"Retrieving team fielding stats for seasons {start_season} to "
            f"{end_season or start_season}, league: {league}, ind: {ind}"
        )

        df = team_fielding(start_season, end_season, league, ind)

        if len(df) == 0:
            raise Exception("No team fielding data found")

        logger.debug(f"Retrieved {len(df)} team fielding records")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving team fielding stats: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_team_pitching(
    start_season: int,
    end_season: Optional[int] = None,
    league: str = "all",
    ind: int = 1,
) -> Dict[str, Any]:
    """
    Get team-level pitching stats.

    Args:
        start_season: First season for team pitching data
        end_season: Last season for team pitching data.
        If None, returns only start_season.
        league: Either "all", "nl", "al", or "mnl"
        ind: 1 for individual season level, 0 for aggregate data

    Returns:
        Dictionary containing team pitching stats

    Raises:
        Exception: If there's an error retrieving team pitching stats
    """
    try:
        logger.debug(
            f"Retrieving team pitching stats for seasons {start_season} to "
            f"{end_season or start_season}, league: {league}, ind: {ind}"
        )

        df = team_pitching(start_season, end_season, league, ind)

        if len(df) == 0:
            raise Exception("No team pitching data found")

        logger.debug(f"Retrieved {len(df)} team pitching records")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving team pitching stats: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_top_prospects(
    team: Optional[str] = None,
    player_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get top prospects by team or leaguewide.

    Args:
        team: Team name (no whitespace). If None, returns leaguewide prospects.
        player_type: Either "pitchers" or "batters". If None, returns both.

    Returns:
        Dictionary containing top prospects data

    Raises:
        Exception: If there's an error retrieving prospects data
    """
    try:
        logger.debug(f"Retrieving top prospects for team: {team}, type: {player_type}")

        df = top_prospects(team, player_type)

        if len(df) == 0:
            raise Exception("No prospects data found")

        logger.debug(f"Retrieved {len(df)} prospects records")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving prospects data: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
