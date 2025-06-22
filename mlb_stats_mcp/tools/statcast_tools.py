"""
Statcast tool implementations for pybaseball statcast functions
"""

import datetime
from typing import Any, Dict, Optional

import pandas as pd
from pybaseball import (
    statcast,
    statcast_batter,
    statcast_batter_exitvelo_barrels,
    statcast_batter_expected_stats,
    statcast_batter_percentile_ranks,
    statcast_batter_pitch_arsenal,
    statcast_pitcher,
    statcast_pitcher_exitvelo_barrels,
    statcast_pitcher_expected_stats,
    statcast_pitcher_percentile_ranks,
    statcast_pitcher_pitch_arsenal,
    statcast_single_game,
)

from mlb_stats_mcp.utils.logging_config import setup_logging

# Initialize logging for the Statcast tools
logger = setup_logging("statcast_tools")


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
        # Create a copy
        df_clean = df.copy()

        # Replace all NaN values with None using pandas' built-in method
        df_clean = df_clean.where(pd.notnull(df_clean), None)

        # Also replace empty strings with None
        df_clean = df_clean.replace("", None)

        # Convert to dict
        records = df_clean.to_dict(orient="records")

        return {"data": records, "count": len(records), "columns": df.columns.tolist()}

    except Exception as e:
        return {
            "data": [],
            "count": 0,
            "error": f"DataFrame serialization failed: {e!s}",
        }


async def get_statcast_data(
    start_dt: Optional[str] = None,
    end_dt: Optional[str] = None,
    team: Optional[str] = None,
    verbose: bool = True,
    parallel: bool = True,
) -> Dict[str, Any]:
    """
    Get pitch-level statcast data for a given date range.

    Args:
        start_dt: First day for which you want to retrieve data (YYYY-MM-DD)
        end_dt: Last day for which you want to retrieve data (YYYY-MM-DD)
        team: MLB team abbreviation (e.g., 'NYY', 'BOS', etc.)
        verbose: If True, provides updates on query progress
        parallel: If True, parallelize HTTP requests for large queries

    Returns:
        Dictionary containing Statcast data

    Raises:
        Exception: If there's an error retrieving Statcast data
    """
    try:
        # If start_dt is None, use yesterday's date
        if start_dt is None:
            yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
            start_dt = yesterday.strftime("%Y-%m-%d")
            logger.debug(f"No start date provided, using yesterday: {start_dt}")

        logger.debug(
            f"Retrieving Statcast data from {start_dt} to {end_dt or start_dt} "
            f"{'for team ' + team if team else ''}"
        )

        # Call pybaseball's statcast function
        df = statcast(start_dt, end_dt, team, verbose, parallel)

        if len(df) == 0:
            raise Exception("No statcast data found")

        if len(df) == 0:
            raise Exception("No statcast data found")

        logger.debug(f"Retrieved {len(df) if df is not None else 0} Statcast records")

        json_output = _convert_dataframe_to_dict(df)
        logger.debug(f"JSON Output: {json_output}")
        return json_output
    except Exception as e:
        error_msg = f"Error retrieving Statcast data: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_statcast_batter_data(
    player_id: int,
    start_dt: Optional[str] = None,
    end_dt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get pitch-level statcast data for a specific batter.

    Args:
        player_id: MLBAM player ID for the batter
        start_dt: First day for which you want to retrieve data (YYYY-MM-DD)
        end_dt: Last day for which you want to retrieve data (YYYY-MM-DD)

    Returns:
        Dictionary containing batter-specific Statcast data

    Raises:
        Exception: If there's an error retrieving batter Statcast data
    """
    try:
        # If start_dt is None, use yesterday's date
        if start_dt is None:
            yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
            start_dt = yesterday.strftime("%Y-%m-%d")
            logger.debug(f"No start date provided, using yesterday: {start_dt}")

        logger.debug(
            f"Retrieving Statcast data for batter ID{player_id} from {start_dt} to {end_dt}"
        )

        # Call pybaseball's statcast_batter function
        df = statcast_batter(start_dt, end_dt, player_id)

        if len(df) == 0:
            raise Exception("No statcast data found")

        logger.debug(
            f"Retrieved {len(df) if df is not None else 0} Statcast records for batter {player_id}"
        )

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving Statcast data for batter {player_id}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_statcast_pitcher_data(
    player_id: int,
    start_dt: Optional[str] = None,
    end_dt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get pitch-level statcast data for a specific pitcher.

    Args:
        player_id: MLBAM player ID for the pitcher
        start_dt: First day for which you want to retrieve data (YYYY-MM-DD)
        end_dt: Last day for which you want to retrieve data (YYYY-MM-DD)

    Returns:
        Dictionary containing pitcher-specific Statcast data

    Raises:
        Exception: If there's an error retrieving pitcher Statcast data
    """
    try:
        # If start_dt is None, use yesterday's date
        if start_dt is None:
            yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
            start_dt = yesterday.strftime("%Y-%m-%d")
            logger.debug(f"No start date provided, using yesterday: {start_dt}")

        logger.debug(
            f"Retrieving Statcast data for pitcher ID {player_id} from {start_dt} to {end_dt}"
        )

        # Call pybaseball's statcast_pitcher function
        df = statcast_pitcher(start_dt, end_dt, player_id)

        if len(df) == 0:
            raise Exception("No statcast data found")

        logger.debug(
            f"Retrieved {len(df) if df is not None else 0} Statcast records for pitcher {player_id}"
        )

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving Statcast data for pitcher {player_id}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_statcast_batter_exitvelo_barrels(
    year: int,
    minBBE: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get batted ball data for all batters in a given year.

    Args:
        year: The year for which you wish to retrieve batted ball data (YYYY)
        minBBE: The minimum number of batted ball events for each player

    Returns:
        Dictionary containing batted ball data for batters

    Raises:
        Exception: If there's an error retrieving batter exit velocity and barrel data
    """
    try:
        logger.debug(
            f"Retrieving batter exit velocity and barrel data for {year} "
            f"with minimum BBE: {minBBE if minBBE is not None else 'qualified'}"
        )

        # Call pybaseball's statcast_batter_exitvelo_barrels function
        df = statcast_batter_exitvelo_barrels(year, minBBE)

        if len(df) == 0:
            raise Exception("No statcast data found")

        logger.debug(f"Retrieved exit velocity data for {len(df) if df is not None else 0} batters")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving batter exit velocity data for {year}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_statcast_pitcher_exitvelo_barrels(
    year: int,
    minBBE: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get batted ball against data for all pitchers in a given year.

    Args:
        year: The year for which you wish to retrieve batted ball against data (YYYY)
        minBBE: The minimum number of batted ball against events for each pitcher

    Returns:
        Dictionary containing batted ball against data for pitchers

    Raises:
        Exception: If there's an error retrieving pitcher exit velocity and barrel data
    """
    try:
        logger.debug(
            f"Retrieving pitcher exit velocity and barrel data for {year} "
            f"with minimum BBE: {minBBE if minBBE is not None else 'qualified'}"
        )

        # Call pybaseball's statcast_pitcher_exitvelo_barrels function
        df = statcast_pitcher_exitvelo_barrels(year, minBBE)

        if len(df) == 0:
            raise Exception("No statcast data found")

        logger.debug(
            f"Retrieved exit velocity data for {len(df) if df is not None else 0} pitchers"
        )

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving pitcher exit velocity data for {year}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_statcast_batter_expected_stats(
    year: int,
    minPA: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get expected stats based on quality of batted ball contact in a given year.

    Args:
        year: The year for which you wish to retrieve expected stats data (YYYY)
        minPA: The minimum number of plate appearances for each player

    Returns:
        Dictionary containing expected stats data for batters

    Raises:
        Exception: If there's an error retrieving batter expected stats
    """
    try:
        logger.debug(
            f"Retrieving batter expected stats for {year} "
            f"with minimum PA: {minPA if minPA is not None else 'qualified'}"
        )

        # Call pybaseball's statcast_batter_expected_stats function
        df = statcast_batter_expected_stats(year, minPA)

        if len(df) == 0:
            raise Exception("No statcast data found")

        logger.debug(f"Retrieved expected stats for {len(df) if df is not None else 0} batters")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving batter expected stats for {year}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_statcast_pitcher_expected_stats(
    year: int,
    minPA: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get expected stats based on quality of batted ball contact against in a given year.

    Args:
        year: The year for which you wish to retrieve expected stats data (YYYY)
        minPA: The minimum number of plate appearances against for each player

    Returns:
        Dictionary containing expected stats data for pitchers

    Raises:
        Exception: If there's an error retrieving pitcher expected stats
    """
    try:
        logger.debug(
            f"Retrieving pitcher expected stats for {year} "
            f"with minimum PA: {minPA if minPA is not None else 'qualified'}"
        )

        # Call pybaseball's statcast_pitcher_expected_stats function
        df = statcast_pitcher_expected_stats(year, minPA)

        if len(df) == 0:
            raise Exception("No statcast data found")

        logger.debug(f"Retrieved expected stats for {len(df) if df is not None else 0} pitchers")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving pitcher expected stats for {year}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_statcast_batter_percentile_ranks(
    year: int,
) -> Dict[str, Any]:
    """
    Get percentile ranks for batters in a given year.

    Args:
        year: The year for which you wish to retrieve percentile data (YYYY)

    Returns:
        Dictionary containing percentile ranks for batters

    Raises:
        Exception: If there's an error retrieving batter percentile ranks
    """
    try:
        logger.debug(f"Retrieving batter percentile ranks for {year}")

        # Call pybaseball's statcast_batter_percentile_ranks function
        df = statcast_batter_percentile_ranks(year)

        if len(df) == 0:
            raise Exception("No statcast data found")

        logger.debug(f"Retrieved percentile ranks for {len(df) if df is not None else 0} batters")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving batter percentile ranks for {year}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_statcast_pitcher_percentile_ranks(
    year: int,
) -> Dict[str, Any]:
    """
    Get percentile ranks for pitchers in a given year.

    Args:
        year: The year for which you wish to retrieve percentile data (YYYY)

    Returns:
        Dictionary containing percentile ranks for pitchers

    Raises:
        Exception: If there's an error retrieving pitcher percentile ranks
    """
    try:
        logger.debug(f"Retrieving pitcher percentile ranks for {year}")

        # Call pybaseball's statcast_pitcher_percentile_ranks function
        df = statcast_pitcher_percentile_ranks(year)

        if len(df) == 0:
            raise Exception("No statcast data found")

        logger.debug(f"Retrieved percentile ranks for {len(df) if df is not None else 0} pitchers")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving pitcher percentile ranks for {year}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_statcast_batter_pitch_arsenal(
    year: int,
    minPA: int = 25,
) -> Dict[str, Any]:
    """
    Get outcome data for batters split by the pitch type in a given year.

    Args:
        year: The year for which you wish to retrieve pitch arsenal data (YYYY)
        minPA: The minimum number of plate appearances for each player

    Returns:
        Dictionary containing pitch arsenal data for batters

    Raises:
        Exception: If there's an error retrieving batter pitch arsenal data
    """
    try:
        logger.debug(f"Retrieving batter pitch arsenal data for {year} with minimum PA: {minPA}")

        # Call pybaseball's statcast_batter_pitch_arsenal function
        df = statcast_batter_pitch_arsenal(year, minPA)

        if len(df) == 0:
            raise Exception("No statcast data found")

        logger.debug(f"Retrieved pitch arsenal data for {len(df) if df is not None else 0} batters")

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving batter pitch arsenal data for {year}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_statcast_pitcher_pitch_arsenal(
    year: int,
    minP: Optional[int] = None,
    arsenal_type: str = "avg_speed",
) -> Dict[str, Any]:
    """
    Get high level stats on each pitcher's arsenal in a given year.

    Args:
        year: The year for which you wish to retrieve pitch arsenal data (YYYY)
        minP: The minimum number of pitches thrown
        arsenal_type: The type of stat to retrieve for pitchers' arsenals
            Options: "avg_speed", "n_", "avg_spin"

    Returns:
        Dictionary containing pitch arsenal data for pitchers

    Raises:
        Exception: If there's an error retrieving pitcher pitch arsenal data
    """
    try:
        logger.debug(
            f"Retrieving pitcher pitch arsenal data for {year} "
            f"with minimum pitches: {minP if minP is not None else 'qualified'} "
            f"and arsenal type: {arsenal_type}"
        )

        # Call pybaseball's statcast_pitcher_pitch_arsenal function
        df = statcast_pitcher_pitch_arsenal(year, minP, arsenal_type)

        if len(df) == 0:
            raise Exception("No statcast data found")

        logger.debug(
            f"Retrieved pitch arsenal data for {len(df) if df is not None else 0} pitchers"
        )

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving pitcher pitch arsenal data for {year}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_statcast_single_game(
    game_pk: int,
) -> Dict[str, Any]:
    """
    Get all statcast data for a single MLB game specified by its game_pk.

    Args:
        game_pk: MLB game ID to retrieve data for

    Returns:
        Dictionary containing Statcast data for the specified game

    Raises:
        Exception: If there's an error retrieving single game Statcast data
    """
    try:
        logger.debug(f"Retrieving Statcast data for game ID: {game_pk}")

        # Call pybaseball's statcast_single_game function
        df = statcast_single_game(game_pk)

        if len(df) == 0:
            raise Exception("No statcast data found")

        logger.debug(
            f"Retrieved {len(df) if df is not None else 0} Statcast records for game {game_pk}"
        )

        return _convert_dataframe_to_dict(df)
    except Exception as e:
        error_msg = f"Error retrieving Statcast data for game {game_pk}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
