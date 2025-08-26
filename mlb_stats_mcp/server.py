"""
MCP server implementation for the baseball project with MLB Stats API integration.
"""

import contextlib
import inspect
import os
import sys
from typing import Any, Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP

from mlb_stats_mcp.prompts import prompts
from mlb_stats_mcp.tools import (
    mlb_statsapi_tools,
    pybaseball_plotting_tools,
    pybaseball_supp_tools,
    statcast_tools,
)
from mlb_stats_mcp.utils.logging_config import setup_logging

# Load environment variables from .env file
load_dotenv()

# Initialize logging for the server
logger = setup_logging("mcp_server")

# Initialize FastMCP server
mcp = FastMCP("baseball", stateless_http=True)


# Automatically register all prompt functions from prompts.py
def _register_prompts():
    """Automatically register all prompt functions from the prompts module."""
    # Get all prompt functions (those starting with 'cd_' or 'web_ui_')
    prompt_functions = [
        name
        for name in dir(prompts)
        if callable(getattr(prompts, name))
        and (name.startswith("cd_") or name.startswith("web_ui_"))
        and not name.startswith("_")
    ]

    for name in prompt_functions:
        func = getattr(prompts, name)

        # Register the prompt directly with MCP
        mcp.prompt()(func)

        # Add to global namespace so it's accessible
        globals()[name] = func

        logger.debug(f"Registered prompt function: {name}")


# Register all prompts automatically
_register_prompts()


def mcp_tool_wrapper(func):
    """Decorator to handle errors from tool functions."""
    sig = inspect.signature(func)

    # Create a wrapper function with the same signature as the original function
    async def wrapper(*args, **kwargs):
        try:
            logger.info(f"tool {func.__name__} called")
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e!s}")
            raise Exception(f"Error in {func.__name__}: {e!s}") from e

    # Copy the signature from the original function
    wrapper.__signature__ = sig

    # Register the tool with MCP
    return mcp.tool(name=func.__name__, description=func.__doc__)(wrapper)


# Core Data Gathering Tools
@mcp_tool_wrapper
async def get_stats(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return await mlb_statsapi_tools.get_stats(endpoint, params)


@mcp_tool_wrapper
async def get_schedule(
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    team_id: Optional[int] = None,
    opponent_id: Optional[int] = None,
    sport_id: int = 1,
    game_id: Optional[str] = None,
    season: Optional[str] = None,
    include_series_status: bool = True,
) -> Dict[str, Any]:
    """
    Get list of games for a given date/range and/or team/opponent.
    """
    return await mlb_statsapi_tools.get_schedule(
        date=date,
        start_date=start_date,
        end_date=end_date,
        team_id=team_id,
        opponent_id=opponent_id,
        sport_id=sport_id,
        game_id=game_id,
        season=season,
        include_series_status=include_series_status,
    )


@mcp_tool_wrapper
async def get_player_stats(
    player_id: int,
    group: str = "hitting",
    season: Optional[int] = None,
    stats: str = "season",
) -> Dict[str, Any]:
    """
    Returns a list of current season or career stat data for a given player.
    """
    return await mlb_statsapi_tools.get_player_stats(player_id, group, season, stats)


@mcp_tool_wrapper
async def get_standings(
    league_id: Optional[int] = None,
    division_id: Optional[int] = None,
    season: Optional[int] = None,
    standings_types: str = "regularSeason",
) -> Dict[str, Any]:
    """
    Returns a dict of standings data for a given league/division and season.
    """
    return await mlb_statsapi_tools.get_standings(league_id, division_id, season, standings_types)


# Team and Player Analysis Tools
@mcp_tool_wrapper
async def get_team_leaders(
    team_id: int,
    leader_category: str = "homeRuns",
    season: Optional[int] = None,
    leader_game_type: str = "R",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns a python list of stat leader data for a given team
    """
    return await mlb_statsapi_tools.get_team_leaders(
        team_id, leader_category, season, leader_game_type, limit
    )


@mcp_tool_wrapper
async def lookup_player(name: str) -> Dict[str, Any]:
    """Get data about players based on first, last, or full name."""
    return await mlb_statsapi_tools.lookup_player(name)


@mcp_tool_wrapper
async def get_boxscore(game_id: int, timecode: Optional[str] = None) -> Dict[str, Any]:
    """Get a formatted boxscore for a given game."""
    return await mlb_statsapi_tools.get_boxscore(game_id, timecode)


@mcp_tool_wrapper
async def get_team_roster(
    team_id: int,
    roster_type: str = "active",
    season: int = 2025,
    date: Optional[str] = None,
) -> Dict[str, Any]:
    """Get the roster for a given team."""
    return await mlb_statsapi_tools.get_team_roster(team_id, roster_type, season, date)


# Historical Context Tools
@mcp_tool_wrapper
async def get_game_pace(season: Optional[int] = None) -> Dict[str, Any]:
    """Returns data about pace of game for a given season (back to 1999)."""
    return await mlb_statsapi_tools.get_game_pace(season)


@mcp_tool_wrapper
async def get_meta(type_name: str, fields: Optional[str] = None) -> Dict[str, Any]:
    """Get available values from StatsAPI for use in other queries,
    or look up descriptions for values found in API results.

    For example, to get a list of leader categories to use when calling team_leaders():
    statsapi.meta('leagueLeaderTypes')
    """
    return await mlb_statsapi_tools.get_meta(type_name, fields)


@mcp_tool_wrapper
async def get_available_endpoints() -> Dict[str, Any]:
    """Get MLB StatsAPI endpoints directly"""
    return await mlb_statsapi_tools.get_available_endpoints()


@mcp_tool_wrapper
async def get_notes(endpoint: str) -> Dict[str, Any]:
    """Get additional notes on an endpoint"""
    return await mlb_statsapi_tools.get_notes(endpoint)


@mcp_tool_wrapper
async def get_game_scoring_play_data(game_id: int) -> Dict[str, Any]:
    """Returns a dictionary of scoring plays for a given game containing 3 keys:

    * home - home team data
    * away - away team data
    * plays - sorted list of scoring play data
    """
    return await mlb_statsapi_tools.get_game_scoring_play_data(game_id)


@mcp_tool_wrapper
async def get_last_game(team_id: int) -> Dict[str, Any]:
    """Get the gamePk (game_id) for the given team's most recent completed game."""
    return await mlb_statsapi_tools.get_last_game(team_id)


@mcp_tool_wrapper
async def get_league_leader_data(
    leader_categories: str,
    season: Optional[int] = None,
    limit: Optional[int] = None,
    stat_group: Optional[str] = None,
    league_id: Optional[int] = None,
    game_types: Optional[str] = None,
) -> Dict[str, Any]:
    """Returns a list of stat leaders overall or for a given league (103=AL, 104=NL)."""
    return await mlb_statsapi_tools.get_league_leader_data(
        leader_categories, season, limit, stat_group, league_id, game_types
    )


@mcp_tool_wrapper
async def get_linescore(game_id: int) -> Dict[str, Any]:
    """Get formatted linescore data for a specific MLB game."""
    return await mlb_statsapi_tools.get_linescore(game_id)


@mcp_tool_wrapper
async def get_next_game(team_id: int) -> Dict[str, Any]:
    """Get the game ID for a team's next scheduled game."""
    return await mlb_statsapi_tools.get_next_game(team_id)


@mcp_tool_wrapper
async def get_game_highlight_data(game_id: int) -> Dict[str, Any]:
    """Returns a list of highlight data for a given game."""
    return await mlb_statsapi_tools.get_game_highlight_data(game_id)


# Statcast Tools
@mcp_tool_wrapper
async def get_statcast_data(
    start_dt: Optional[str] = None,
    end_dt: Optional[str] = None,
    team: Optional[str] = None,
    verbose: bool = True,
    parallel: bool = True,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Dict[str, Any]:
    """Pulls statcast play-level data from Baseball Savant for a given date range.

    INPUTS:
    start_dt: YYYY-MM-DD : the first date for which you want statcast data
    end_dt: YYYY-MM-DD : the last date for which you want statcast data
    team: optional (defaults to None) :
        city abbreviation of the team you want data for (e.g. SEA or BOS)
    verbose: bool (defaults to True) :
        whether to print updates on query progress
    parallel: bool (defaults to True) :
        whether to parallelize HTTP requests in large queries
    start_row: optional (defaults to None) :
        starting row index for truncating large results (0-based, inclusive)
    end_row: optional (defaults to None) :
        ending row index for truncating large results (0-based, exclusive)

    Use start_row and end_row to limit response size when dealing with large datasets.
    If no arguments are provided, this will return yesterday's statcast data.
    If one date is provided, it will return that date's statcast data."""
    return await statcast_tools.get_statcast_data(
        start_dt, end_dt, team, verbose, parallel, start_row, end_row
    )


@mcp_tool_wrapper
async def get_statcast_batter_data(
    player_id: int,
    start_dt: Optional[str] = None,
    end_dt: Optional[str] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Pulls statcast pitch-level data from Baseball Savant for a given batter.

    ARGUMENTS
        start_dt : YYYY-MM-DD :
            the first date for which you want a player's statcast data
        end_dt : YYYY-MM-DD : the final date for which you want data
        player_id : INT :
            the player's MLBAM ID.
            Find this by via the get_playerid_lookup tool,
            finding the correct player, and selecting their key_mlbam.
        start_row: optional (defaults to None) :
            starting row index for truncating large results (0-based, inclusive)
        end_row: optional (defaults to None) :
            ending row index for truncating large results (0-based, exclusive)

    Use start_row and end_row to limit response size when dealing with large datasets.
    """
    return await statcast_tools.get_statcast_batter_data(
        player_id, start_dt, end_dt, start_row, end_row
    )


@mcp_tool_wrapper
async def get_statcast_pitcher_data(
    player_id: int,
    start_dt: Optional[str] = None,
    end_dt: Optional[str] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Pulls statcast pitch-level data from Baseball Savant for a given pitcher.

    ARGUMENTS
        start_dt : YYYY-MM-DD :
            the first date for which you want a player's statcast data
        end_dt : YYYY-MM-DD : the final date for which you want data
        player_id : INT :
            the player's MLBAM ID.
            Find this by calling pthe get_playerid_lookup tool,
            finding the correct player, and selecting their key_mlbam.
        start_row: optional (defaults to None) :
            starting row index for truncating large results (0-based, inclusive)
        end_row: optional (defaults to None) :
            ending row index for truncating large results (0-based, exclusive)

    Use start_row and end_row to limit response size when dealing with large datasets.
    """
    return await statcast_tools.get_statcast_pitcher_data(
        player_id, start_dt, end_dt, start_row, end_row
    )


@mcp_tool_wrapper
async def get_statcast_batter_exitvelo_barrels(
    year: int,
    minBBE: Optional[int] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Retrieves batted ball data for all batters in a given year.

    ARGUMENTS
        year: The year for which you wish to retrieve batted ball data. Format: YYYY.
        minBBE:
            The minimum number of batted ball events for each player.
            If a player falls below this threshold, they will be excluded from the results.
            If no value is specified, only qualified batters will be returned.
        start_row: optional (defaults to None) :
            starting row index for truncating large results (0-based, inclusive)
        end_row: optional (defaults to None) :
            ending row index for truncating large results (0-based, exclusive)

    Use start_row and end_row to limit response size when dealing with large datasets.
    """
    return await statcast_tools.get_statcast_batter_exitvelo_barrels(
        year, minBBE, start_row, end_row
    )


@mcp_tool_wrapper
async def get_statcast_pitcher_exitvelo_barrels(
    year: int,
    minBBE: Optional[int] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Retrieves batted ball against data for all qualified pitchers in a given year.

    ARGUMENTS
        year: The year for which you wish to retrieve batted ball against data. Format: YYYY.
        minBBE:
            The minimum number of batted ball against events for each pitcher.
            If a player falls below this threshold, they will be excluded from the results.
            If no value is specified, only qualified pitchers will be returned.
        start_row: optional (defaults to None) :
            starting row index for truncating large results (0-based, inclusive)
        end_row: optional (defaults to None) :
            ending row index for truncating large results (0-based, exclusive)

    Use start_row and end_row to limit response size when dealing with large datasets.
    """
    return await statcast_tools.get_statcast_pitcher_exitvelo_barrels(
        year, minBBE, start_row, end_row
    )


@mcp_tool_wrapper
async def get_statcast_batter_expected_stats(
    year: int,
    minPA: Optional[int] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Retrieves expected stats based on quality of batted ball contact in a given year.

    ARGUMENTS
        year: The year for which you wish to retrieve expected stats data. Format: YYYY.
        minPA: The minimum number of plate appearances for each player.
            If a player falls below this threshold, they will be excluded from the results.
            If no value is specified, only qualified batters will be returned.
        start_row: optional (defaults to None) :
            starting row index for truncating large results (0-based, inclusive)
        end_row: optional (defaults to None) :
            ending row index for truncating large results (0-based, exclusive)

    Use start_row and end_row to limit response size when dealing with large datasets.
    """
    return await statcast_tools.get_statcast_batter_expected_stats(year, minPA, start_row, end_row)


@mcp_tool_wrapper
async def get_statcast_pitcher_expected_stats(
    year: int,
    minPA: Optional[int] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Retrieves expected stats based on quality of batted ball contact against in a given year.

    ARGUMENTS
        year: The year for which you wish to retrieve expected stats data. Format: YYYY.
        minPA: The minimum number of plate appearances against for each pitcher.
            If a player falls below this
            threshold, they will be excluded from the results.
            If no value is specified, only qualified pitchers
            will be returned.
        start_row: optional (defaults to None) :
            starting row index for truncating large results (0-based, inclusive)
        end_row: optional (defaults to None) :
            ending row index for truncating large results (0-based, exclusive)

    Use start_row and end_row to limit response size when dealing with large datasets.
    """
    return await statcast_tools.get_statcast_pitcher_expected_stats(year, minPA, start_row, end_row)


@mcp_tool_wrapper
async def get_statcast_batter_percentile_ranks(
    year: int, start_row: Optional[int] = None, end_row: Optional[int] = None
) -> Dict[str, Any]:
    """
    Retrieves percentile ranks for batters in a given year.

    ARGUMENTS
        year: The year for which you wish to retrieve percentile data. Format: YYYY.
        start_row: optional (defaults to None) :
            starting row index for truncating large results (0-based, inclusive)
        end_row: optional (defaults to None) :
            ending row index for truncating large results (0-based, exclusive)

    Use start_row and end_row to limit response size when dealing with large datasets.
    """
    return await statcast_tools.get_statcast_batter_percentile_ranks(year, start_row, end_row)


@mcp_tool_wrapper
async def get_statcast_pitcher_percentile_ranks(
    year: int, start_row: Optional[int] = None, end_row: Optional[int] = None
) -> Dict[str, Any]:
    """
    Retrieves percentile ranks for each player in a given year,
    including batters with 2.1 PA per team game and 1.25
    for pitchers. It includes percentiles on expected stats,
    batted ball data, and spin rates, among others.

    ARGUMENTS
        year: The year for which you wish to retrieve percentile data. Format: YYYY.
        start_row: optional (defaults to None) :
            starting row index for truncating large results (0-based, inclusive)
        end_row: optional (defaults to None) :
            ending row index for truncating large results (0-based, exclusive)

    Use start_row and end_row to limit response size when dealing with large datasets.
    """
    return await statcast_tools.get_statcast_pitcher_percentile_ranks(year, start_row, end_row)


@mcp_tool_wrapper
async def get_statcast_batter_pitch_arsenal(
    year: int,
    minPA: int = 25,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Retrieves outcome data for batters split by the pitch type in a given year.

    ARGUMENTS
        year: The year for which you wish to retrieve pitch arsenal data. Format: YYYY.
        minPA: The minimum number of plate appearances for each player.
            If a player falls below this threshold,
                they will be excluded from the results.
            If no value is specified, the default number of plate appearances is 25.
        start_row: optional (defaults to None) :
            starting row index for truncating large results (0-based, inclusive)
        end_row: optional (defaults to None) :
            ending row index for truncating large results (0-based, exclusive)

    Use start_row and end_row to limit response size when dealing with large datasets.
    """
    return await statcast_tools.get_statcast_batter_pitch_arsenal(year, minPA, start_row, end_row)


@mcp_tool_wrapper
async def get_statcast_pitcher_pitch_arsenal(
    year: int,
    minP: Optional[int] = None,
    arsenal_type: str = "average_speed",
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Retrieves high level stats on each pitcher's arsenal in a given year.

    ARGUMENTS
        year: The year for which you wish to retrieve expected stats data. Format: YYYY.
        minP: The minimum number of pitches thrown.
            If a player falls below this threshold, they will be excluded
            from the results. If no value is specified, only qualified pitchers will be returned.
        arsenal_type:
            The type of stat to retrieve for the pitchers' arsenals.
            Options include ["average_speed", "n_", "average_spin"],
                where "n_" corresponds to the percentage share for each pitch.
            If no value is specified, it will default to average speed.
        start_row: optional (defaults to None) :
            starting row index for truncating large results (0-based, inclusive)
        end_row: optional (defaults to None) :
            ending row index for truncating large results (0-based, exclusive)

    Use start_row and end_row to limit response size when dealing with large datasets.
    """
    return await statcast_tools.get_statcast_pitcher_pitch_arsenal(
        year, minP, arsenal_type, start_row, end_row
    )


@mcp_tool_wrapper
async def get_statcast_single_game(
    game_pk: int, start_row: Optional[int] = None, end_row: Optional[int] = None
) -> Dict[str, Any]:
    """
    Pulls statcast play-level data from Baseball Savant for a single game,
    identified by its MLB game ID (game_pk in statcast data)

    INPUTS:
    game_pk : 6-digit integer MLB game ID to retrieve
    start_row: optional (defaults to None) :
        starting row index for truncating large results (0-based, inclusive)
    end_row: optional (defaults to None) :
        ending row index for truncating large results (0-based, exclusive)

    Use start_row and end_row to limit response size when dealing with large datasets.
    """
    return await statcast_tools.get_statcast_single_game(game_pk, start_row, end_row)


@mcp_tool_wrapper
async def create_strike_zone_plot(
    data: Dict[str, Any],
    title: str = "",
    colorby: str = "pitch_type",
    legend_title: str = "",
    annotation: str = "pitch_type",
) -> Dict[str, Any]:
    """
    Produces a pitches overlaid on a strike zone using StatCast data

    Args:
        data: (pandas.DataFrame)
            StatCast pandas.DataFrame of StatCast pitcher data
        title: (str), default = ''
            Optional: Title of plot
        colorby: (str), default = 'pitch_type'
            Optional: Which category to color the mark with.
            'pitch_type', 'pitcher', 'description' or a column within data
        legend_title: (str), default = based on colorby
            Optional: Title for the legend
        annotation: (str), default = 'pitch_type'
            Optional: What to annotate in the marker.
            'pitch_type', 'release_speed', 'effective_speed',
              'launch_speed', or something else in the data
    """
    return await pybaseball_plotting_tools.create_strike_zone_plot(
        data, title, colorby, legend_title, annotation
    )


@mcp_tool_wrapper
async def create_spraychart_plot(
    data: Dict[str, Any],
    team_stadium: str = "generic",
    title: str = "",
    colorby: str = "events",
    legend_title: str = "",
    size: int = 100,
    width: int = 500,
    height: int = 500,
) -> Dict[str, Any]:
    """
    Produces a spraychart using statcast data overlayed on specified stadium


    Args:
        data: (pandas.DataFrame)
            StatCast pandas.DataFrame of StatCast batter data
        team_stadium: (str)
            Team whose stadium the hits will be overlaid on
        title: (str), default = ''
            Optional: Title of plot
        size: (int), default = 100
            Optional: Size of hit circles on plot
        colorby: (str), default = 'events'
            Optional: Which category to color the mark with.
                'events','player', or a column within data
        legend_title: (str), default = based on colorby
            Optional: Title for the legend
        width: (int), default = 500
            Optional: Width of plot (not counting the legend)
        height: (int), default = 500
            Optional: Height of plot
    """
    return await pybaseball_plotting_tools.create_spraychart_plot(
        data, team_stadium, title, colorby, legend_title, size, width, height
    )


@mcp_tool_wrapper
async def create_bb_profile_plot(
    data: Dict[str, Any],
    parameter: str = "launch_angle",
) -> Dict[str, Any]:
    """Plots a given StatCast parameter split by bb_type

    Args:
        df: (pandas.DataFrame)
            pandas.DataFrame of StatCast batter data
            (retrieved through statcast, statcast_batter, etc)
        parameter: (str), default = 'launch_angle'
            Optional: Parameter to plot
    """
    return await pybaseball_plotting_tools.create_bb_profile_plot(data, parameter)


@mcp_tool_wrapper
async def create_teams_plot(
    data: Dict[str, Any],
    x_axis: str,
    y_axis: str,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Plots a scatter plot with each MLB team

    Args:
        data: (pandas.DataFrame)
            pandas.DataFrame of Fangraphs team data
                (retrieved through team_batting or team_pitching)
        x_axis: (str)
            Stat name to be plotted as the x_axis of the chart
        y_axis: (str)
            Stat name to be plotted as the y_axis of the chart
        title: (str), default = None
            Optional: Title of the plot
    """
    return await pybaseball_plotting_tools.create_teams_plot(data, x_axis, y_axis, title)


# Supplemental pybaseball tools
@mcp_tool_wrapper
async def get_pitching_stats_bref(season: Optional[int] = None) -> Dict[str, Any]:
    """
    Get all pitching stats for a set season. If no argument is supplied, gives stats for
    current season to date.
    """
    return await pybaseball_supp_tools.get_pitching_stats_bref(season)


@mcp_tool_wrapper
async def get_pitching_stats_range(
    start_dt: str,
    end_dt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get all pitching stats for a set time range. This can be the past week, the
    month of August, anything. Just supply the start and end date in YYYY-MM-DD
    format.
    """
    return await pybaseball_supp_tools.get_pitching_stats_range(start_dt, end_dt)


@mcp_tool_wrapper
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
    """
    return await pybaseball_supp_tools.get_pitching_stats(
        start_season, end_season, league, qual, ind
    )


@mcp_tool_wrapper
async def get_playerid_lookup(
    last: str,
    first: Optional[str] = None,
    fuzzy: bool = False,
) -> Dict[str, Any]:
    """Lookup playerIDs (MLB AM, bbref, retrosheet, FG) for a given player

    Args:
        last (str, required): Player's last name.
        first (str, optional): Player's first name. Defaults to None.
        fuzzy (bool, optional): In case of typos, returns players with names close to input.
            Defaults to False.

    Returns:
        pd.DataFrame: DataFrame of playerIDs, name, years played
    """
    return await pybaseball_supp_tools.get_playerid_lookup(last, first, fuzzy)


@mcp_tool_wrapper
async def reverse_lookup_player(
    player_ids: list[int],
    key_type: str = "mlbam",
) -> Dict[str, Any]:
    """Retrieve a table of player information given a list of player ids

    :param player_ids: list of player ids
    :type player_ids: list
    :param key_type: name of the key type being looked up
        (one of "mlbam", "retro", "bbref", or "fangraphs")
    :type key_type: str

    :rtype: :class:`pandas.core.frame.DataFrame`
    """
    return await pybaseball_supp_tools.reverse_lookup_player(player_ids, key_type)


@mcp_tool_wrapper
async def get_schedule_and_record(season: int, team: str) -> Dict[str, Any]:
    """
    Retrieve a team's game-level results for a given season,
        including win/loss/tie result, score, attendance,
    and winning/losing/saving pitcher. If the season is incomplete,
        it will provide scheduling information for
    future games.

    ARGUMENTS
        season: Integer. The season for which you want a team's record data.
        team: String.
        The abbreviation of the team for which you are requesting data (e.g. "PHI", "BOS", "LAD").
    """
    return await pybaseball_supp_tools.get_schedule_and_record(season, team)


@mcp_tool_wrapper
async def get_player_splits(
    playerid: str,
    year: Optional[int] = None,
    player_info: bool = False,
    pitching_splits: bool = False,
) -> Dict[str, Any]:
    """
    Returns a dataframe of all split stats for a given player.
    If player_info is True, this will also return a dictionary that includes player position,
        handedness, height, weight, position, and team
    """
    return await pybaseball_supp_tools.get_player_splits(
        playerid, year, player_info, pitching_splits
    )


@mcp_tool_wrapper
async def get_pybaseball_standings(season: Optional[int] = None) -> Dict[str, Any]:
    """
    Returns a pandas DataFrame of the standings for a given MLB season, or the most recent standings
    if the date is not specified.

    ARGUMENTS
        season (int): the year of the season
    """
    return await pybaseball_supp_tools.get_standings(season)


@mcp_tool_wrapper
async def get_team_batting(
    start_season: int,
    end_season: Optional[int] = None,
    league: str = "all",
    ind: int = 1,
) -> Dict[str, Any]:
    """
    Get season-level Batting Statistics for Specific Team (from Baseball-Reference)

    ARGUMENTS:
    team : str : The Team Abbreviation (i.e. 'NYY' for Yankees) of the Team you want data for
    start_season : int : first season you want data for
        (or the only season if you do not specify an end_season)
    end_season : int : final season you want data for
    """
    return await pybaseball_supp_tools.get_team_batting(start_season, end_season, league, ind)


@mcp_tool_wrapper
async def get_team_fielding(
    start_season: int,
    end_season: Optional[int] = None,
    league: str = "all",
    ind: int = 1,
) -> Dict[str, Any]:
    """
    Get season-level Fielding Statistics for Specific Team (from Baseball-Reference)

    ARGUMENTS:
    team            : str : The Team Abbreviation (i.e., 'NYY' for Yankees) of the
        Team you want data for
    start_season    : int : first season you want data for
        (or the only season if you do not specify an end_season)
    end_season      : int : final season you want data for
    """
    return await pybaseball_supp_tools.get_team_fielding(start_season, end_season, league, ind)


@mcp_tool_wrapper
async def get_team_pitching(
    start_season: int,
    end_season: Optional[int] = None,
    league: str = "all",
    ind: int = 1,
) -> Dict[str, Any]:
    """
    Get season-level Pitching Statistics for Specific Team (from Baseball-Reference)

    ARGUMENTS:
    team : str : The Team Abbreviation (i.e. 'NYY' for Yankees) of the Team you want data for
    start_season : int : first season you want data for
        (or the only season if you do not specify an end_season)
    end_season : int : final season you want data for
    """
    return await pybaseball_supp_tools.get_team_pitching(start_season, end_season, league, ind)


@mcp_tool_wrapper
async def get_top_prospects(
    team: Optional[str] = None,
    player_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves the top prospects by team or leaguewide.
        It can return top prospect pitchers, batters, or both.

    ARGUMENTS
    team: The team name for which you wish to retrieve top prospects. If not specified,
        the function will return leaguewide top prospects.
    playerType: Either "pitchers" or "batters".
        If not specified, the function will return top prospects for both
        pitchers and batters.
    """
    return await pybaseball_supp_tools.get_top_prospects(team, player_type)


# Add this lifespan manager after your mcp = FastMCP("baseball") line
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the MCP session manager."""
    async with mcp.session_manager.run():
        yield


# Add this function to create the FastAPI app
def create_fastapi_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # Your Next.js dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/", mcp.streamable_http_app())
    return app


def main():
    """Initialize and run the MCP baseball server."""
    # Check if running with HTTP transport
    if "--http" in sys.argv:
        logger.info("Starting MLB Stats MCP server over HTTP with FastAPI")
        port = int(os.environ.get("PORT", 8081))
        uvicorn.run(
            "mlb_stats_mcp.server:create_fastapi_app",
            host="0.0.0.0",  # This binds to all interfaces
            port=port,
            log_level="info",
            reload=False,
            factory=True,
        )
    else:
        logger.info("Starting MLB Stats MCP server with stdio transport")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
