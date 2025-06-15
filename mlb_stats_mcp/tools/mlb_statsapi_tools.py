"""
MLB Stats API tool implementations for the baseball project.
Contains the core functionality for interacting with the MLB Stats API.
"""

from typing import Any, Dict, Optional

import statsapi

from mlb_stats_mcp.utils.logging_config import setup_logging

# Initialize logging for the MLB Stats API tools
logger = setup_logging("mlb_statsapi_tools")


# Core Data Gathering Tools
async def get_stats(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Access any endpoint in the MLB Stats API with custom parameters.

    Args:
        endpoint: The API endpoint to query (e.g., 'stats', 'schedule', 'standings')
        params: Dictionary of parameters to pass to the API

    Returns:
        JSON response from the MLB Stats API

    Raises:
        Exception: If there's an error accessing the MLB Stats API
    """
    try:
        logger.debug(f"Calling MLB Stats API endpoint: {endpoint} with params: {params}")
        result = statsapi.get(endpoint, params)
        logger.debug(f"MLB Stats API response received for endpoint: {endpoint}")
        return result
    except Exception as e:
        error_msg = f"Error accessing MLB Stats API endpoint {endpoint}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


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
    Get MLB game schedule information for specified criteria.

    Use this tool to find games by date range, specific teams, or individual games.
    You can search by a single date, date range, specific teams, or combinations.

    Args:
        date: Specific date in format 'MM/DD/YYYY' or 'YYYY-MM-DD'
        start_date: Start of date range in format 'MM/DD/YYYY' or 'YYYY-MM-DD'
        end_date: End of date range in format 'MM/DD/YYYY' or 'YYYY-MM-DD'
        team_id: MLB team ID (e.g., 143 for Phillies, 121 for Mets)
        opponent_id: Opponent team ID to find head-to-head matchups
        sport_id: Sport ID (1 for MLB, default)
        game_id: Specific game ID to get details for one game
        season: Season year (e.g., '2023')
        include_series_status: Whether to include series status info

    Returns:
        Dictionary with "games" key containing list of game dictionaries.
        Each game contains comprehensive info including:
        - Basic info: game_id, game_date, game_datetime, status
        - Teams: away_name, home_name, away_id, home_id
        - Scores: away_score, home_score, winning/losing teams
        - Pitchers: probable starters, winning/losing/save pitchers with notes
        - Venue: venue_id, venue_name
        - Broadcast: national_broadcasts list
        - Game context: doubleheader status, game_num, series_status
        - Live game info: current_inning, inning_state
        - Summary: formatted game summary string

    Examples:
        - Get today's games: get_schedule(date="06/01/2025")
        - Get team's games in date range:
            get_schedule(start_date="07/01/2018", end_date="07/31/2018", team_id=143)
        - Get head-to-head series:
            get_schedule(
                start_date="07/01/2018",
                end_date="07/31/2018",
                team_id=143,
                opponent_id=121
            )
        - Get specific game: get_schedule(game_id="530769")
        - Get full season: get_schedule(season="2023", team_id=143)
    """
    try:
        kwargs = {}
        if date:
            kwargs["date"] = date
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        if team_id:
            kwargs["team"] = team_id
        if opponent_id:
            kwargs["opponent"] = opponent_id
        if game_id:
            kwargs["game_id"] = game_id
        if season:
            kwargs["season"] = season
        kwargs["sportId"] = sport_id
        kwargs["include_series_status"] = include_series_status

        logger.debug(f"Retrieving schedule with params: {kwargs}")
        result = statsapi.schedule(**kwargs)
        logger.debug(f"Retrieved schedule data: {len(result)} game(s)")
        return {"games": result}
    except Exception as e:
        error_msg = f"Error retrieving schedule: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_player_stats(
    player_id: int,
    group: str = "hitting",
    season: Optional[int] = None,
    stats: str = "season",
) -> Dict[str, Any]:
    """
    Get comprehensive player statistics.

    Args:
        player_id: MLB player ID
        group: Stat group (hitting, pitching, fielding)
        season: Season year (defaults to current season)
        stats: Stat type (For type use career or season or yearByYear.
        Include multiple types in the following format
        (this is a string, not actually a list):
        group='[career,season,yearByYear]')

    Returns:
        Player statistics from the MLB Stats API

    Raises:
        Exception: If there's an error retrieving player stats
    """
    try:
        kwargs = {
            "personId": player_id,
            "group": group,
            "type": stats,
        }

        if season is not None:
            kwargs["season"] = season

        logger.debug(f"Stats for player ID: {player_id}, group: {group}, type: {stats}")
        result = statsapi.player_stat_data(**kwargs)
        logger.debug(f"Retrieved stats for player ID: {player_id}")
        return result
    except Exception as e:
        error_msg = f"Error retrieving player stats for player ID {player_id}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_standings(
    league_id: Optional[int] = None,
    division_id: Optional[int] = None,
    season: Optional[int] = None,
    standings_types: str = "regularSeason",
) -> Dict[str, Any]:
    """
    Get team standings information.

    Args:
        league_id: MLB league ID to filter by
        division_id: MLB division ID to filter by
        season: Season year (defaults to current season)
        standings_types: Type of standings (regularSeason, springTraining, etc.)

    Returns:
        Standings data from the MLB Stats API

    Raises:
        Exception: If there's an error retrieving standings
    """
    try:
        kwargs = {"standingsTypes": standings_types}

        if league_id is not None:
            kwargs["leagueId"] = league_id
        if division_id is not None:
            kwargs["divisionId"] = division_id
        if season is not None:
            kwargs["season"] = season

        logger.debug(f"Retrieving standings with params: {kwargs}")
        result = statsapi.standings_data(**kwargs)
        logger.debug(f"Retrieved standings data for {standings_types}")
        return result
    except Exception as e:
        error_msg = f"Error retrieving standings: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


# Team and Player Analysis
async def get_team_roster(
    team_id: int,
    roster_type: str = "active",
    season: int = 2025,
    date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get team roster information.

    Args:
        team_id: MLB team ID
        roster_type: Type of roster (active, 40man, etc.)
        season: Season year (defaults to current season)
        date: Date in YYYY-MM-DD format (defaults to today)

    Returns:
        Team roster data from the MLB Stats API

    Raises:
        Exception: If there's an error retrieving team roster
    """
    try:
        logger.debug(f"Retrieving team roster for team ID: {team_id}")
        result = statsapi.roster(team_id, roster_type, season, date)
        logger.debug(f"Retrieved team roster data for team ID: {team_id}")
        return result
    except Exception as e:
        error_msg = f"Error retrieving team roster for team ID {team_id}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_team_leaders(
    team_id: int,
    leader_category: str = "homeRuns",
    season: Optional[int] = None,
    leader_game_type: str = "R",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get team statistical leaders.

    Args:
        team_id: MLB team ID
        leader_category: Statistic to sort by (homeRuns, battingAverage, etc.)
        season: Season year (defaults to current season)
        leader_game_type: Game type (R=Regular Season, etc.)
        limit: Number of leaders to return (defaults to all)

    Returns:
        Team leader data from the MLB Stats API

    Raises:
        Exception: If there's an error retrieving team leaders
    """
    try:
        logger.debug(f"Retrieving team leaders for team: {team_id} | category: {leader_category}")

        leaders_text = statsapi.team_leaders(
            team_id,
            leader_category,
            limit=10 if limit is None else limit,
            leaderGameTypes=leader_game_type,
            season=season,
        )

        logger.debug(f"Retrieved team leaders data for team ID: {team_id}")

        return {
            "teamId": team_id,
            "leaderCategory": leader_category,
            "season": season,
            "results": leaders_text,
            "teamLeaders": True,
        }
    except Exception as e:
        error_msg = f"Error retrieving team leaders for team ID {team_id}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def lookup_player(name: str) -> Dict[str, Any]:
    """
    Look up a player by name to get their MLB ID.

    Args:
        name: Player name to search for

    Returns:
        Player lookup results from the MLB Stats API

    Raises:
        Exception: If there's an error looking up player
    """
    try:
        logger.debug(f"Looking up player with name: {name}")
        result = statsapi.lookup_player(name)

        if result:
            logger.debug(f"Found {len(result)} player(s) matching: {name}")
            return {"people": result}
        else:
            logger.info(f"No players found matching: {name}")
            raise Exception(f"No players found matching: {name}")
    except Exception as e:
        error_msg = f"Error looking up player {name}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_boxscore(game_id: int, timecode: Optional[str] = None) -> Dict[str, Any]:
    """
    Get detailed boxscore data for a game.

    Args:
        game_id: MLB game ID
        timecode: Optional timecode for in-game data

    Returns:
        Boxscore data from the MLB Stats API

    Raises:
        Exception: If there's an error retrieving boxscore
    """
    try:
        logger.debug(f"Retrieving boxscore for game ID: {game_id}")

        # Add timecode if provided
        kwargs = {}
        if timecode:
            kwargs["timecode"] = timecode
            logger.debug(f"Using timecode: {timecode}")

        boxscore_text = statsapi.boxscore(game_id, **kwargs)
        logger.debug(f"Retrieved boxscore data for game ID: {game_id}")

        # Create a structured response
        return {"game_id": game_id, "boxscore": boxscore_text, "success": True}
    except Exception as e:
        error_msg = f"Error retrieving boxscore for game ID {game_id}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


# Historical Context Tools
async def get_game_pace(season: Optional[int] = None) -> Dict[str, Any]:
    """
    Get game pace data to analyze how game length affects performance.

    Args:
        season: Season year (defaults to current season)
        team_id: MLB team ID to filter by

    Returns:
        Game pace data from the MLB Stats API

    Raises:
        Exception: If there's an error retrieving game pace data
    """
    try:
        kwargs = {"sportId": 1}  # 1 is MLB

        if season is not None:
            kwargs["season"] = season

        logger.debug(f"Retrieving game pace data with params: {kwargs}")
        result = statsapi.game_pace_data(**kwargs)

        season_txt = f"season {season}" if season else "current season"
        logger.debug(f"Checking season: {season_txt}")

        return result
    except Exception as e:
        error_msg = f"Error retrieving game pace data: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_meta(type_name: str, fields: Optional[str] = None) -> Dict[str, Any]:
    """
    Get available values from StatsAPI for use in other queries,
        or look up descriptions.

    Args:
        type_name: Type of metadata to retrieve
            (e.g., 'leagueLeaderTypes', 'positions', 'statGroups')
        fields: Optional fields to return (limits response fields)

    Returns:
        Metadata information from the MLB Stats API

    Raises:
        Exception: If there's an error retrieving metadata
    """
    try:
        logger.debug(f"Retrieving metadata for type: {type_name}")
        if fields:
            logger.debug(f"With field filtering: {fields}")
            result = statsapi.meta(type_name, fields=fields)
        else:
            result = statsapi.meta(type_name)

        logger.debug(f"Retrieved metadata for type: {type_name}")
        return result
    except Exception as e:
        error_msg = f"Error retrieving metadata for type {type_name}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_available_endpoints() -> Dict[str, Any]:
    """
    Get information about all available MLB Stats API endpoints that can be used with
    the get_stats tool.

    Returns:
        A dictionary containing details about all MLB Stats API endpoints

    Raises:
        Exception: If there's an error retrieving endpoint information
    """
    try:
        logger.debug("Retrieving available MLB Stats API endpoints information")

        # Define all available endpoints with their details
        endpoints = {
            "attendance": {
                "url": "https://statsapi.mlb.com/api/{ver}/attendance",
                "required_params": ["teamId", "leagueId", "leagueListId"],
                "all_params": [
                    "ver",
                    "teamId",
                    "leagueId",
                    "season",
                    "date",
                    "leagueListId",
                    "gameType",
                    "fields",
                ],
                "notes": None,
            },
            "awards": {
                "url": "https://statsapi.mlb.com/api/{ver}/awards{awardId}{recipients}",
                "required_params": [],
                "all_params": [
                    "ver",
                    "awardId",
                    "recipients",
                    "sportId",
                    "leagueId",
                    "season",
                    "hydrate",
                    "fields",
                ],
                "notes": ("Call awards endpoint with no parameters to return a list of awardIds."),
            },
            "conferences": {
                "url": "https://statsapi.mlb.com/api/{ver}/conferences",
                "required_params": [],
                "all_params": ["ver", "conferenceId", "season", "fields"],
                "notes": None,
            },
            "divisions": {
                "url": "https://statsapi.mlb.com/api/{ver}/divisions",
                "required_params": [],
                "all_params": ["ver", "divisionId", "leagueId", "sportId", "season"],
                "notes": (
                    "Call divisions endpoint with no parameters to return a list of divisions."
                ),
            },
            "draft": {
                "url": "https://statsapi.mlb.com/api/{ver}/draft{prospects}{year}{latest}",
                "required_params": [],
                "all_params": [
                    "ver",
                    "prospects",
                    "year",
                    "latest",
                    "limit",
                    "fields",
                    "round",
                    "name",
                    "school",
                    "state",
                    "country",
                    "position",
                    "teamId",
                    "playerId",
                    "bisPlayerId",
                ],
                "notes": (
                    "No query parameters are honored when 'latest' endpoint is queried "
                    "(year is still required). Prospects and Latest cannot be used "
                    "together."
                ),
            },
            "game": {
                "url": "https://statsapi.mlb.com/api/{ver}/game/{gamePk}/feed/live",
                "required_params": ["gamePk"],
                "all_params": ["ver", "gamePk", "timecode", "hydrate", "fields"],
                "notes": None,
            },
            "game_diff": {
                "url": "https://statsapi.mlb.com/api/{ver}/game/{gamePk}/feed/live/diffPatch",
                "required_params": ["gamePk", "startTimecode", "endTimecode"],
                "all_params": ["ver", "gamePk", "startTimecode", "endTimecode"],
                "notes": None,
            },
            "game_timestamps": {
                "url": "https://statsapi.mlb.com/api/{ver}/game/{gamePk}/feed/live/timestamps",
                "required_params": ["gamePk"],
                "all_params": ["ver", "gamePk"],
                "notes": None,
            },
            "game_changes": {
                "url": "https://statsapi.mlb.com/api/{ver}/game/changes",
                "required_params": ["updatedSince"],
                "all_params": [
                    "ver",
                    "updatedSince",
                    "sportId",
                    "gameType",
                    "season",
                    "fields",
                ],
                "notes": None,
            },
            "game_contextMetrics": {
                "url": "https://statsapi.mlb.com/api/{ver}/game/{gamePk}/contextMetrics",
                "required_params": ["gamePk"],
                "all_params": ["ver", "gamePk", "timecode", "fields"],
                "notes": None,
            },
            "game_winProbability": {
                "url": "https://statsapi.mlb.com/api/{ver}/game/{gamePk}/winProbability",
                "required_params": ["gamePk"],
                "all_params": ["ver", "gamePk", "timecode", "fields"],
                "notes": (
                    "If you only want the current win probability for each team, try "
                    "the game_contextMetrics endpoint instead."
                ),
            },
            "game_boxscore": {
                "url": "https://statsapi.mlb.com/api/{ver}/game/{gamePk}/boxscore",
                "required_params": ["gamePk"],
                "all_params": ["ver", "gamePk", "timecode", "fields"],
                "notes": None,
            },
            "game_content": {
                "url": "https://statsapi.mlb.com/api/{ver}/game/{gamePk}/content",
                "required_params": ["gamePk"],
                "all_params": ["ver", "gamePk", "highlightLimit"],
                "notes": None,
            },
            "game_linescore": {
                "url": "https://statsapi.mlb.com/api/{ver}/game/{gamePk}/linescore",
                "required_params": ["gamePk"],
                "all_params": ["ver", "gamePk", "timecode", "fields"],
                "notes": None,
            },
            "game_playByPlay": {
                "url": "https://statsapi.mlb.com/api/{ver}/game/{gamePk}/playByPlay",
                "required_params": ["gamePk"],
                "all_params": ["ver", "gamePk", "timecode", "fields"],
                "notes": None,
            },
            "people": {
                "url": "https://statsapi.mlb.com/api/{ver}/people",
                "required_params": ["personIds"],
                "all_params": ["ver", "personIds", "hydrate", "fields"],
                "notes": None,
            },
            "person": {
                "url": "https://statsapi.mlb.com/api/{ver}/people/{personId}",
                "required_params": ["personId"],
                "all_params": ["ver", "personId", "hydrate", "fields"],
                "notes": None,
            },
            "person_stats": {
                "url": "https://statsapi.mlb.com/api/{ver}/people/{personId}/stats/game/{gamePk}",
                "required_params": ["personId", "gamePk"],
                "all_params": ["ver", "personId", "gamePk", "fields"],
                "notes": (
                    "Specify 'current' instead of a gamePk for a player's current game stats."
                ),
            },
            "schedule": {
                "url": "https://statsapi.mlb.com/api/{ver}/schedule",
                "required_params": ["sportId or gamePk or gamePks"],
                "all_params": [
                    "ver",
                    "scheduleType",
                    "eventTypes",
                    "hydrate",
                    "teamId",
                    "leagueId",
                    "sportId",
                    "gamePk",
                    "gamePks",
                    "venueIds",
                    "gameTypes",
                    "date",
                    "startDate",
                    "endDate",
                    "opponentId",
                    "fields",
                    "season",
                ],
                "notes": None,
            },
            "standings": {
                "url": "https://statsapi.mlb.com/api/{ver}/standings",
                "required_params": ["leagueId"],
                "all_params": [
                    "ver",
                    "leagueId",
                    "season",
                    "standingsTypes",
                    "date",
                    "hydrate",
                    "fields",
                ],
                "notes": None,
            },
            "stats": {
                "url": "https://statsapi.mlb.com/api/{ver}/stats",
                "required_params": ["stats", "group"],
                "all_params": [
                    "ver",
                    "stats",
                    "playerPool",
                    "position",
                    "teamId",
                    "leagueId",
                    "limit",
                    "offset",
                    "group",
                    "gameType",
                    "season",
                    "sportIds",
                    "sortStat",
                    "order",
                    "hydrate",
                    "fields",
                    "personId",
                    "metrics",
                    "startDate",
                    "endDate",
                ],
                "notes": ("If no limit is specified, the response will be limited to 50 records."),
            },
            "stats_leaders": {
                "url": "https://statsapi.mlb.com/api/{ver}/stats/leaders",
                "required_params": ["leaderCategories"],
                "all_params": [
                    "ver",
                    "leaderCategories",
                    "playerPool",
                    "leaderGameTypes",
                    "statGroup",
                    "season",
                    "leagueId",
                    "sportId",
                    "hydrate",
                    "limit",
                    "fields",
                    "statType",
                ],
                "notes": (
                    "If excluding season parameter to get all time leaders, include "
                    "statType=statsSingleSeason or you will likely not get any "
                    "results."
                ),
            },
            "teams": {
                "url": "https://statsapi.mlb.com/api/{ver}/teams",
                "required_params": [],
                "all_params": [
                    "ver",
                    "season",
                    "activeStatus",
                    "leagueIds",
                    "sportId",
                    "sportIds",
                    "gameType",
                    "hydrate",
                    "fields",
                ],
                "notes": None,
            },
            "team": {
                "url": "https://statsapi.mlb.com/api/{ver}/teams/{teamId}",
                "required_params": ["teamId"],
                "all_params": [
                    "ver",
                    "teamId",
                    "season",
                    "sportId",
                    "hydrate",
                    "fields",
                ],
                "notes": None,
            },
            "team_roster": {
                "url": "https://statsapi.mlb.com/api/{ver}/teams/{teamId}/roster",
                "required_params": ["teamId"],
                "all_params": [
                    "ver",
                    "teamId",
                    "rosterType",
                    "season",
                    "date",
                    "hydrate",
                    "fields",
                ],
                "notes": None,
            },
            "team_leaders": {
                "url": "https://statsapi.mlb.com/api/{ver}/teams/{teamId}/leaders",
                "required_params": ["teamId", "leaderCategories", "season"],
                "all_params": [
                    "ver",
                    "teamId",
                    "leaderCategories",
                    "season",
                    "leaderGameTypes",
                    "hydrate",
                    "limit",
                    "fields",
                ],
                "notes": None,
            },
            "venues": {
                "url": "https://statsapi.mlb.com/api/{ver}/venues",
                "required_params": ["venueIds"],
                "all_params": ["ver", "venueIds", "season", "hydrate", "fields"],
                "notes": None,
            },
        }

        logger.debug(f"Retrieved information for {len(endpoints)} MLB Stats API endpoints")
        return {
            "endpoints": endpoints,
            "usage_note": (
                "Use these endpoints with the get_stats tool by specifying "
                "the endpoint name and required parameters"
            ),
            "example": {"endpoint": "teams", "params": {"sportId": 1}},
        }
    except Exception as e:
        error_msg = f"Error retrieving available endpoints information: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_notes(endpoint: str) -> Dict[str, Any]:
    """
    Retrieve notes for a given MLB Stats API endpoint,
        including required parameters and hints.

    Args:
        endpoint: The API endpoint to get notes for
            (e.g., 'stats', 'schedule', 'standings')

    Returns:
        Dictionary containing notes about the endpoint

    Raises:
        Exception: If there's an error retrieving notes
    """
    try:
        logger.debug(f"Retrieving notes for endpoint: {endpoint}")
        notes_text = statsapi.notes(endpoint)
        logger.debug(f"Retrieved notes for endpoint: {endpoint}")

        # Parse the notes text into a structured dictionary
        result = {
            "endpoint": endpoint,
            "required_params": [],
            "all_params": [],
            "hints": "",
            "path_params": [],
            "query_params": [],
        }

        # Split the notes into lines for processing
        lines = notes_text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "All path parameters:" in line:
                params = line.split(":")[1].strip().strip("[]").replace("'", "").split(", ")
                result["path_params"] = [p for p in params if p]
            elif "All query parameters:" in line:
                params = line.split(":")[1].strip().strip("[]").replace("'", "").split(", ")
                result["query_params"] = [p for p in params if p]
            elif "Required path parameters" in line:
                params = line.split(":")[1].strip().strip("[]").replace("'", "").split(", ")
                result["required_params"].extend([p for p in params if p])
            elif "Required query parameters:" in line:
                params = line.split(":")[1].strip().strip("[]").replace("'", "").split(", ")
                if params and params[0] != "None":
                    result["required_params"].extend([p for p in params if p])
            elif line.startswith("The hydrate function") or line.startswith("Call the endpoint"):
                result["hints"] += line + "\n"

        # Combine path and query params for all_params
        result["all_params"] = result["path_params"] + result["query_params"]
        if not result["all_params"]:
            logger.error(f"No parameters found for endpoint: {endpoint}")
            raise Exception(f"No parameters found for endpoint: {endpoint}")
        return result
    except Exception as e:
        error_msg = f"Error retrieving notes for endpoint {endpoint}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_game_scoring_play_data(game_id: int) -> Dict[str, Any]:
    """
    Retrieve scoring play data for a specific MLB game.

    Args:
        game_id: The MLB game ID to get scoring play data for

    Returns:
        Dictionary containing scoring play data for the game

    Raises:
        Exception: If there's an error retrieving scoring play data
    """
    try:
        logger.debug(f"Retrieving scoring play data for game ID: {game_id}")
        result = statsapi.game_scoring_play_data(game_id)
        if "plays" not in result or not result["plays"]:
            logger.error(f"No plays found for game ID: {game_id}")
            raise Exception(f"No plays found for game ID: {game_id}")
        logger.debug(f"Retrieved scoring play data for game ID: {game_id}")
        return result
    except Exception as e:
        error_msg = f"Error retrieving scoring play data for game ID {game_id}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_last_game(team_id: int) -> Dict[str, Any]:
    """
    Get the game ID for a team's most recent game.

    Args:
        team_id: The MLB team ID to get the last game for

    Returns:
        Dictionary containing game information

    Raises:
        Exception: If there's an error retrieving last game
    """
    try:
        logger.debug(f"Retrieving last game for team ID: {team_id}")
        game_id = statsapi.last_game(team_id)
        logger.debug(f"Retrieved last game ID {game_id} for team ID: {team_id}")

        # Get additional game details
        game_data = statsapi.get("game", {"gamePk": game_id})

        result = {
            "game_id": game_id,
            "team_id": team_id,
            "date": game_data["gameData"]["datetime"]["dateTime"].split("T")[0],
            "status": game_data["gameData"]["status"]["detailedState"],
        }

        return result
    except Exception as e:
        error_msg = f"Error retrieving last game for team ID {team_id}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_league_leader_data(
    leader_categories: str,
    season: Optional[int] = None,
    limit: Optional[int] = None,
    stat_group: Optional[str] = None,
    league_id: Optional[int] = None,
    game_types: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get league leader statistics for specified categories.

    Args:
        leader_categories: Comma-separated list of stat categories
        season: Season year (defaults to current season)
        limit: Number of leaders to return (defaults to all)
        stat_group: Stat group (e.g., 'hitting', 'pitching', 'fielding')
        league_id: MLB league ID to filter by

    Returns:
        Dictionary containing leader data

    Raises:
        Exception: If there's an error retrieving league leader data
    """
    try:
        logger.debug(f"Retrieving league leader data for categories: {leader_categories}")

        # Build parameters dictionary
        params = {
            "leaderCategories": leader_categories,
        }

        if season is not None:
            params["season"] = season
        if limit is not None:
            params["limit"] = limit
        if stat_group is not None:
            params["statGroup"] = stat_group
        if league_id is not None:
            params["leagueId"] = league_id
        if game_types is not None:
            params["gameTypes"] = game_types

        # Get leader data
        leader_data = statsapi.league_leader_data(**params)

        result = {
            "leaders": leader_data,
            "season": season if season is not None else "current",
            "categories": leader_categories.split(","),
        }

        logger.debug(f"Retrieved league leader data for {len(leader_data)} categories")
        return result
    except Exception as e:
        error_msg = f"Error retrieving league leader data: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_linescore(game_id: int) -> Dict[str, Any]:
    """
    Get formatted linescore data for a specific MLB game.

    Args:
        game_id: The MLB game ID to get linescore data for

    Returns:
        Dictionary containing linescore data

    Raises:
        Exception: If there's an error retrieving linescore
    """
    try:
        logger.debug(f"Retrieving linescore for game ID: {game_id}")
        linescore_text = statsapi.linescore(game_id)
        logger.debug(f"Retrieved linescore for game ID: {game_id}")

        # Get additional game details for structured data
        game_data = statsapi.get("game", {"gamePk": game_id})

        # Extract team information
        teams = {
            "away": {
                "id": game_data["gameData"]["teams"]["away"]["id"],
                "name": game_data["gameData"]["teams"]["away"]["name"],
                "abbreviation": game_data["gameData"]["teams"]["away"]["abbreviation"],
            },
            "home": {
                "id": game_data["gameData"]["teams"]["home"]["id"],
                "name": game_data["gameData"]["teams"]["home"]["name"],
                "abbreviation": game_data["gameData"]["teams"]["home"]["abbreviation"],
            },
        }

        # Extract inning scores
        innings = []
        for inning in game_data["liveData"]["linescore"]["innings"]:
            inning_data = {
                "num": inning["num"],
                "away": inning.get("away", {}).get("runs", 0),
                "home": inning.get("home", {}).get("runs", 0),
            }
            innings.append(inning_data)

        # Extract totals
        totals = {
            "runs": {
                "away": game_data["liveData"]["linescore"]["teams"]["away"]["runs"],
                "home": game_data["liveData"]["linescore"]["teams"]["home"]["runs"],
            },
            "hits": {
                "away": game_data["liveData"]["linescore"]["teams"]["away"]["hits"],
                "home": game_data["liveData"]["linescore"]["teams"]["home"]["hits"],
            },
            "errors": {
                "away": game_data["liveData"]["linescore"]["teams"]["away"]["errors"],
                "home": game_data["liveData"]["linescore"]["teams"]["home"]["errors"],
            },
        }

        result = {
            "linescore": linescore_text,
            "game_id": game_id,
            "teams": teams,
            "innings": innings,
            "totals": totals,
        }

        return result
    except Exception as e:
        error_msg = f"Error retrieving linescore for game ID {game_id}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_next_game(team_id: int) -> Dict[str, Any]:
    """
    Get the game ID for a team's next scheduled game.

    Args:
        team_id: The MLB team ID to get the next game for

    Returns:
        Dictionary containing game information

    Raises:
        Exception: If there's an error retrieving next game
    """
    try:
        logger.debug(f"Retrieving next game for team ID: {team_id}")
        game_id = statsapi.next_game(team_id)
        logger.debug(f"Retrieved next game ID {game_id} for team ID: {team_id}")

        # Get additional game details
        game_data = statsapi.get("game", {"gamePk": game_id})

        # Determine opponent team
        home_team = game_data["gameData"]["teams"]["home"]
        away_team = game_data["gameData"]["teams"]["away"]
        opponent = home_team if home_team["id"] != team_id else away_team

        result = {
            "game_id": game_id,
            "team_id": team_id,
            "date": game_data["gameData"]["datetime"]["dateTime"].split("T")[0],
            "opponent": {
                "id": opponent["id"],
                "name": opponent["name"],
                "abbreviation": opponent["abbreviation"],
            },
            "status": game_data["gameData"]["status"]["detailedState"],
            "is_home": home_team["id"] == team_id,
        }

        return result
    except Exception as e:
        error_msg = f"Error retrieving next game for team ID {team_id}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def get_game_highlight_data(game_id: int) -> Dict[str, Any]:
    """
    Get highlight data for a specific MLB game.

    Args:
        game_id: The MLB game ID to get highlight data for

    Returns:
        Dictionary containing highlight data for the game

    Raises:
        Exception: If there's an error retrieving highlight data
    """
    try:
        logger.debug(f"Retrieving highlight data for game ID: {game_id}")
        highlight_data = statsapi.game_highlight_data(game_id)
        logger.debug(f"Retrieved highlight data for game ID: {game_id}")
        return {"data": highlight_data}
    except Exception as e:
        error_msg = f"Error retrieving highlight data for game ID {game_id}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
