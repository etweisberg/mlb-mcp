"""
Tests for MLB Stats API tools in the baseball project.
"""

import json
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from mlb_stats_mcp.utils.logging_config import setup_logging

logger = setup_logging("[TEST] mlbstatsapi")


def simplify_session_setup():
    """Helper to create server params for tests."""
    server_path = Path(__file__).parent.parent / "server.py"
    return StdioServerParameters(command="python", args=[str(server_path)], env=None)


@pytest.mark.asyncio
async def test_get_schedule_tool():
    """Test the get_schedule tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_schedule",
                {
                    "start_date": "07/01/2018",
                    "end_date": "07/31/2018",
                    "team_id": 143,
                    "opponent_id": 121,
                },
            )

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)["games"]
            logger.debug(data)
            # Verify we got all 4 games
            assert len(data) == 4, f"Expected 4 games, got {len(data)}"

            # Verify all games are between the correct teams
            for game in data:
                assert game["away_id"] == 143 or game["home_id"] == 143, (
                    "Phillies (143) should be in every game"
                )
                assert game["away_id"] == 121 or game["home_id"] == 121, (
                    "Mets (121) should be in every game"
                )

            # Verify game dates are within the requested range
            game_dates = [game["game_date"] for game in data]
            assert all("2018-07" in date for date in game_dates), "All games should be in July 2018"

            # Verify required fields exist in each game
            required_fields = [
                "game_id",
                "game_date",
                "away_name",
                "home_name",
                "status",
            ]
            for game in data:
                for field in required_fields:
                    assert field in game, f"Missing '{field}' in game {game.get('game_id')}"

            # Verify specific games from the logs
            game_ids = [game["game_id"] for game in data]
            expected_game_ids = [530769, 529466, 530781, 530796]
            assert set(game_ids) == set(expected_game_ids), (
                f"Expected game IDs {expected_game_ids}, got {game_ids}"
            )

            # Verify the doubleheader games
            doubleheader_games = [game for game in data if game["doubleheader"] == "Y"]
            assert len(doubleheader_games) == 2, "Should have 2 doubleheader games"
            assert all(game["game_date"] == "2018-07-09" for game in doubleheader_games), (
                "Doubleheader games should be on 2018-07-09"
            )

            # Verify all games have final status
            assert all(game["status"] == "Final" for game in data), (
                "All games should have 'Final' status"
            )

            # Test with invalid date
            result = await session.call_tool("get_schedule", {"date": "invalid_date"})
            assert result.isError, "Expected error response for invalid date"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_schedule" in error_text, (
                "Error message should mention get_schedule"
            )


@pytest.mark.asyncio
async def test_lookup_player_tool():
    """Test the lookup_player tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid player name
            result = await session.call_tool("lookup_player", {"name": "Aaron Judge"})

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            assert "people" in data, "Response should contain 'people' key"
            assert len(data["people"]) > 0, "Player data should not be empty"

            # Test with invalid player name
            result = await session.call_tool("lookup_player", {"name": "Invalid Player Name 123"})
            print(result)
            assert result.isError, "Expected error response for invalid player name"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in lookup_player" in error_text, (
                "Error message should mention lookup_player"
            )


@pytest.mark.asyncio
async def test_get_standings_tool():
    """Test the get_standings tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_standings",
                {"season": 2023, "standings_types": "regularSeason"},
            )

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            assert "200" in data and "div_name" in data["200"], (
                "Missing expected division structure"
            )

            # Test with invalid standings type
            result = await session.call_tool(
                "get_standings",
                {"standings_types": "invalid_type"},
            )
            print(result)
            assert result.isError, "Expected error response for invalid standings type"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_standings" in error_text, (
                "Error message should mention get_standings"
            )


@pytest.mark.asyncio
async def test_get_team_leaders_tool():
    """Test the get_team_leaders tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Check if the tool is available
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]

            if "get_team_leaders" not in tool_names:
                pytest.skip("get_team_leaders tool not available")

            # Test with valid parameters
            result = await session.call_tool(
                "get_team_leaders",
                {
                    "team_id": 147,
                    "season": 2023,
                    "leader_category": "walks",
                    "limit": 10,
                },
            )

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            assert "teamLeaders" in data or "results" in data, (
                "Missing team leader data in response"
            )

            # Test with invalid team ID
            result = await session.call_tool(
                "get_team_leaders",
                {
                    "team_id": 999999,
                    "leader_category": "walks",
                },
            )
            assert result.isError, "Expected error response for invalid team ID"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_team_leaders" in error_text, (
                "Error message should mention get_team_leaders"
            )


@pytest.mark.asyncio
async def test_get_stats_tool():
    """Test the get_stats tool which provides access to any MLB Stats API endpoint."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_stats",
                {"endpoint": "teams", "params": {"sportId": 1}},
            )

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            assert "teams" in data, "Missing 'teams' in response"

            # Test with invalid endpoint
            result = await session.call_tool(
                "get_stats",
                {"endpoint": "invalid_endpoint", "params": {}},
            )
            assert result.isError, "Expected error response for invalid endpoint"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_stats" in error_text, "Error message should mention get_stats"


@pytest.mark.asyncio
async def test_get_player_stats_tool():
    """Test the get_player_stats tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_player_stats",
                {
                    "player_id": 592450,  # Aaron Judge
                    "group": "hitting",
                    "season": 2023,
                    "stats": "season",
                },
            )

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            assert "stats" in data, "Missing 'stats' in response"

            # Test with invalid player ID
            result = await session.call_tool(
                "get_player_stats",
                {
                    "player_id": 999999,
                    "group": "hitting",
                    "stats": "season",
                },
            )
            assert result.isError, "Expected error response for invalid player ID"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_player_stats" in error_text, (
                "Error message should mention get_player_stats"
            )


@pytest.mark.asyncio
async def test_get_boxscore_tool():
    """Test the get_boxscore tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid game ID
            result = await session.call_tool("get_boxscore", {"game_id": 565997})

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            common_keys = ["game_id", "boxscore", "success"]
            assert any(key in data for key in common_keys), (
                f"Response missing expected boxscore structure, expected one of: {common_keys}"
            )

            # Test with invalid game ID
            result = await session.call_tool("get_boxscore", {"game_id": 999999})
            assert result.isError, "Expected error response for invalid game ID"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_boxscore" in error_text, (
                "Error message should mention get_boxscore"
            )


@pytest.mark.asyncio
async def test_get_game_pace_tool():
    """Test the get_game_pace tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool("get_game_pace", {"season": 2023})

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            expected_keys = ["copyright", "sports", "teams", "leagues"]
            assert any(key in data for key in expected_keys), (
                f"Response missing expected data structure, expected one of: {expected_keys}"
            )

            # Test with invalid season
            result = await session.call_tool("get_game_pace", {"season": 9999})
            assert result.isError, "Expected error response for invalid season"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_game_pace" in error_text, (
                "Error message should mention get_game_pace"
            )


@pytest.mark.asyncio
async def test_get_meta_tool():
    """Test the get_meta tool for accessing MLB Stats API metadata."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Check if the tool is available
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]

            if "get_meta" not in tool_names:
                pytest.skip("get_meta tool not available")

            # Test with valid parameters
            result = await session.call_tool("get_meta", {"type_name": "positions"})

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            assert len(data) > 0, "Metadata results should not be empty"

            # Test with invalid type name
            result = await session.call_tool("get_meta", {"type_name": "invalid_type"})
            assert result.isError, "Expected error response for invalid type name"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_meta" in error_text, "Error message should mention get_meta"


@pytest.mark.asyncio
async def test_get_available_endpoints_tool():
    """Test the get_available_endpoints tool for retrieving MLB Stats API endpoint information."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Check if the tool is available
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]

            if "get_available_endpoints" not in tool_names:
                pytest.skip("get_available_endpoints tool not available")

            # Test the tool
            result = await session.call_tool("get_available_endpoints", {})

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            assert "endpoints" in data, "Response missing 'endpoints' dictionary"
            assert "usage_note" in data, "Response missing 'usage_note' field"
            assert "example" in data, "Response missing usage example"


@pytest.mark.asyncio
async def test_get_notes_tool():
    """Test the get_notes tool for retrieving notes about MLB Stats API endpoints."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Check if the tool is available
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]

            if "get_notes" not in tool_names:
                pytest.skip("get_notes tool not available")

            # Test with valid endpoint
            result = await session.call_tool("get_notes", {"endpoint": "teams"})

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            expected_fields = [
                "endpoint",
                "required_params",
                "all_params",
                "hints",
                "path_params",
                "query_params",
            ]
            for field in expected_fields:
                assert field in data, f"Notes missing expected field: {field}"

            # Test with invalid endpoint
            result = await session.call_tool("get_notes", {"endpoint": "invalid_endpoint"})
            assert result.isError, "Expected error response for invalid endpoint"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_notes" in error_text, "Error message should mention get_notes"


@pytest.mark.asyncio
async def test_get_game_scoring_play_data_tool():
    """Test the get_game_scoring_play_data tool for retrieving scoring play data."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Check if the tool is available
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]

            if "get_game_scoring_play_data" not in tool_names:
                pytest.skip("get_game_scoring_play_data tool not available")

            # Test with valid game ID
            result = await session.call_tool("get_game_scoring_play_data", {"game_id": 565997})

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            expected_fields = ["away", "home", "plays"]
            for field in expected_fields:
                assert field in data, f"Scoring play data missing expected field: {field}"

            # Test with invalid game ID
            result = await session.call_tool("get_game_scoring_play_data", {"game_id": 999999999})
            assert result.isError, "Expected error response for invalid game ID"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_game_scoring_play_data" in error_text, (
                "Error message should mention get_game_scoring_play_data"
            )


@pytest.mark.asyncio
async def test_get_last_game_tool():
    """Test the get_last_game tool for retrieving a team's most recent game ID."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Check if the tool is available
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]

            if "get_last_game" not in tool_names:
                pytest.skip("get_last_game tool not available")

            # Test with valid team ID
            result = await session.call_tool("get_last_game", {"team_id": 147})

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            expected_fields = ["game_id", "team_id", "date", "status"]
            for field in expected_fields:
                assert field in data, f"Last game data missing expected field: {field}"

            # Test with invalid team ID
            result = await session.call_tool("get_last_game", {"team_id": 999999999})
            assert result.isError, "Expected error response for invalid team ID"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_last_game" in error_text, (
                "Error message should mention get_last_game"
            )


@pytest.mark.asyncio
async def test_get_league_leader_data_tool():
    """Test the get_league_leader_data tool for retrieving league leader statistics."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Check if the tool is available
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]

            if "get_league_leader_data" not in tool_names:
                pytest.skip("get_league_leader_data tool not available")

            # Test with valid parameters
            result = await session.call_tool(
                "get_league_leader_data",
                {
                    "leader_categories": "homeRuns,strikeouts",
                    "limit": 5,
                    "stat_group": "hitting",
                },
            )

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            expected_fields = ["leaders", "season", "categories"]
            for field in expected_fields:
                assert field in data, f"League leader data missing expected field: {field}"

            # Test with invalid parameters
            result = await session.call_tool(
                "get_league_leader_data",
                {
                    "leader_categories": "invalidCategory",
                    "limit": 5,
                },
            )
            assert result.isError, "Expected error response for invalid parameters"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_league_leader_data" in error_text, (
                "Error message should mention get_league_leader_data"
            )


@pytest.mark.asyncio
async def test_get_linescore_tool():
    """Test the get_linescore tool for retrieving game linescore data."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Check if the tool is available
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]

            if "get_linescore" not in tool_names:
                pytest.skip("get_linescore tool not available")

            # Test with valid game ID
            result = await session.call_tool("get_linescore", {"game_id": 565997})

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            expected_fields = ["linescore", "game_id", "teams", "innings", "totals"]
            for field in expected_fields:
                assert field in data, f"Linescore data missing expected field: {field}"

            # Test with invalid game ID
            result = await session.call_tool("get_linescore", {"game_id": 999999999})
            assert result.isError, "Expected error response for invalid game ID"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_linescore" in error_text, (
                "Error message should mention get_linescore"
            )


@pytest.mark.asyncio
async def test_get_next_game_tool():
    """Test the get_next_game tool for retrieving a team's next game ID."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Check if the tool is available
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]

            if "get_next_game" not in tool_names:
                pytest.skip("get_next_game tool not available")

            # Test with valid team ID
            result = await session.call_tool("get_next_game", {"team_id": 147})

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            expected_fields = [
                "game_id",
                "team_id",
                "date",
                "opponent",
                "status",
                "is_home",
            ]
            for field in expected_fields:
                assert field in data, f"Next game data missing expected field: {field}"

            # Test with invalid team ID
            result = await session.call_tool("get_next_game", {"team_id": 999999999})
            assert result.isError, "Expected error response for invalid team ID"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_next_game" in error_text, (
                "Error message should mention get_next_game"
            )


@pytest.mark.asyncio
async def test_get_team_roster_tool():
    """Test the get_team_roster tool for retrieving team roster information."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Check if the tool is available
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]

            if "get_team_roster" not in tool_names:
                pytest.skip("get_team_roster tool not available")

            # Test with valid parameters
            result = await session.call_tool(
                "get_team_roster",
                {
                    "team_id": 147,
                    "roster_type": "active",
                    "season": 2023,
                },
            )

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify roster text format
            roster_text = result.content[0].text
            assert roster_text, "Roster text should not be empty"

            # Test with invalid team ID
            result = await session.call_tool(
                "get_team_roster",
                {
                    "team_id": 999999999,
                    "roster_type": "active",
                },
            )
            assert result.isError, "Expected error response for invalid team ID"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_team_roster" in error_text, (
                "Error message should mention get_team_roster"
            )


@pytest.mark.asyncio
async def test_get_game_highlight_data_tool():
    """Test the get_game_highlight_data tool for retrieving game highlight data."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Check if the tool is available
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]

            if "get_game_highlight_data" not in tool_names:
                pytest.skip("get_game_highlight_data tool not available")

            # Test with valid game ID
            result = await session.call_tool("get_game_highlight_data", {"game_id": 565997})

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            assert "data" in data, "Response missing 'data' field"
            highlight_data = data["data"]

            # If there are highlights, verify their structure
            if highlight_data:
                assert isinstance(highlight_data, list), "Highlight data should be a list"
                first_highlight = highlight_data[0]
                expected_fields = [
                    "date",
                    "type",
                    "headline",
                    "description",
                    "duration",
                    "playbacks",
                    "title",
                ]
                for field in expected_fields:
                    assert field in first_highlight, f"Highlight missing expected field: {field}"

            # Test with invalid game ID
            result = await session.call_tool("get_game_highlight_data", {"game_id": 999999999})
            assert result.isError, "Expected error response for invalid game ID"
            assert result.content, "No error content returned"
            error_text = result.content[0].text
            assert "Error in get_game_highlight_data" in error_text, (
                "Error message should mention get_game_highlight_data"
            )
