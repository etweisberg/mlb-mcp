"""
Tests for pybaseball tools.
"""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from mlb_stats_mcp.utils.images import display_base64_image
from mlb_stats_mcp.utils.logging_config import setup_logging

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

logger = setup_logging("[TEST] pybaseball")

logger.debug(f"SHOW_IMAGE SET TO {os.environ.get('SHOW_IMAGE')}")


def simplify_session_setup():
    """Helper to create server params for tests."""
    server_path = Path(__file__).parent.parent / "server.py"
    return StdioServerParameters(command="python", args=[str(server_path)], env=None)


@pytest.mark.asyncio
async def test_get_statcast_batter_exitvelo_barrels():
    """Test the get_statcast_batter_exitvelo_barrels tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_statcast_batter_exitvelo_barrels", {"year": 2023, "minBBE": 50}
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_statcast_pitcher_exitvelo_barrels():
    """Test the get_statcast_pitcher_exitvelo_barrels tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_statcast_pitcher_exitvelo_barrels", {"year": 2023, "minBBE": 50}
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_statcast_batter_expected_stats():
    """Test the get_statcast_batter_expected_stats tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_statcast_batter_expected_stats", {"year": 2023, "minPA": 50}
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_statcast_pitcher_expected_stats():
    """Test the get_statcast_pitcher_expected_stats tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_statcast_pitcher_expected_stats", {"year": 2023, "minPA": 50}
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_statcast_batter_percentile_ranks():
    """Test the get_statcast_batter_percentile_ranks tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool("get_statcast_batter_percentile_ranks", {"year": 2023})

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_statcast_pitcher_percentile_ranks():
    """Test the get_statcast_pitcher_percentile_ranks tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_statcast_pitcher_percentile_ranks", {"year": 2023}
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_statcast_batter_pitch_arsenal():
    """Test the get_statcast_batter_pitch_arsenal tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_statcast_batter_pitch_arsenal", {"year": 2023, "minPA": 50}
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_statcast_pitcher_pitch_arsenal():
    """Test the get_statcast_pitcher_pitch_arsenal tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_statcast_pitcher_pitch_arsenal",
                {"year": 2023, "minP": 50, "arsenal_type": "avg_speed"},
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"

            # Test with different arsenal type
            result = await session.call_tool(
                "get_statcast_pitcher_pitch_arsenal",
                {"year": 2023, "minP": 50, "arsenal_type": "avg_spin"},
            )
            # This should succeed or hit length limit, but not a tool error
            assert not result.isError, "Expected successful response or length limit handling"

            # Test with invalid arsenal type
            result = await session.call_tool(
                "get_statcast_pitcher_pitch_arsenal",
                {"year": 2023, "minP": 50, "arsenal_type": "invalid_type"},
            )
            assert result.isError, "Expected error for invalid arsenal type"


@pytest.mark.asyncio
async def test_get_statcast_single_game():
    """Test the get_statcast_single_game tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters - using a known game ID
            result = await session.call_tool(
                "get_statcast_single_game",
                {"game_pk": 717953},  # Example game from 2023
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"

            # Test with invalid game ID
            result = await session.call_tool("get_statcast_single_game", {"game_pk": 999999999})
            assert result.isError, "Expected error response for invalid game ID"


@pytest.mark.asyncio
async def test_image_create_spraychart_altuve():
    """Test the create_spraychart_plot tool following the Altuve example."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            try:
                # Get statcast batter data for Jose Altuve - using end_row to limit response size
                batter_result = await session.call_tool(
                    "get_statcast_batter_data",
                    {
                        "player_id": 514888,
                        "start_dt": "2019-05-01",
                        "end_dt": "2019-07-01",
                        "end_row": 100,  # Limit to first 100 rows
                    },
                )

                assert not batter_result.isError
                batter_data = json.loads(batter_result.content[0].text)

                # Check if we got successful data or length limit error
                if "error" in batter_data:
                    # If still too long even with end_row, use smaller limit
                    batter_result = await session.call_tool(
                        "get_statcast_batter_data",
                        {
                            "player_id": 514888,
                            "start_dt": "2019-05-01",
                            "end_dt": "2019-07-01",
                            "end_row": 50,  # Even smaller limit
                        },
                    )
                    batter_data = json.loads(batter_result.content[0].text)

                assert "data" in batter_data
                assert len(batter_data["data"]) > 0
            except Exception as e:
                raise Exception(f"Exception occured in pre-req batter data: {e}") from e

            # Create spraychart
            try:
                spraychart_result = await session.call_tool(
                    "create_spraychart_plot",
                    {
                        "data": batter_data,
                        "team_stadium": "astros",
                        "title": "Jose Altuve: May-June 2019",
                        "colorby": "events",
                        "size": 120,
                        "width": 1024,
                        "height": 1024,
                    },
                )
            except Exception as e:
                raise Exception(f"Exception occured in spraychart tool: {e}") from e

            assert not spraychart_result.isError
            result_json = json.loads(spraychart_result.content[0].text)

            # Verify response structure
            assert result_json["plot_type"] == "spraychart"
            assert "image_base64" in result_json
            assert len(result_json["image_base64"]) > 100
            assert result_json["hit_count"] > 0
            assert result_json["stadium"] == "astros"
            assert result_json["title"] == "Jose Altuve: May-June 2019"
            assert "metadata" in result_json
            assert result_json["metadata"]["colorby"] == "events"
            assert isinstance(result_json["metadata"]["events"], dict)

            # Display the image if SHOW_IMAGE environment variable is set to true
            if os.environ.get("SHOW_IMAGE", "false").lower() == "true":
                display_base64_image(result_json["image_base64"])


@pytest.mark.asyncio
async def test_image_create_spraychart_plot_votto_aquino():
    """Test spraychart with Joey Votto vs. Aristedes Aquino data."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            try:
                # Get statcast batter data for Joey Votto (458015)
                votto_result = await session.call_tool(
                    "get_statcast_batter_data",
                    {
                        "player_id": 458015,
                        "start_dt": "2019-08-01",
                        "end_dt": "2019-10-01",
                        "end_row": 50,
                    },
                )

                assert not votto_result.isError
                votto_data = json.loads(votto_result.content[0].text)

                # Get statcast batter data for Aristedes Aquino (606157)
                aquino_result = await session.call_tool(
                    "get_statcast_batter_data",
                    {
                        "player_id": 606157,
                        "start_dt": "2019-08-01",
                        "end_dt": "2019-10-01",
                        "end_row": 50,
                    },
                )

                assert not aquino_result.isError
                aquino_data = json.loads(aquino_result.content[0].text)

                # Combine the data (concatenate the data arrays)
                combined_data = {"data": votto_data["data"] + aquino_data["data"]}

                assert len(combined_data["data"]) > 0
            except Exception as e:
                raise Exception(f"Exception occurred in pre-req batter data: {e}") from e

            # Create spraychart
            try:
                spraychart_result = await session.call_tool(
                    "create_spraychart_plot",
                    {
                        "data": combined_data,
                        "team_stadium": "reds",
                        "title": "Joey Votto vs. Aristedes Aquino",
                        "colorby": "player_name",  # Color by player
                        "size": 120,
                        "width": 1024,
                        "height": 1024,
                    },
                )
            except Exception as e:
                raise Exception(f"Exception occurred in spraychart tool: {e}") from e

            assert not spraychart_result.isError
            result_json = json.loads(spraychart_result.content[0].text)

            # Verify response structure
            assert result_json["plot_type"] == "spraychart"
            assert "image_base64" in result_json
            assert len(result_json["image_base64"]) > 100
            assert result_json["hit_count"] > 0
            assert result_json["stadium"] == "reds"
            assert result_json["title"] == "Joey Votto vs. Aristedes Aquino"
            assert "metadata" in result_json
            assert result_json["metadata"]["colorby"] == "player_name"
            assert isinstance(result_json["metadata"]["events"], dict)

            # Display the image if SHOW_IMAGE environment variable is set to true
            if os.environ.get("SHOW_IMAGE", "false").lower() == "true":
                display_base64_image(result_json["image_base64"])


@pytest.mark.asyncio
async def test_image_create_bb_profile_plot():
    """Test bb_profile plot recreating the example logic."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            try:
                # Get statcast data for specific date range - using smaller end_row to avoid length limit
                statcast_result = await session.call_tool(
                    "get_statcast_data",
                    {
                        "start_dt": "2018-05-01",
                        "end_dt": "2018-05-04",
                        "verbose": True,
                        "parallel": True,
                        "end_row": 100,  # Smaller limit to avoid length issue
                    },
                )

                assert not statcast_result.isError
                statcast_data = json.loads(statcast_result.content[0].text)

                # Check if we got successful data or still hit length limit
                if "error" in statcast_data:
                    # If still too long, use even smaller limit
                    statcast_result = await session.call_tool(
                        "get_statcast_data",
                        {
                            "start_dt": "2018-05-01",
                            "end_dt": "2018-05-04",
                            "verbose": True,
                            "parallel": True,
                            "end_row": 50,  # Even smaller limit
                        },
                    )
                    statcast_data = json.loads(statcast_result.content[0].text)

                assert "data" in statcast_data
                assert len(statcast_data["data"]) > 0
            except Exception as e:
                raise Exception(f"Exception occurred in pre-req statcast data: {e}") from e

            # Create bb_profile plot
            try:
                bb_profile_result = await session.call_tool(
                    "create_bb_profile_plot",
                    {
                        "data": statcast_data,
                        "parameter": "launch_angle",
                    },
                )
            except Exception as e:
                raise Exception(f"Exception occurred in bb_profile tool: {e}") from e

            assert not bb_profile_result.isError
            result_json = json.loads(bb_profile_result.content[0].text)

            # Verify response structure
            assert result_json["plot_type"] == "bb_profile"
            assert "image_base64" in result_json
            assert len(result_json["image_base64"]) > 100
            assert result_json["bb_count"] > 0
            assert result_json["parameter"] == "launch_angle"
            assert "metadata" in result_json
            assert isinstance(result_json["metadata"]["bb_types"], dict)

            # Display the image if SHOW_IMAGE environment variable is set to true
            if os.environ.get("SHOW_IMAGE", "false").lower() == "true":
                display_base64_image(result_json["image_base64"])


@pytest.mark.asyncio
async def test_image_plot_teams():
    """Test plotting teams based on team batting data."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Get team batting data for 2023
            result = await session.call_tool(
                "get_team_batting",
                {"start_season": 2023, "league": "all"},
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"

            # Create plot_teams visualization
            plot_result = await session.call_tool(
                "create_teams_plot",
                {
                    "data": data,
                    "x_axis": "HR",
                    "y_axis": "BB",
                    "title": "Team HR vs BB (2023)",
                },
            )

            # Verify successful plot response
            assert not plot_result.isError, "Expected successful plot response"
            assert plot_result.content, "No content returned from plot tool"
            assert plot_result.content[0].type == "text", "Expected text response"

            # Verify plot response structure
            plot_data = json.loads(plot_result.content[0].text)
            assert "plot_type" in plot_data, "Response should contain 'plot_type' key"
            assert plot_data["plot_type"] == "teams"
            assert "image_base64" in plot_data, "Response should contain 'image_base64' key"
            assert len(plot_data["image_base64"]) > 100, "Image data should be substantial"
            assert "team_count" in plot_data, "Response should contain 'team_count' key"
            assert plot_data["team_count"] > 0, "Should have team data"
            assert "x_axis" in plot_data, "Response should contain 'x_axis' key"
            assert plot_data["x_axis"] == "HR"
            assert "y_axis" in plot_data, "Response should contain 'y_axis' key"
            assert plot_data["y_axis"] == "BB"

            # Display the image if SHOW_IMAGE environment variable is set to true
            if os.environ.get("SHOW_IMAGE", "false").lower() == "true":
                display_base64_image(plot_data["image_base64"])


@pytest.mark.asyncio
async def test_get_pitching_stats_bref():
    """Test the get_pitching_stats_bref tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool("get_pitching_stats_bref", {"season": 2023})

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"
            assert data["count"] > 0, "Should have pitching data"


@pytest.mark.asyncio
async def test_get_pitching_stats_range():
    """Test the get_pitching_stats_range tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_pitching_stats_range",
                {"start_dt": "2023-04-01", "end_dt": "2023-04-07"},
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_pitching_stats():
    """Test the get_pitching_stats tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_pitching_stats",
                {"start_season": 2023, "qual": 50},
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_schedule_and_record():
    """Test the get_schedule_and_record tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_schedule_and_record",
                {"season": 2023, "team": "LAD"},
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_player_splits():
    """Test the get_player_splits tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters - Mike Trout's Baseball Reference ID
            result = await session.call_tool(
                "get_player_splits",
                {"playerid": "troutmi01", "year": 2023},
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_pybaseball_standings():
    """Test the get_pybaseball_standings tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_pybaseball_standings",
                {"season": 2023},
            )

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify response structure
            data = json.loads(result.content[0].text)
            assert "data" in data, "Response should contain 'data' key"


@pytest.mark.asyncio
async def test_get_team_batting():
    """Test the get_team_batting tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_team_batting",
                {"start_season": 2023, "league": "all"},
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_team_fielding():
    """Test the get_team_fielding tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_team_fielding",
                {"start_season": 2023, "league": "all"},
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_team_pitching():
    """Test the get_team_pitching tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_team_pitching",
                {"start_season": 2023, "league": "all"},
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_get_top_prospects():
    """Test the get_top_prospects tool."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with valid parameters
            result = await session.call_tool(
                "get_top_prospects",
                {"team": "angels", "player_type": "batters"},
            )

            # Verify successful response or proper length limit handling
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation structure
                assert "length" in data, "Length limit response should contain 'length' key"
                assert "limit" in data, "Length limit response should contain 'limit' key"
                assert "total_rows" in data, "Length limit response should contain 'total_rows' key"
            else:
                # Verify successful response structure
                assert "data" in data, "Response should contain 'data' key"
                assert "count" in data, "Response should contain 'count' key"
                assert "columns" in data, "Response should contain 'columns' key"


@pytest.mark.asyncio
async def test_statcast_length_limit_exceeded():
    """Test that statcast tools handle length limit properly when response is too large."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test get_statcast_data with a large date range that should exceed limit
            result = await session.call_tool(
                "get_statcast_data",
                {"start_dt": "2023-04-01", "end_dt": "2023-04-30"},  # Full month
            )

            # Verify response structure - could be success or length limit error
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Verify correct length limitation
            assert "error" in data
            assert "length" in data
            assert "limit" in data
            assert "total_rows" in data

            # Try new call with end_row (use conservative calculation)
            bytes_per_row = data["length"] // data["total_rows"]
            max_safe_rows = (data["limit"] * 0.8) // bytes_per_row  # 80% of limit for safety
            new_row_estimate = max(1, int(max_safe_rows))
            result = await session.call_tool(
                "get_statcast_data",
                {
                    "start_dt": "2023-04-01",
                    "end_dt": "2023-04-30",
                    "end_row": new_row_estimate,
                },  # row limitation
            )
            data = json.loads(result.content[0].text)
            assert "data" in data
            assert "count" in data
            assert "total_rows" in data
            assert "columns" in data
            assert data["count"] == new_row_estimate


@pytest.mark.asyncio
async def test_statcast_batter_length_limit_handling():
    """Test that get_statcast_batter_data handles length limit with large date range."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with Aaron Judge and a large date range
            result = await session.call_tool(
                "get_statcast_batter_data",
                {
                    "player_id": 592450,  # Aaron Judge
                    "start_dt": "2022-04-01",
                    "end_dt": "2022-10-31",  # Full 2022 season
                },
            )

            # Verify response structure
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Verify correct length limitation
            assert "error" in data
            assert "length" in data
            assert "limit" in data
            assert "total_rows" in data

            # Try new call with end_row (use conservative calculation)
            bytes_per_row = data["length"] // data["total_rows"]
            max_safe_rows = (data["limit"] * 0.8) // bytes_per_row  # 80% of limit for safety
            new_row_estimate = max(1, int(max_safe_rows))
            result = await session.call_tool(
                "get_statcast_batter_data",
                {
                    "player_id": 592450,
                    "start_dt": "2022-04-01",
                    "end_dt": "2022-10-31",
                    "end_row": new_row_estimate,
                },
            )
            data = json.loads(result.content[0].text)
            assert "data" in data
            assert "count" in data
            assert "total_rows" in data
            assert "columns" in data
            assert data["count"] == new_row_estimate


@pytest.mark.asyncio
async def test_statcast_pitcher_length_limit_handling():
    """Test that get_statcast_pitcher_data handles length limit with large date range."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with Gerrit Cole and a large date range
            result = await session.call_tool(
                "get_statcast_pitcher_data",
                {
                    "player_id": 543037,  # Gerrit Cole
                    "start_dt": "2022-04-01",
                    "end_dt": "2022-10-31",  # Full 2022 season
                },
            )

            # Verify response structure
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Verify correct length limitation
            assert "error" in data
            assert "length" in data
            assert "limit" in data
            assert "total_rows" in data

            # Try new call with end_row (use conservative calculation)
            bytes_per_row = data["length"] // data["total_rows"]
            max_safe_rows = (data["limit"] * 0.8) // bytes_per_row  # 80% of limit for safety
            new_row_estimate = max(1, int(max_safe_rows))
            result = await session.call_tool(
                "get_statcast_pitcher_data",
                {
                    "player_id": 543037,
                    "start_dt": "2022-04-01",
                    "end_dt": "2022-10-31",
                    "end_row": new_row_estimate,
                },
            )
            data = json.loads(result.content[0].text)
            assert "data" in data
            assert "count" in data
            assert "total_rows" in data
            assert "columns" in data
            assert data["count"] == new_row_estimate


@pytest.mark.asyncio
async def test_statcast_batter_exitvelo_barrels_length_limit_handling():
    """Test that get_statcast_batter_exitvelo_barrels handles length limit properly."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with parameters that should trigger length limit (very low minBBE)
            result = await session.call_tool(
                "get_statcast_batter_exitvelo_barrels",
                {
                    "year": 2023,
                    "minBBE": 1,  # Very low threshold to get many batters
                },
            )

            # Verify response structure
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation
                assert "length" in data
                assert "limit" in data
                assert "total_rows" in data

                # Try new call with end_row (use conservative calculation)
                bytes_per_row = data["length"] // data["total_rows"]
                max_safe_rows = (data["limit"] * 0.8) // bytes_per_row  # 80% of limit for safety
                new_row_estimate = max(1, int(max_safe_rows))
                result = await session.call_tool(
                    "get_statcast_batter_exitvelo_barrels",
                    {
                        "year": 2023,
                        "minBBE": 1,
                        "end_row": new_row_estimate,
                    },
                )
                data = json.loads(result.content[0].text)
                assert "data" in data
                assert "count" in data
                assert "total_rows" in data
                assert "columns" in data
                assert data["count"] == new_row_estimate
            else:
                # If no length limit hit, just verify we got valid data
                assert "data" in data
                assert "count" in data
                assert "columns" in data


@pytest.mark.asyncio
async def test_statcast_pitcher_exitvelo_barrels_length_limit_handling():
    """Test that get_statcast_pitcher_exitvelo_barrels handles length limit properly."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with parameters that should trigger length limit (very low minBBE)
            result = await session.call_tool(
                "get_statcast_pitcher_exitvelo_barrels",
                {
                    "year": 2023,
                    "minBBE": 1,  # Very low threshold to get many pitchers
                },
            )

            # Verify response structure
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation
                assert "length" in data
                assert "limit" in data
                assert "total_rows" in data

                # Try new call with end_row (use conservative calculation)
                bytes_per_row = data["length"] // data["total_rows"]
                max_safe_rows = (data["limit"] * 0.8) // bytes_per_row  # 80% of limit for safety
                new_row_estimate = max(1, int(max_safe_rows))
                result = await session.call_tool(
                    "get_statcast_pitcher_exitvelo_barrels",
                    {
                        "year": 2023,
                        "minBBE": 1,
                        "end_row": new_row_estimate,
                    },
                )
                data = json.loads(result.content[0].text)
                assert "data" in data
                assert "count" in data
                assert "total_rows" in data
                assert "columns" in data
                assert data["count"] == new_row_estimate
            else:
                # If no length limit hit, just verify we got valid data
                assert "data" in data
                assert "count" in data
                assert "columns" in data


@pytest.mark.asyncio
async def test_statcast_batter_expected_stats_length_limit_handling():
    """Test that get_statcast_batter_expected_stats handles length limit properly."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with parameters that should trigger length limit (very low minPA)
            result = await session.call_tool(
                "get_statcast_batter_expected_stats",
                {
                    "year": 2023,
                    "minPA": 1,  # Very low threshold to get many batters
                },
            )

            # Verify response structure
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation
                assert "length" in data
                assert "limit" in data
                assert "total_rows" in data

                # Try new call with end_row (use conservative calculation)
                bytes_per_row = data["length"] // data["total_rows"]
                max_safe_rows = (data["limit"] * 0.8) // bytes_per_row  # 80% of limit for safety
                new_row_estimate = max(1, int(max_safe_rows))
                result = await session.call_tool(
                    "get_statcast_batter_expected_stats",
                    {
                        "year": 2023,
                        "minPA": 1,
                        "end_row": new_row_estimate,
                    },
                )
                data = json.loads(result.content[0].text)
                assert "data" in data
                assert "count" in data
                assert "total_rows" in data
                assert "columns" in data
                assert data["count"] == new_row_estimate
            else:
                # If no length limit hit, just verify we got valid data
                assert "data" in data
                assert "count" in data
                assert "columns" in data


@pytest.mark.asyncio
async def test_statcast_pitcher_expected_stats_length_limit_handling():
    """Test that get_statcast_pitcher_expected_stats handles length limit properly."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with parameters that should trigger length limit (very low minPA)
            result = await session.call_tool(
                "get_statcast_pitcher_expected_stats",
                {
                    "year": 2023,
                    "minPA": 1,  # Very low threshold to get many pitchers
                },
            )

            # Verify response structure
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation
                assert "length" in data
                assert "limit" in data
                assert "total_rows" in data

                # Try new call with end_row (use conservative calculation)
                bytes_per_row = data["length"] // data["total_rows"]
                max_safe_rows = (data["limit"] * 0.8) // bytes_per_row  # 80% of limit for safety
                new_row_estimate = max(1, int(max_safe_rows))
                result = await session.call_tool(
                    "get_statcast_pitcher_expected_stats",
                    {
                        "year": 2023,
                        "minPA": 1,
                        "end_row": new_row_estimate,
                    },
                )
                data = json.loads(result.content[0].text)
                assert "data" in data
                assert "count" in data
                assert "total_rows" in data
                assert "columns" in data
                assert data["count"] == new_row_estimate
            else:
                # If no length limit hit, just verify we got valid data
                assert "data" in data
                assert "count" in data
                assert "columns" in data


@pytest.mark.asyncio
async def test_statcast_batter_percentile_ranks_length_limit_handling():
    """Test that get_statcast_batter_percentile_ranks handles length limit properly."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with year that should trigger length limit
            result = await session.call_tool(
                "get_statcast_batter_percentile_ranks",
                {
                    "year": 2023,
                },
            )

            # Verify response structure
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation
                assert "length" in data
                assert "limit" in data
                assert "total_rows" in data

                # Try new call with end_row (use conservative calculation)
                bytes_per_row = data["length"] // data["total_rows"]
                max_safe_rows = (data["limit"] * 0.8) // bytes_per_row  # 80% of limit for safety
                new_row_estimate = max(1, int(max_safe_rows))
                result = await session.call_tool(
                    "get_statcast_batter_percentile_ranks",
                    {
                        "year": 2023,
                        "end_row": new_row_estimate,
                    },
                )
                data = json.loads(result.content[0].text)
                assert "data" in data
                assert "count" in data
                assert "total_rows" in data
                assert "columns" in data
                assert data["count"] == new_row_estimate
            else:
                # If no length limit hit, just verify we got valid data
                assert "data" in data
                assert "count" in data
                assert "columns" in data


@pytest.mark.asyncio
async def test_statcast_pitcher_percentile_ranks_length_limit_handling():
    """Test that get_statcast_pitcher_percentile_ranks handles length limit properly."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with year that should trigger length limit
            result = await session.call_tool(
                "get_statcast_pitcher_percentile_ranks",
                {
                    "year": 2023,
                },
            )

            # Verify response structure
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation
                assert "length" in data
                assert "limit" in data
                assert "total_rows" in data

                # Try new call with end_row (use conservative calculation)
                bytes_per_row = data["length"] // data["total_rows"]
                max_safe_rows = (data["limit"] * 0.8) // bytes_per_row  # 80% of limit for safety
                new_row_estimate = max(1, int(max_safe_rows))
                result = await session.call_tool(
                    "get_statcast_pitcher_percentile_ranks",
                    {
                        "year": 2023,
                        "end_row": new_row_estimate,
                    },
                )
                data = json.loads(result.content[0].text)
                assert "data" in data
                assert "count" in data
                assert "total_rows" in data
                assert "columns" in data
                assert data["count"] == new_row_estimate
            else:
                # If no length limit hit, just verify we got valid data
                assert "data" in data
                assert "count" in data
                assert "columns" in data


@pytest.mark.asyncio
async def test_statcast_batter_pitch_arsenal_length_limit_handling():
    """Test that get_statcast_batter_pitch_arsenal handles length limit properly."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with parameters that might exceed limit (low minPA)
            result = await session.call_tool(
                "get_statcast_batter_pitch_arsenal",
                {
                    "year": 2023,
                    "minPA": 1,  # Very low threshold to get many batters
                },
            )

            # Verify response structure
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation
                assert "length" in data
                assert "limit" in data
                assert "total_rows" in data

                # Try new call with end_row (use conservative calculation)
                bytes_per_row = data["length"] // data["total_rows"]
                max_safe_rows = (data["limit"] * 0.8) // bytes_per_row  # 80% of limit for safety
                new_row_estimate = max(1, int(max_safe_rows))
                result = await session.call_tool(
                    "get_statcast_batter_pitch_arsenal",
                    {
                        "year": 2023,
                        "minPA": 1,
                        "end_row": new_row_estimate,
                    },
                )
                data = json.loads(result.content[0].text)
                assert "data" in data
                assert "count" in data
                assert "total_rows" in data
                assert "columns" in data
                assert data["count"] == new_row_estimate
            else:
                # If no length limit hit, just verify we got valid data
                assert "data" in data
                assert "count" in data
                assert "columns" in data


@pytest.mark.asyncio
async def test_statcast_pitcher_pitch_arsenal_length_limit_handling():
    """Test that get_statcast_pitcher_pitch_arsenal handles length limit properly."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Test with parameters that should trigger length limit (very low minP)
            result = await session.call_tool(
                "get_statcast_pitcher_pitch_arsenal",
                {
                    "year": 2023,
                    "minP": 1,  # Very low threshold to get many pitchers
                    "arsenal_type": "avg_speed",
                },
            )

            # Verify response structure
            assert not result.isError, "Expected successful response or length limit handling"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            data = json.loads(result.content[0].text)

            # Check if we got successful data or length limit error
            if "error" in data:
                # Verify correct length limitation
                assert "length" in data
                assert "limit" in data
                assert "total_rows" in data

                # Try new call with end_row (use conservative calculation)
                bytes_per_row = data["length"] // data["total_rows"]
                max_safe_rows = (data["limit"] * 0.8) // bytes_per_row  # 80% of limit for safety
                new_row_estimate = max(1, int(max_safe_rows))
                result = await session.call_tool(
                    "get_statcast_pitcher_pitch_arsenal",
                    {
                        "year": 2023,
                        "minP": 1,
                        "arsenal_type": "avg_speed",
                        "end_row": new_row_estimate,
                    },
                )
                data = json.loads(result.content[0].text)
                assert "data" in data
                assert "count" in data
                assert "total_rows" in data
                assert "columns" in data
                assert data["count"] == new_row_estimate
            else:
                # If no length limit hit, just verify we got valid data
                assert "data" in data
                assert "count" in data
                assert "columns" in data


@pytest.mark.asyncio
async def test_statcast_single_game_with_truncation():
    """Test that get_statcast_single_game works with start_row and end_row parameters."""
    params = simplify_session_setup()

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # First, get a game that might have a lot of data
            result = await session.call_tool(
                "get_statcast_single_game",
                {"game_pk": 717953, "start_row": 0, "end_row": 10},  # Just first 10 rows
            )

            # Verify successful response
            assert not result.isError, "Expected successful response"
            assert result.content, "No content returned from tool"
            assert result.content[0].type == "text", "Expected text response"

            # Verify successful response with truncation
            data = json.loads(result.content[0].text)
            assert "data" in data
            assert "count" in data
            assert "total_rows" in data
            assert "columns" in data
            assert data["count"] == 10
