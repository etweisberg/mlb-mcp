"""
Tools for pybaseball plotting functions

Refactor note:
- Plotting tools now accept source parameters (e.g., player_id, date range, game_pk)
  and fetch the necessary data internally instead of receiving raw data dicts.
  They are designed to be extensible to cover common pybaseball plotting use cases.
"""

import base64
import io
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pybaseball import (
    plot_teams,
    spraychart,
    statcast,
    statcast_batter,
    statcast_pitcher,
    statcast_single_game,
)
from pybaseball.plotting import plot_bb_profile, plot_strike_zone

from mlb_stats_mcp.utils.logging_config import setup_logging
from mlb_stats_mcp.tools import pybaseball_supp_tools

logger = setup_logging("pybaseball_plotting_tools")


def _axes_to_base64(ax: matplotlib.axes.Axes) -> str:
    """
    Convert matplotlib Axes to base64 encoded PNG image.

    Args:
        ax: The matplotlib Axes object

    Returns:
        Base64 encoded string of the plot image
    """
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return f"data:image/png;base64,{image_base64}"


@contextmanager
def no_show():
    """Context manager to temporarily disable plt.show()"""
    original_show = plt.show
    plt.show = lambda: None
    try:
        yield
    finally:
        plt.show = original_show


def _apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Apply simple equality/in filters to a DataFrame."""
    if not filters:
        return df
    filtered = df
    for column, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            filtered = filtered[filtered[column].isin(list(value))]
        else:
            filtered = filtered[filtered[column] == value]
    return filtered


async def create_strike_zone_plot(
    title: str = "",
    colorby: str = "pitch_type",
    legend_title: str = "",
    annotation: str = "pitch_type",
    *,
    # Data source selectors
    player_id: Optional[int] = None,
    player_role: str = "pitcher",  # 'pitcher' | 'batter'
    start_dt: Optional[str] = None,
    end_dt: Optional[str] = None,
    game_pk: Optional[int] = None,
    team: Optional[str] = None,  # team abbrev for statcast range
    filters: Optional[Dict[str, Any]] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a strike zone plot with pitch locations overlaid.

    Args:
        data: Dictionary containing Statcast data with 'data' key
        title: Title for the plot
        colorby: Column to color code the pitches by
        legend_title: Title for the legend
        annotation: Column to annotate the pitches with

    Returns:
        Dictionary containing plot metadata and base64 encoded image
    """
    try:
        # Determine data source
        if player_id is not None:
            if player_role not in ("pitcher", "batter"):
                raise ValueError("player_role must be 'pitcher' or 'batter'")
            df = (
                statcast_pitcher(start_dt, end_dt, player_id)
                if player_role == "pitcher"
                else statcast_batter(start_dt, end_dt, player_id)
            )
        elif game_pk is not None:
            df = statcast_single_game(game_pk)
        elif start_dt is not None or end_dt is not None or team is not None:
            df = statcast(start_dt, end_dt, team)
        else:
            raise ValueError(
                "Must provide a data source: player_id, game_pk, or date range (start_dt/end_dt) optionally with team"
            )

        if df is None or len(df) == 0:
            raise ValueError("No data available for plotting")

        # Apply optional filters and row limit
        df = _apply_filters(df, filters)
        if max_rows is not None and max_rows > 0:
            df = df.head(max_rows)

        # Use pybaseball's plot_strike_zone function
        with no_show():
            ax = plot_strike_zone(
                df,
                title=title,
                colorby=colorby,
                legend_title=legend_title,
                annotation=annotation,
            )

        # Reduce figure size to minimize base64 size
        try:
            ax.figure.set_size_inches(4, 4)
        except Exception:
            pass
        plot_image = _axes_to_base64(ax)

        logger.debug(f"Created strike zone plot with {len(df)} pitches")

        return {
            "plot_type": "strike_zone",
            "image_base64": plot_image,
            "pitch_count": int(len(df)),
            "title": title,
            "metadata": {
                "colorby": colorby,
                "annotation": annotation,
                "player_id": player_id,
                "player_role": player_role,
                "game_pk": game_pk,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "team": team,
            },
        }

    except Exception as e:
        error_msg = f"Error creating strike zone plot: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def create_spraychart_plot(
    team_stadium: str = "generic",
    title: str = "",
    colorby: str = "events",
    legend_title: str = "",
    size: int = 100,
    width: int = 500,
    height: int = 500,
    *,
    # Data source selectors
    players: Optional[List[int]] = None,
    start_dt: Optional[str] = None,
    end_dt: Optional[str] = None,
    home_team: Optional[str] = None,
    game_pk: Optional[int] = None,
    team: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a spraychart plot showing hit locations overlaid on a stadium.

    Args:
        data: Dictionary containing Statcast data with 'data' key
        team_stadium: Team name for stadium overlay
        title: Title for the plot
        colorby: Column to color code the hits by
        legend_title: Title for the legend
        size: Size of the hit markers
        width: Width of the plot
        height: Height of the plot

    Returns:
        Dictionary containing plot metadata and base64 encoded image
    """
    try:
        # Determine data source(s)
        if players:
            frames: List[pd.DataFrame] = []
            for pid in players:
                frames.append(statcast_batter(start_dt, end_dt, pid))
            df = pd.concat(frames, ignore_index=True)
        elif game_pk is not None:
            df = statcast_single_game(game_pk)
        elif start_dt is not None or end_dt is not None or team is not None:
            df = statcast(start_dt, end_dt, team)
        else:
            raise ValueError(
                "Must provide a data source: players, game_pk, or date range (start_dt/end_dt) optionally with team"
            )

        if df is None or len(df) == 0:
            raise ValueError("No data available for plotting")

        # Optional home team filter (common use-case in docs)
        if home_team:
            if "home_team" in df.columns:
                df = df[df["home_team"] == home_team]

        # Apply additional filters and row limits
        df = _apply_filters(df, filters)
        if max_rows is not None and max_rows > 0:
            df = df.head(max_rows)

        # Filter for balls in play with coordinates
        hit_data = df.dropna(subset=["hc_x", "hc_y"])
        hit_data = hit_data[hit_data["events"].notna()]

        if len(hit_data) == 0:
            raise ValueError("No valid hit coordinate data found")

        # Use pybaseball's spraychart function
        with no_show():
            ax = spraychart(
                hit_data,
                team_stadium,
                title=title,
                size=size,
                colorby=colorby,
                legend_title=legend_title,
                width=width,
                height=height,
            )

        # Reduce figure size
        try:
            ax.figure.set_size_inches(width / 100.0, height / 100.0)
        except Exception:
            pass
        plot_image = _axes_to_base64(ax)

        logger.debug(
            f"Created spraychart with {len(hit_data)} hits for {team_stadium} stadium"
        )

        return {
            "plot_type": "spraychart",
            "image_base64": plot_image,
            "hit_count": len(hit_data),
            "stadium": team_stadium,
            "title": title,
            "metadata": {
                "colorby": colorby,
                "events": (
                    hit_data["events"].value_counts().to_dict()
                    if "events" in hit_data.columns
                    else {}
                ),
                "players": players,
                "home_team": home_team,
                "game_pk": game_pk,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "team": team,
            },
        }

    except Exception as e:
        error_msg = f"Error creating spraychart: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def create_bb_profile_plot(
    parameter: str = "launch_angle",
    *,
    # Data source selectors
    player_id: Optional[int] = None,
    player_role: str = "batter",  # 'batter' | 'pitcher'
    start_dt: Optional[str] = None,
    end_dt: Optional[str] = None,
    game_pk: Optional[int] = None,
    team: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a batted ball profile plot showing distribution by batted ball type.

    Args:
        data: Dictionary containing Statcast data with 'data' key
        parameter: Parameter to plot (launch_angle, exit_velocity, etc.)

    Returns:
        Dictionary containing plot metadata and base64 encoded image
    """
    try:
        # Determine data source
        if player_id is not None:
            if player_role not in ("batter", "pitcher"):
                raise ValueError("player_role must be 'batter' or 'pitcher'")
            df = (
                statcast_batter(start_dt, end_dt, player_id)
                if player_role == "batter"
                else statcast_pitcher(start_dt, end_dt, player_id)
            )
        elif game_pk is not None:
            df = statcast_single_game(game_pk)
        elif start_dt is not None or end_dt is not None or team is not None:
            df = statcast(start_dt, end_dt, team)
        else:
            raise ValueError(
                "Must provide a data source: player_id, game_pk, or date range (start_dt/end_dt) optionally with team"
            )

        if df is None or len(df) == 0:
            raise ValueError("No data available for plotting")

        # Apply filters and max_rows
        df = _apply_filters(df, filters)
        if max_rows is not None and max_rows > 0:
            df = df.head(max_rows)

        # Create new, smaller figure
        plt.figure(figsize=(6, 4))

        # Use pybaseball's plot_bb_profile function (plots to current axes)
        plot_bb_profile(df, parameter=parameter)

        # Get the current axes object after plotting
        ax = plt.gca()

        # Add title and labels if not already set
        if not ax.get_title():
            ax.set_title(f"Batted Ball Profile: {parameter.replace('_', ' ').title()}")
        if not ax.get_xlabel():
            ax.set_xlabel(parameter.replace("_", " ").title())
        if not ax.get_ylabel():
            ax.set_ylabel("Frequency")

        # Add legend if not already present
        if not ax.get_legend():
            ax.legend()

        # Save plot to base64
        plot_image = _axes_to_base64(ax)

        # Clean up
        plt.close()

        logger.debug(f"Created batted ball profile plot with parameter: {parameter}")

        return {
            "plot_type": "bb_profile",
            "image_base64": plot_image,
            "bb_count": len(df),
            "parameter": parameter,
            "metadata": {
                "bb_types": (
                    df["bb_type"].value_counts().to_dict()
                    if "bb_type" in df.columns
                    else {}
                ),
                "player_id": player_id,
                "player_role": player_role,
                "game_pk": game_pk,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "team": team,
            },
        }

    except Exception as e:
        error_msg = f"Error creating batted ball profile plot: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def create_teams_plot(
    x_axis: str,
    y_axis: str,
    title: Optional[str] = None,
    *,
    dataset: str = "batting",  # 'batting' | 'pitching'
    start_season: Optional[int] = None,
    end_season: Optional[int] = None,
    league: str = "all",
    ind: int = 1,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a team statistics plot comparing two stats.

    Args:
        data: Dictionary containing team data with 'data' key
        x_axis: Name of stat to be plotted as the x_axis
        y_axis: Name of stat to be plotted as the y_axis
        title: Title for the chart

    Returns:
        Dictionary containing plot metadata and base64 encoded image
    """
    try:
        if start_season is None:
            raise ValueError("start_season is required for team plots")

        # Retrieve team data
        if dataset not in ("batting", "pitching"):
            raise ValueError("dataset must be 'batting' or 'pitching'")

        # Prefer direct pybaseball calls if available; fallback to our supplemental tools
        df: pd.DataFrame
        try:
            if dataset == "batting":
                # pybaseball has team_batting; but if not importable, fallback below
                from pybaseball import team_batting as _team_batting  # type: ignore

                df = _team_batting(start_season, end_season, league=league, ind=ind)
            else:
                from pybaseball import team_pitching as _team_pitching  # type: ignore

                df = _team_pitching(start_season, end_season, league=league, ind=ind)
        except Exception:
            # Fallback to our async supplemental tools which return dicts
            if dataset == "batting":
                data = await pybaseball_supp_tools.get_team_batting(
                    start_season, end_season, league, ind
                )
            else:
                data = await pybaseball_supp_tools.get_team_pitching(
                    start_season, end_season, league, ind
                )
            df = pd.DataFrame(data.get("data", []))

        if df is None or len(df) == 0:
            raise ValueError("No team data available for plotting")

        # Apply filters
        df = _apply_filters(df, filters)

        # Create new, smaller figure
        plt.figure(figsize=(6, 4))

        # Use context manager to disable show()
        with no_show():
            plot_teams(df, x_axis, y_axis, title)

        # Get the current axes object after plotting
        ax = plt.gca()

        # Save plot to base64
        plot_image = _axes_to_base64(ax)

        # Clean up
        plt.close()

        logger.debug(f"Created team plot: {x_axis} vs {y_axis}")

        return {
            "plot_type": "teams",
            "image_base64": plot_image,
            "team_count": int(len(df)),
            "x_axis": x_axis,
            "y_axis": y_axis,
            "title": title,
            "metadata": {
                "teams": df["Team"].tolist() if "Team" in df.columns else [],
                "dataset": dataset,
                "start_season": start_season,
                "end_season": end_season,
                "league": league,
                "ind": ind,
            },
        }

    except Exception as e:
        error_msg = f"Error creating teams plot: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def create_stadium_plot(
    team: str = "generic", width: int = 350, height: int = 350
) -> Dict[str, Any]:
    """
    Plot the outline of a specified team's stadium using MLBAM coordinates.

    Args:
        team: Stadium team key (e.g., 'astros', 'reds', 'generic', etc.)
        width: Width of the plot
        height: Height of the plot

    Returns:
        Dictionary containing plot metadata and base64 encoded image
    """
    try:
        with no_show():
            from pybaseball import plot_stadium

            ax = plot_stadium(team)
            # Ensure figure size
            ax.figure.set_size_inches(width / 100.0, height / 100.0)

        plot_image = _axes_to_base64(ax)

        return {
            "plot_type": "stadium",
            "image_base64": plot_image,
            "stadium": team,
            "metadata": {"width": width, "height": height},
        }
    except Exception as e:
        error_msg = f"Error creating stadium plot: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
