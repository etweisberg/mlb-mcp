"""
Tools for pybaseball plotting functions
"""

import base64
import io
from contextlib import contextmanager
from typing import Any, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pybaseball import plot_teams, spraychart
from pybaseball.plotting import plot_bb_profile, plot_strike_zone

from mlb_stats_mcp.utils.logging_config import setup_logging

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
    ax.figure.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
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


async def create_strike_zone_plot(
    data: Dict[str, Any],
    title: str = "",
    colorby: str = "pitch_type",
    legend_title: str = "",
    annotation: str = "pitch_type",
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
        if not data.get("data"):
            raise ValueError("No data provided for plotting")

        df = pd.DataFrame(data["data"])

        # Use pybaseball's plot_strike_zone function
        with no_show():
            ax = plot_strike_zone(
                df,
                title=title,
                colorby=colorby,
                legend_title=legend_title,
                annotation=annotation,
            )

        # Save plot to base64
        plot_image = _axes_to_base64(ax)

        logger.debug(f"Created strike zone plot with {len(df)} pitches")

        return {
            "plot_type": "strike_zone",
            "image_base64": plot_image,
            "pitch_count": len(df),
            "title": title,
            "metadata": {
                "colorby": colorby,
                "annotation": annotation,
            },
        }

    except Exception as e:
        error_msg = f"Error creating strike zone plot: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


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
        if not data.get("data"):
            raise ValueError("No data provided for plotting")

        df = pd.DataFrame(data["data"])

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

        # Save plot to base64
        plot_image = _axes_to_base64(ax)

        logger.debug(f"Created spraychart with {len(hit_data)} hits for {team_stadium} stadium")

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
            },
        }

    except Exception as e:
        error_msg = f"Error creating spraychart: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def create_bb_profile_plot(
    data: Dict[str, Any],
    parameter: str = "launch_angle",
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
        if not data.get("data"):
            raise ValueError("No data provided for plotting")

        df = pd.DataFrame(data["data"])

        # Create new figure
        plt.figure(figsize=(10, 6))

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
                    df["bb_type"].value_counts().to_dict() if "bb_type" in df.columns else {}
                )
            },
        }

    except Exception as e:
        error_msg = f"Error creating batted ball profile plot: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


async def create_teams_plot(
    data: Dict[str, Any],
    x_axis: str,
    y_axis: str,
    title: Optional[str] = None,
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
        if not data.get("data"):
            raise ValueError("No data provided for plotting")

        df = pd.DataFrame(data["data"])

        # Create new figure
        plt.figure(figsize=(12, 8))

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
            "team_count": len(df),
            "x_axis": x_axis,
            "y_axis": y_axis,
            "title": title,
            "metadata": {"teams": df["Team"].tolist() if "Team" in df.columns else []},
        }

    except Exception as e:
        error_msg = f"Error creating teams plot: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
