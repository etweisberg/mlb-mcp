"""
MCP prompt functions for the baseball server.
These prompts provide structured templates to guide LLM interactions.
"""

from typing import Optional


def player_report(player_name: str, season: Optional[int] = None) -> str:
    """
    Generate a comprehensive player performance report with statistics and visualizations.

    Args:
        player_name: Full name or partial name of the player
        season: Season year (defaults to current season if not specified)

    Returns:
        Detailed prompt instructing the LLM how to create a comprehensive player report
    """
    season_text = f"the {season} season" if season else "the current/most recent season"

    return f"""Create a comprehensive performance report for {player_name} for {season_text}.

STEP 1: PLAYER IDENTIFICATION & BASIC INFO
1. Use lookup_player("{player_name}") to find the player and get their MLB ID
2. If multiple players are found, select the most relevant/current one
3. Use get_playerid_lookup() if needed for additional player identification

STEP 2: DETERMINE PLAYER TYPE & GET CORE STATS
1. Use get_player_stats(player_id, group="hitting") to get batting stats
2. Use get_player_stats(player_id, group="pitching") to get pitching stats
3. Based on which stats are more relevant, determine if this is primarily a:
   - BATTER: Focus on hitting stats, batting visualizations
   - PITCHER: Focus on pitching stats, pitcher visualizations
   - TWO-WAY PLAYER: Include both hitting and pitching analysis

STEP 3: GET ADVANCED METRICS (STATCAST DATA) - PRIMARY SOURCE FOR EXIT VELOCITY
**IMPORTANT: Use Statcast tools for ALL exit velocity analysis, NOT traditional MLB Stats API**

For BATTERS:
- **get_statcast_batter_data(player_id{
        ', start_dt="' + str(season) + '-01-01", end_dt="' + str(season) + '-12-31"'
        if season
        else ""
    }) - PRIMARY source for exit velocity data**
- get_statcast_batter_expected_stats({season if season else "current_year"}) - For xwOBA, xBA, xSLG
- get_statcast_batter_percentile_ranks({
        season if season else "current_year"
    }) - For league percentile rankings
- **get_statcast_batter_exitvelo_barrels({
        season if season else "current_year"
    }) - SPECIALIZED exit velocity and barrel metrics**

For PITCHERS:
- **get_statcast_pitcher_data(player_id{
        ', start_dt="' + str(season) + '-01-01", end_dt="' + str(season) + '-12-31"'
        if season
        else ""
    }) - PRIMARY source for exit velocity allowed**
- get_statcast_pitcher_expected_stats({season if season else "current_year"})
- get_statcast_pitcher_percentile_ranks({season if season else "current_year"})
- **get_statcast_pitcher_exitvelo_barrels({
        season if season else "current_year"
    }) - SPECIALIZED exit velocity allowed metrics**

STEP 4: DISCOVER VALID PARAMETERS FOR TRADITIONAL STATS (if needed)
**Before using get_league_leader_data or similar functions, discover valid parameters:**
- get_meta(type_name="leagueLeaderTypes") - Get valid leaderCategories
- get_meta(type_name="statGroups") - Get valid statGroups (REQUIRED for accurate results)
- get_meta(type_name="gameTypes") - Get valid gameTypes
- get_meta(type_name="statTypes") - Get valid statTypes

**Note**: When using get_league_leader_data(), ALWAYS include statGroup parameter for
accurate results. For example: leaderCategories='earnedRunAverage' returns different
results with statGroup='pitching' vs statGroup='catching'.

STEP 5: CREATE VISUALIZATIONS
For BATTERS - Create these plots using the statcast data:
1. create_strike_zone_plot(statcast_data, title="{player_name} Strike Zone Profile",
   colorby="events")
2. create_spraychart_plot(statcast_data, title="{player_name} Spray Chart",
   colorby="events")
3. create_bb_profile_plot(statcast_data, parameter="launch_angle")
4. **create_bb_profile_plot(statcast_data, parameter="exit_velocity") - REQUIRED:
   Exit velocity distribution from STATCAST data**

For PITCHERS - Create these plots using the statcast data:
1. create_strike_zone_plot(statcast_data, title="{player_name} Pitch Locations",
   colorby="pitch_type")
2. create_bb_profile_plot(statcast_data, parameter="release_speed")
3. **create_bb_profile_plot(statcast_data, parameter="exit_velocity") - REQUIRED:
   Exit velocity allowed from STATCAST data**

STEP 6: GET CONTEXTUAL DATA
1. Use get_team_roster() to find what team the player is currently on
2. Use get_standings() to see how their team is performing
3. **ONLY if traditional stats are needed**: Use get_league_leader_data() with proper
   statGroup parameter

STEP 7: CREATE HTML REPORT
Generate a comprehensive HTML report with the following structure:

```html
<!DOCTYPE html>
<html>
<head>
    <title>{player_name} Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #003366;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .section {{ margin-bottom: 30px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
            }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #003366; }}
        .stat-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .statcast-highlight {{
            background-color: #fff3cd;
            border: 2px solid #ffc107;
            }}
        .exitvelo-highlight {{
            background-color: #d1ecf1;
            border: 2px solid #17a2b8;
            }}
        .plot-container {{ text-align: center; margin: 20px 0; }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            }}
        .highlights {{ background-color: #e7f3ff; padding: 15px; border-radius: 8px; }}
        .analysis {{ background-color: #fff3e0; padding: 15px; border-radius: 8px; }}
        .statcast-section {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{player_name} Performance Report</h1>
            <h3>{season_text} • Powered by Statcast</h3>
        </div>

        <div class="section">
            <h2>Player Overview</h2>
            <!-- Include basic player info, team, position, etc. -->
        </div>

        <div class="section statcast-section">
            <h2>Statcast Exit Velocity & Contact Quality</h2>
            <p><strong>Data Source:</strong> All exit velocity metrics from Statcast
               pitch-by-pitch data</p>
            <div class="stats-grid">
                <!-- REQUIRED: Include these exit velocity stats from STATCAST data -->
                <div class="stat-card exitvelo-highlight">
                    <div class="stat-value">[STATCAST_AVG_EXIT_VELO]</div>
                    <div class="stat-label">Avg Exit Velocity (Statcast)</div>
                </div>
                <div class="stat-card exitvelo-highlight">
                    <div class="stat-value">[STATCAST_MAX_EXIT_VELO]</div>
                    <div class="stat-label">Max Exit Velocity (Statcast)</div>
                </div>
                <div class="stat-card statcast-highlight">
                    <div class="stat-value">[STATCAST_BARREL_RATE]</div>
                    <div class="stat-label">Barrel Rate (Statcast)</div>
                </div>
                <div class="stat-card statcast-highlight">
                    <div class="stat-value">[STATCAST_HARD_HIT_RATE]</div>
                    <div class="stat-label">Hard Hit Rate 95+ mph (Statcast)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">[STATCAST_SWEET_SPOT]</div>
                    <div class="stat-label">Sweet Spot % (Statcast)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">[STATCAST_SOLID_CONTACT]</div>
                    <div class="stat-label">Solid Contact % (Statcast)</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Traditional Statistics</h2>
            <div class="stats-grid">
                <!-- Create stat cards for traditional metrics -->
                <div class="stat-card">
                    <div class="stat-value">[VALUE]</div>
                    <div class="stat-label">[STAT NAME]</div>
                </div>
                <!-- Repeat for other key stats -->
            </div>
        </div>

        <div class="section">
            <h2>Performance Visualizations</h2>
            <!-- REQUIRED: Include exit velocity distribution plot from Statcast data -->
            <div class="plot-container">
                <h3>{player_name} Exit Velocity Distribution (Statcast)</h3>
                <img
                    src="data:image/png;base64,[STATCAST_EXIT_VELO_PLOT_BASE64]"
                    alt="Statcast Exit Velocity Distribution"
                >
                <p>
                    <strong>Source:</strong> Statcast pitch-by-pitch data. This chart shows
                    the distribution of exit velocities for all batted balls tracked by
                    Statcast, providing the most accurate measure of contact quality.
                </p>
            </div>
            <!-- Include other generated plots as base64 images -->
            <div class="plot-container">
                <h3>[Plot Title]</h3>
                <img src="data:image/png;base64,[BASE64_IMAGE_DATA]" alt="[Plot Description]">
                <p>[Brief description of what the plot shows]</p>
            </div>
        </div>

        <div class="section highlights">
            <h2>Season Highlights</h2>
            <h3>Statcast Exit Velocity Milestones</h3>
            <ul>
                <li>Hardest hit ball (Statcast): [MAX_EV] mph on [DATE]</li>
                <li>Number of 110+ mph batted balls (Statcast): [COUNT]</li>
                <li>Statcast exit velocity percentile ranking: [RANKING]</li>
                <li>Barrel rate ranking among qualified hitters: [BARREL_RANKING]</li>
            </ul>
        </div>

        <div class="section analysis">
            <h2>Statcast-Based Performance Analysis</h2>
            <h3>Contact Quality Assessment (Statcast Data)</h3>
            <p><strong>Exit Velocity Analysis:</strong> Based on Statcast pitch-tracking data:</p>
            <ul>
                <li><strong>Average Exit Velocity:</strong> Compare to league average (~89 mph)</li>
                <li><strong>Hard Hit Rate (95+ mph):</strong> Compare to league average (~35%)</li>
                <li><strong>Barrel Rate:</strong> Optimal EV + launch angle combinations</li>
                <li><strong>Sweet Spot Rate:</strong> Launch angles between 8-32 degrees</li>
                <li><strong>Max Exit Velocity:</strong> Peak power demonstration</li>
                <li><strong>Consistency:</strong> 90th percentile vs average exit velocity</li>
            </ul>
            <p><strong>Key Statcast Metrics:</strong></p>
            <ul>
                <li>avg_hit_speed: Overall contact quality</li>
                <li>max_hit_speed: Peak power capability</li>
                <li>barrel_batted_rate: Optimal contact percentage</li>
                <li>solidcontact_percent: Consistent hard contact</li>
                <li>sweetspot_percent: Ideal launch angle contact</li>
            </ul>
        </div>

        <div class="section">
            <h2>League Context</h2>
            <h3>Statcast League Rankings</h3>
            <p>Player's ranking among qualified hitters in key Statcast metrics:</p>
            <ul>
                <li>Average Exit Velocity: [RANKING]/[TOTAL]</li>
                <li>Hard Hit Rate (95+ mph): [RANKING]/[TOTAL]</li>
                <li>Barrel Rate: [RANKING]/[TOTAL]</li>
                <li>Sweet Spot Rate: [RANKING]/[TOTAL]</li>
            </ul>
        </div>
    </div>
</body>
</html>
```

**CRITICAL GUIDELINES:**

**EXIT VELOCITY DATA SOURCE:**
- **ALWAYS use Statcast tools** (get_statcast_batter_data,
  get_statcast_batter_exitvelo_barrels) for exit velocity analysis
- **NEVER use traditional MLB Stats API** for exit velocity - it may not be available
  or accurate
- Statcast provides pitch-by-pitch exit velocity measurements from Doppler radar
- Traditional stats may not include comprehensive exit velocity data

**META ENDPOINT USAGE:**
- **Before any get_league_leader_data() call:** Use get_meta() to discover valid parameters
- **Always include statGroup** when using leaderCategories to avoid unexpected results
- Use get_meta(type_name="leagueLeaderTypes") to find valid categories like "homeRuns",
  "battingAverage", etc.

**KEY STATCAST EXIT VELOCITY METRICS:**
- avg_hit_speed: Primary exit velocity metric
- max_hit_speed: Peak exit velocity achieved
- barrel_batted_rate: Percentage of optimal contact
- solidcontact_percent: Consistent quality contact rate
- sweetspot_percent: Ideal launch angle percentage (8-32°)
- hardhit_percent: Percentage of 95+ mph contact

**ANALYSIS FOCUS:**
- **Prioritize Statcast exit velocity analysis** as the primary measure of contact quality
- Compare Statcast metrics to league averages and percentiles
- Use traditional stats as supporting context, not primary analysis
- Emphasize the precision and accuracy of Statcast measurements

Generate a complete HTML report emphasizing Statcast-based exit velocity analysis as the
gold standard for contact quality assessment."""


def team_comparison(team1: str, team2: str, focus_area: str = "overall") -> str:
    """
    Generate a comprehensive comparison between two MLB teams.

    Args:
        team1: First team abbreviation (e.g., "NYY", "BOS")
        team2: Second team abbreviation
        focus_area: Area to focus on ("overall", "hitting", "pitching", "recent")

    Returns:
        Prompt for detailed team comparison analysis
    """
    return f"""Create a comprehensive comparison between {team1} and {team2} with focus on
{focus_area}.

STEP 1: BASIC TEAM INFO
1. Use get_standings() to get current records and division standings for both teams
2. Use get_schedule() to find recent head-to-head matchups this season
3. Get team rosters with get_team_roster() for both teams

STEP 2: DISCOVER VALID PARAMETERS FOR TRADITIONAL STATS
**Before using traditional MLB Stats API functions, discover valid parameters:**
- get_meta(type_name="leagueLeaderTypes") - Get valid leaderCategories for team leaders
- get_meta(type_name="statGroups") - Get valid statGroups (CRITICAL for accurate results)
- get_meta(type_name="gameTypes") - Get valid gameTypes if filtering by game type

STEP 3: STATISTICAL COMPARISON
Based on focus area "{focus_area}":

If "hitting" or "overall":
- get_team_batting(current_season) for both teams
- **Use proper statGroup**: get_team_leaders(team_id, leader_category="[VALID_CATEGORY]",
  statGroup="hitting")
- **Primary exit velocity analysis**: get_statcast_batter_exitvelo_barrels(current_year)
  for league context

If "pitching" or "overall":
- get_team_pitching(current_season) for both teams
- **Use proper statGroup**: get_team_leaders(team_id, leader_category="[VALID_CATEGORY]",
  statGroup="pitching")
- **Primary exit velocity allowed**: get_statcast_pitcher_exitvelo_barrels(current_year)
  for league context

If "recent":
- get_schedule() for last 10-15 games for each team
- Focus on recent performance trends

STEP 4: ADVANCED METRICS (STATCAST FOCUS)
**Prioritize Statcast data for contact quality analysis:**
- get_statcast_batter_expected_stats() - Team-level expected statistics
- get_statcast_pitcher_expected_stats() - Team-level pitcher expected statistics
- get_statcast_batter_exitvelo_barrels() - League exit velocity context for hitting comparison
- get_statcast_pitcher_exitvelo_barrels() - League exit velocity allowed context for
  pitching comparison

STEP 5: CREATE VISUALIZATIONS
Use create_teams_plot() to generate comparison charts:
1. Team offensive comparison using Statcast metrics (if hitting focus)
2. Team pitching comparison using Statcast metrics (if pitching focus)
3. Exit velocity comparison scatter plot using Statcast data

STEP 6: HEAD-TO-HEAD ANALYSIS
- Historical matchup data
- Key player matchups using Statcast contact quality metrics
- Recent series results

STEP 7: GENERATE HTML REPORT
Create detailed comparison report with:
- **Statcast-based contact quality comparisons** as primary analysis
- Traditional stats as supporting context
- Team logos, stats tables, and visual comparisons
- **Emphasis on exit velocity metrics** for offensive and defensive evaluation

**CRITICAL NOTES:**
- **Always use get_meta() before traditional API calls** to ensure valid parameters
- **Prioritize Statcast data** for all contact quality and power analysis
- **Include statGroup parameter** when using get_league_leader_data() or get_team_leaders()
- Focus on identifying competitive advantages through advanced metrics, not just
  traditional stats"""


def game_recap(game_id: int) -> str:
    """
    Generate a comprehensive game recap with statistics and key moments.

    Args:
        game_id: MLB game ID

    Returns:
        Prompt for detailed game recap generation
    """
    return f"""Create a comprehensive recap for game {game_id}.

STEP 1: BASIC GAME INFORMATION
1. get_boxscore({game_id}) - Get final score, basic stats, and game summary
2. get_linescore({game_id}) - Get inning-by-inning scoring breakdown
3. get_game_scoring_play_data({game_id}) - Get detailed scoring plays and key moments

STEP 2: ADVANCED GAME DATA (STATCAST PRIORITY)
1. **get_statcast_single_game({game_id}) - PRIMARY source for pitch-level data and
   exit velocity analysis**
2. get_game_highlight_data({game_id}) - Get video highlights and notable plays

STEP 3: KEY PLAYER PERFORMANCES
- Identify standout hitting and pitching performances from boxscore
- **Use Statcast single-game data to analyze:**
  - Exit velocity on key hits
  - Pitch velocity and movement on strikeouts
  - Contact quality throughout the game
  - Barrel rate and hard-hit rate for both teams
- Highlight clutch moments with Statcast context (exit velocity, launch angle, etc.)

STEP 4: VISUALIZATIONS (STATCAST-BASED)
Using the Statcast single-game data:
1. create_strike_zone_plot() for key pitchers' locations with velocity context
2. create_spraychart_plot() for notable hits/home runs with exit velocity annotations
3. **create_bb_profile_plot(parameter="exit_velocity") for game contact quality analysis**
4. create_bb_profile_plot(parameter="launch_angle") for batted ball analysis

STEP 5: DISCOVER TRADITIONAL STAT CONTEXT (if needed)
**Only if traditional leaderboard context is needed:**
- get_meta(type_name="leagueLeaderTypes") - Get valid categories for context
- get_meta(type_name="statGroups") - Ensure proper statGroup usage

STEP 6: GENERATE HTML GAME RECAP
Create detailed HTML report including:
- Game summary header with final score
- **Statcast highlight section** featuring:
  - Hardest hit balls of the game (exit velocity)
  - Fastest pitches thrown
  - Best contact quality moments
  - Barrel rate and sweet spot percentage by team
- Inning-by-inning scoring summary with Statcast context
- Key player stat lines enhanced with Statcast metrics
- Turning point analysis using exit velocity and contact quality data
- Visual breakdowns of key moments with Statcast measurements
- Post-game implications (standings, playoff picture, etc.)

**HTML STRUCTURE EMPHASIS:**
```html
<div class="statcast-game-highlights">
    <h2>Statcast Game Highlights</h2>
    <div class="highlight-grid">
        <div class="highlight-card">
            <h3>Hardest Hit Ball</h3>
            <p>[PLAYER]: [EXIT_VELO] mph, [DISTANCE] ft</p>
        </div>
        <div class="highlight-card">
            <h3>Fastest Pitch</h3>
            <p>[PITCHER]: [VELOCITY] mph [PITCH_TYPE]</p>
        </div>
        <div class="highlight-card">
            <h3>Best Barrel</h3>
            <p>[PLAYER]: [EXIT_VELO] mph at [LAUNCH_ANGLE]°</p>
        </div>
    </div>
</div>
```

**CRITICAL GUIDELINES:**
- **Statcast single-game data is the primary source** for all contact and pitch
  quality analysis
- Traditional boxscore provides game flow and basic stats
- **Focus on exit velocity, barrel rate, and pitch velocity** as key performance indicators
- Use Statcast data to provide context that traditional stats cannot

Structure the recap like a modern analytics-driven game report with Statcast insights
taking precedence."""


def statistical_deep_dive(
    stat_category: str, season: Optional[int] = None, min_qualifier: Optional[int] = None
) -> str:
    """
    Generate an in-depth statistical analysis for a specific category.

    Args:
        stat_category: Statistical category to analyze (e.g., "home_runs", "era", "steals")
        season: Season to analyze (current if not specified)
        min_qualifier: Minimum qualifying threshold

    Returns:
        Prompt for comprehensive statistical analysis
    """
    season_text = f"{season}" if season else "current season"

    return f"""Create a comprehensive statistical deep dive analysis for {stat_category} in the
{season_text}.

STEP 1: DISCOVER VALID PARAMETERS
**CRITICAL: Before using any traditional MLB Stats API functions:**
- get_meta(type_name="leagueLeaderTypes") - Get valid leaderCategories (ensure
  {stat_category} is valid)
- get_meta(type_name="statGroups") - Get valid statGroups (REQUIRED for accurate results)
- get_meta(type_name="gameTypes") - Get valid gameTypes if analysis needs filtering
- get_meta(type_name="statTypes") - Get valid statTypes for comprehensive understanding

**Note**: Always include appropriate statGroup when using get_league_leader_data().
Example: stat_category='earnedRunAverage' requires statGroup='pitching', not
statGroup='hitting'.

STEP 2: GATHER LEAGUE-WIDE DATA
1. **get_league_leader_data("{stat_category}", season={season or "current"},
   statGroup="[APPROPRIATE_GROUP]", limit=50)**
2. Get appropriate team-level data using get_team_batting() or get_team_pitching()
3. **Primary analysis using Statcast data:**
   - get_statcast_batter_expected_stats({season or "current_year"}) for batting statistics
   - get_statcast_pitcher_expected_stats({season or "current_year"}) for pitching statistics
   - **get_statcast_batter_exitvelo_barrels({season or "current_year"}) if stat_category
     relates to power/contact**
   - **get_statcast_pitcher_exitvelo_barrels({season or "current_year"}) if stat_category
     relates to contact allowed**

STEP 3: IDENTIFY KEY INSIGHTS
- Current leaders and their performance levels
- **Statcast context**: How does traditional stat relate to exit velocity, expected stats, etc.
- Historical context (how does this season compare?)
- Team-by-team breakdowns
- Notable trends or surprises
- **Expected vs Actual performance** using Statcast expected statistics

STEP 4: ADVANCED ANALYSIS (STATCAST PRIORITY)
- **Correlation with Statcast metrics** (exit velocity, barrel rate, expected stats)
- Impact on team success
- Predictive indicators using advanced metrics
- **Quality of contact analysis** for offensive statistics
- **Contact management analysis** for pitching statistics

STEP 5: VISUALIZATIONS
Create relevant plots based on the statistic:
- Distribution charts using create_bb_profile_plot() for Statcast context
- Team comparison plots using create_teams_plot() with Statcast metrics
- **Player-specific exit velocity visualizations** for top performers in power categories
- Expected vs Actual performance scatter plots

STEP 6: GENERATE COMPREHENSIVE REPORT
HTML format with:

```html
<div class="analysis-container">
    <div class="statcast-context-section">
        <h2>Statcast Context for {stat_category}</h2>
        <p>Analysis of how {stat_category} correlates with:</p>
        <ul>
            <li>Exit Velocity Metrics</li>
            <li>Expected Statistics (xwOBA, xBA, xSLG)</li>
            <li>Contact Quality Measures</li>
            <li>Barrel Rate and Hard Hit Rate</li>
        </ul>
    </div>

    <div class="traditional-leaders">
        <h2>Traditional {stat_category} Leaders</h2>
        <!-- Top 10 leaderboard with Statcast context -->
    </div>

    <div class="expected-vs-actual">
        <h2>Expected vs Actual Performance</h2>
        <!-- Compare traditional stats to Statcast expected metrics -->
    </div>
</div>
```

**Structure includes:**
- **Executive summary emphasizing Statcast insights**
- Top 10 leaderboard with traditional and expected stats
- **Exit velocity context** for power-related statistics
- Team rankings with advanced metric analysis
- Historical perspective enhanced with modern metrics
- **Statistical correlations focusing on Statcast data**
- Predictions and trends using advanced analytics

**CRITICAL REQUIREMENTS:**
- **Always verify parameter validity** using get_meta() before API calls
- **Prioritize Statcast analysis** over traditional statistics alone
- **Include statGroup parameter** in all get_league_leader_data() calls
- **Focus on contact quality metrics** for comprehensive player evaluation
- **Use expected statistics** to identify over/under-performers

Make the analysis suitable for serious analysts who prioritize advanced metrics and
contact quality assessment."""
