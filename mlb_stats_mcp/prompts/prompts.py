"""
MCP prompt functions for the baseball server.
Two distinct sets: Claude Desktop (scouting reports) and Web UI (structured plots).
"""

from typing import Optional

# ============================================================================
# CLAUDE DESKTOP PROMPTS - COMPREHENSIVE SCOUTING REPORTS AS ARTIFACTS
# ============================================================================


def cd_player_report(player_name: str, season: Optional[int] = None) -> str:
    """
    Generate a comprehensive player scouting report as an artifact for Claude Desktop.

    Args:
        player_name: Full name or partial name of the player
        season: Season year (defaults to current season if not specified)

    Returns:
        Detailed prompt for creating a professional scouting report artifact
    """
    season_text = f"the {season} season" if season else "the current/most recent season"

    return f"""Create a comprehensive professional scouting report for {player_name} for \
{season_text}.
Deliver this as a detailed markdown artifact that resembles a professional MLB scouting report.

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

STEP 3: GET ADVANCED METRICS (STATCAST DATA) - PRIMARY SOURCE FOR CONTACT QUALITY
**IMPORTANT: Use Statcast tools for ALL exit velocity and contact quality analysis**

For BATTERS:
- **get_statcast_batter_data(player_id{
        ', start_dt="' + str(season) + '-01-01", end_dt="' + str(season) + '-12-31"'
        if season
        else ""
    }) - PRIMARY source for \
pitch-level data**
- get_statcast_batter_expected_stats({season if season else "current_year"}) - \
For xwOBA, xBA, xSLG
- get_statcast_batter_percentile_ranks({season if season else "current_year"}) - \
For league percentile rankings
- **get_statcast_batter_exitvelo_barrels({season if season else "current_year"}) - \
SPECIALIZED exit velocity and barrel metrics**

For PITCHERS:
- **get_statcast_pitcher_data(player_id{
        ', start_dt="' + str(season) + '-01-01", end_dt="' + str(season) + '-12-31"'
        if season
        else ""
    }) - PRIMARY source for \
pitch-level data**
- get_statcast_pitcher_expected_stats({season if season else "current_year"})
- get_statcast_pitcher_percentile_ranks({season if season else "current_year"})
- **get_statcast_pitcher_exitvelo_barrels({season if season else "current_year"}) - \
SPECIALIZED exit velocity allowed metrics**

STEP 4: DISCOVER VALID PARAMETERS FOR TRADITIONAL STATS (if needed)
**Before using get_league_leader_data or similar functions, discover valid parameters:**
- get_meta(type_name="leagueLeaderTypes") - Get valid leaderCategories
- get_meta(type_name="statGroups") - Get valid statGroups (REQUIRED for accurate results)
- get_meta(type_name="gameTypes") - Get valid gameTypes
- get_meta(type_name="statTypes") - Get valid statTypes

STEP 5: GET CONTEXTUAL DATA
1. Use get_team_roster() to find what team the player is currently on
2. Use get_standings() to see how their team is performing
3. **ONLY if traditional stats are needed**: Use get_league_leader_data() with \
proper statGroup parameter

STEP 6: CREATE COMPREHENSIVE MARKDOWN SCOUTING REPORT ARTIFACT
Generate a detailed markdown report with the following structure:

# {player_name} Scouting Report
## {season_text} • Advanced Analytics Profile

### Executive Summary
- Overall player grade and projection
- Key strengths and areas for improvement
- Role fit and usage recommendations

### Player Profile
- Basic information (team, position, age, experience)
- Physical tools and measurables
- Current team context and standing

### Offensive Analysis (for batters)
**Contact Quality & Exit Velocity (Statcast)**
- Average Exit Velocity: [STATCAST VALUE] mph (League Rank: X/XXX)
- Max Exit Velocity: [STATCAST VALUE] mph
- Hard Hit Rate (95+ mph): [STATCAST VALUE]% (League Average: ~35%)
- Barrel Rate: [STATCAST VALUE]% (Elite: 15%+)
- Sweet Spot Rate: [STATCAST VALUE]% (Launch angles 8-32°)
- Solid Contact Rate: [STATCAST VALUE]%

**Expected Performance vs Reality**
- xwOBA: [VALUE] vs wOBA: [VALUE] (Difference: [+/-VALUE])
- xBA: [VALUE] vs BA: [VALUE]
- xSLG: [VALUE] vs SLG: [VALUE]

**Traditional Statistics**
- Key offensive numbers with league context
- Situational performance (RISP, late innings, etc.)

### Pitching Analysis (for pitchers)
**Contact Management (Statcast)**
- Average Exit Velocity Allowed: [STATCAST VALUE] mph
- Hard Hit Rate Allowed: [STATCAST VALUE]%
- Barrel Rate Allowed: [STATCAST VALUE]%
- Whiff Rate: [STATCAST VALUE]%
- Chase Rate: [STATCAST VALUE]%

**Pitch Arsenal**
- Pitch mix and velocity profiles
- Movement profiles and effectiveness
- Command and control metrics

**Expected Performance**
- xERA vs ERA comparison
- xFIP and SIERA context

### Percentile Rankings
League percentile rankings in key Statcast metrics:
- Exit Velocity: XXth percentile
- Hard Hit Rate: XXth percentile
- Barrel Rate: XXth percentile
- (Additional relevant percentiles)

### Situational Performance
- Performance in high-leverage situations
- Platoon splits and matchup considerations
- Home/road splits
- Recent trends and hot/cold streaks

### Scouting Notes
**Strengths:**
- Elite contact quality metrics
- Specific skill advantages
- Performance in key situations

**Areas for Development:**
- Weaknesses identified through analytics
- Opportunities for improvement
- Consistency concerns

**Usage Recommendations:**
- Optimal lineup position/role
- Matchup advantages to exploit
- Development focus areas

### League Context & Projections
- Ranking among qualified players at position
- Team impact and value
- Rest-of-season outlook
- Contract/team control status

**CRITICAL GUIDELINES:**
- **Prioritize Statcast data** as primary evaluation tool
- Use traditional stats as supporting context only
- **Always include exit velocity analysis** for contact quality assessment
- Compare all metrics to league averages and percentiles
- Focus on actionable insights for player development and usage
- Structure like a professional scouting report with clear recommendations"""


def cd_team_comparison(team1: str, team2: str, focus_area: str = "overall") -> str:
    """
    Generate a comprehensive team comparison report as an artifact for Claude Desktop.

    Args:
        team1: First team abbreviation (e.g., "NYY", "BOS")
        team2: Second team abbreviation
        focus_area: Area to focus on ("overall", "hitting", "pitching", "recent")

    Returns:
        Prompt for detailed team comparison scouting report
    """
    return f"""Create a comprehensive team comparison scouting report between {team1} and {team2}
with focus on {focus_area}. Deliver as a detailed markdown artifact.

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
- **Use proper statGroup**: get_team_leaders(team_id, \
leader_category="[VALID_CATEGORY]", statGroup="hitting")
- **Primary contact quality analysis**: \
get_statcast_batter_exitvelo_barrels(current_year) for league context

If "pitching" or "overall":
- get_team_pitching(current_season) for both teams
- **Use proper statGroup**: get_team_leaders(team_id, \
leader_category="[VALID_CATEGORY]", statGroup="pitching")
- **Primary contact management**: \
get_statcast_pitcher_exitvelo_barrels(current_year) for league context

If "recent":
- get_schedule() for last 10-15 games for each team
- Focus on recent performance trends

STEP 4: ADVANCED METRICS (STATCAST FOCUS)
**Prioritize Statcast data for contact quality analysis:**
- get_statcast_batter_expected_stats() - Team-level expected statistics
- get_statcast_pitcher_expected_stats() - Team-level pitcher expected statistics
- get_statcast_batter_exitvelo_barrels() - League exit velocity context for hitting comparison
- get_statcast_pitcher_exitvelo_barrels() - League exit velocity allowed context \
for pitching comparison

STEP 5: HEAD-TO-HEAD ANALYSIS
- Historical matchup data
- Key player matchups using Statcast contact quality metrics
- Recent series results

STEP 6: GENERATE COMPREHENSIVE TEAM COMPARISON REPORT ARTIFACT
Create detailed markdown comparison report with:

# Team Comparison: {team1} vs {team2}
## {focus_area.title()} Analysis • Advanced Metrics Focus

### Executive Summary
- Head-to-head competitive assessment
- Key advantages for each team
- Projected series/matchup outcomes

### Current Standings & Context
- Division standings and playoff positioning
- Recent performance trends
- Season trajectory analysis

### Offensive Comparison (Statcast-Based)
**Contact Quality Metrics**
- Team exit velocity averages and rankings
- Hard hit rate and barrel rate comparison
- Expected vs actual offensive production
- Key offensive contributors analysis

### Pitching Comparison (Statcast-Based)
**Contact Management Analysis**
- Exit velocity allowed comparison
- Hard contact prevention metrics
- Expected vs actual pitching performance
- Key pitching staff contributors

### Head-to-Head Matchup Analysis
- Historical series results
- Key individual matchups
- Stylistic advantages and disadvantages
- Recent series performance

### Situational Advantages
- Home/road performance splits
- Performance vs similar opponents
- Clutch performance metrics
- Bullpen usage and effectiveness

### Roster Construction Analysis
- Depth chart comparisons
- Injury concerns and roster flexibility
- Platoon advantages
- Bench and bullpen strength

### Competitive Assessment
**{team1} Advantages:**
- Specific strengths vs this opponent
- Matchup advantages to exploit

**{team2} Advantages:**
- Specific strengths vs this opponent
- Matchup advantages to exploit

### Series/Matchup Prediction
- Projected competitive balance
- Key factors likely to determine outcomes
- X-factors and wild cards

**CRITICAL GUIDELINES:**
- **Always use get_meta() before traditional API calls** to ensure valid parameters
- **Prioritize Statcast data** for all contact quality and power analysis
- **Include statGroup parameter** when using get_league_leader_data() or get_team_leaders()
- Focus on identifying competitive advantages through advanced metrics
- Structure as professional scouting report with actionable insights"""


def cd_game_recap(date: str, team1: str, team2: str) -> str:
    """
    Generate a comprehensive game recap scouting report as an artifact for Claude Desktop.

    Args:
        date: Game date in YYYY-MM-DD format
        team1: First team abbreviation (e.g., "NYY", "BOS")
        team2: Second team abbreviation

    Returns:
        Prompt for detailed game recap analysis
    """
    return f"""Create a comprehensive game recap analysis for the game between \
{team1} and {team2} on {date}.
Deliver as a detailed markdown artifact structured like a post-game scouting report.

STEP 1: FIND THE GAME
1. Use get_schedule(date="{date}", team_id=None) to get all games on {date}
2. Filter the schedule results to find the game between {team1} and \
{team2}
3. Extract the game_id (or game_pk) from the matching game
4. If no game found between these teams on this date, inform the user and \
suggest checking the date or team abbreviations

STEP 2: BASIC GAME INFORMATION
1. get_boxscore(game_id) - Get final score, basic stats, and game summary
2. get_linescore(game_id) - Get inning-by-inning scoring breakdown
3. get_game_scoring_play_data(game_id) - Get detailed scoring plays and key moments

STEP 3: ADVANCED GAME DATA (STATCAST PRIORITY)
1. **get_statcast_single_game(game_id) - PRIMARY source for pitch-level data \
and contact quality analysis**
2. get_game_highlight_data(game_id) - Get video highlights and notable plays

STEP 4: KEY PLAYER PERFORMANCES
- Identify standout hitting and pitching performances from boxscore
- **Use Statcast single-game data to analyze:**
  - Exit velocity on key hits and contact quality
  - Pitch velocity and movement effectiveness
  - Barrel rate and hard-hit rate for both teams
  - Clutch moment performance with Statcast context

STEP 5: DISCOVER TRADITIONAL STAT CONTEXT (if needed)
**Only if traditional leaderboard context is needed:**
- get_meta(type_name="leagueLeaderTypes") - Get valid categories for context
- get_meta(type_name="statGroups") - Ensure proper statGroup usage

STEP 6: GENERATE COMPREHENSIVE GAME RECAP ARTIFACT
Create detailed markdown game analysis with:

# Game Recap: {team1} @ {team2}
## Final Score: [Score] • {date} • Statcast Analysis

### Game Summary
- Final score and key turning points
- Winning/losing pitcher decisions
- Attendance and game duration
- Weather and field conditions

### Statcast Game Highlights
**Contact Quality Standouts**
- Hardest hit ball: [Player] - [Exit Velocity] mph, [Distance] ft
- Fastest pitch: [Pitcher] - [Velocity] mph [Pitch Type]
- Best barrel: [Player] - [Exit Velocity] mph at [Launch Angle]°
- Team hard-hit rate comparison
- Team average exit velocity comparison

### Offensive Performance Analysis
**{team1} Offense**
- Key hits and contact quality breakdown
- Clutch performance with Statcast context
- Exit velocity distribution and quality of contact
- Situational hitting effectiveness

**{team2} Offense**
- Key hits and contact quality breakdown
- Clutch performance with Statcast context
- Exit velocity distribution and quality of contact
- Situational hitting effectiveness

### Pitching Performance Analysis
**Starting Pitching**
- Velocity profiles and movement effectiveness
- Contact management and exit velocity allowed
- Pitch mix effectiveness and whiff rates
- Command and control metrics

**Bullpen Usage**
- High-leverage situation performance
- Velocity and stuff effectiveness
- Contact quality allowed analysis

### Turning Point Analysis
- Key moments that changed game momentum
- Statcast context for crucial plays
- Managerial decisions and their impact
- Defensive plays and baserunning impact

### Player Spotlight Performances
**Offensive Stars**
- Best individual hitting performances with Statcast metrics
- Clutch hit analysis with exit velocity context

**Pitching Stars**
- Dominant pitching performances with velocity/movement data
- Strikeout analysis with pitch effectiveness

### Post-Game Implications
- Impact on standings and playoff races
- Notable achievements or milestones
- Injury concerns or roster implications
- Series implications if applicable

### Scouting Takeaways
**What We Learned**
- Player development observations
- Tactical insights and adjustments
- Performance validation or concerns
- Matchup insights for future reference

**CRITICAL GUIDELINES:**
- **Statcast single-game data is the primary source** for all contact and pitch quality analysis
- Traditional boxscore provides game flow and basic context
- **Focus on exit velocity, barrel rate, and pitch velocity** as key performance indicators
- Structure like a professional post-game scouting report with insights for future reference"""


def cd_statistical_deep_dive(
    stat_category: str,
    season: Optional[int] = None,
    min_qualifier: Optional[int] = None,
) -> str:
    """
    Generate an in-depth statistical analysis report as an artifact for Claude Desktop.

    Args:
        stat_category: Statistical category to analyze (e.g., "home_runs", "era", "steals")
        season: Season to analyze (current if not specified)
        min_qualifier: Minimum qualifying threshold

    Returns:
        Prompt for comprehensive statistical analysis report
    """
    season_text = f"{season}" if season else "current season"

    return f"""Create a comprehensive statistical deep dive analysis for {stat_category} in the
{season_text}. Deliver as a detailed markdown artifact structured as an analytical scouting report.

STEP 1: DISCOVER VALID PARAMETERS
**CRITICAL: Before using any traditional MLB Stats API functions:**
- get_meta(type_name="leagueLeaderTypes") - \
Get valid leaderCategories (ensure {stat_category} is valid)
- get_meta(type_name="statGroups") - Get valid statGroups (REQUIRED for accurate results)
- get_meta(type_name="gameTypes") - Get valid gameTypes if analysis needs filtering
- get_meta(type_name="statTypes") - Get valid statTypes for comprehensive understanding

**Note**: Always include appropriate statGroup when using get_league_leader_data().

STEP 2: GATHER LEAGUE-WIDE DATA
1. **get_league_leader_data("{stat_category}", season={season or "current"}, statGroup="[APPROPRIATE_GROUP]", limit=50)**
2. Get appropriate team-level data using get_team_batting() or get_team_pitching()
3. **Primary analysis using Statcast data:**
   - get_statcast_batter_expected_stats({season or "current_year"}) for batting statistics
   - get_statcast_pitcher_expected_stats({season or "current_year"}) for pitching statistics
   - \
**get_statcast_batter_exitvelo_barrels({season or "current_year"}) if stat_category relates to power/contact**
   - \
**get_statcast_pitcher_exitvelo_barrels({season or "current_year"}) if stat_category relates to contact allowed**

STEP 3: IDENTIFY KEY INSIGHTS
- Current leaders and their performance levels
- **Statcast context**: How does traditional stat relate to exit velocity, expected stats, etc.
- Historical context and trends
- Team-by-team breakdowns
- **Expected vs Actual performance** using Statcast expected statistics

STEP 4: ADVANCED ANALYSIS (STATCAST PRIORITY)
- **Correlation with Statcast metrics** (exit velocity, barrel rate, expected stats)
- Impact on team success and predictive value
- **Quality of contact analysis** for offensive statistics
- **Contact management analysis** for pitching statistics

STEP 5: GENERATE COMPREHENSIVE STATISTICAL ANALYSIS ARTIFACT
Create detailed markdown analysis report:

# Statistical Deep Dive: {stat_category.title()}
## {season_text.title()} League Analysis • Advanced Metrics Perspective

### Executive Summary
- Key findings and trends in {stat_category}
- How Statcast metrics correlate with traditional performance
- Predictive insights and implications

### League Leaders Analysis
**Top 10 {stat_category.title()} Leaders**
- Traditional leaderboard with Statcast context
- Expected vs actual performance differential
- Contact quality metrics for top performers
- Sustainability analysis based on underlying metrics

### Statcast Context for {stat_category.title()}
**Correlation Analysis**
- How {stat_category} correlates with:
  - Average Exit Velocity
  - Hard Hit Rate (95+ mph)
  - Barrel Rate and Sweet Spot Rate
  - Expected Statistics (xwOBA, xBA, xSLG)
  - Contact quality measures

### Expected vs Actual Performance
**Over-Performers** (Actual > Expected)
- Players exceeding expected performance based on contact quality
- Sustainability concerns and regression candidates
- Potential explanations for outperformance

**Under-Performers** (Expected > Actual)
- Players with strong underlying metrics but poor results
- Potential breakout candidates
- Luck and timing factors

### Team-Level Analysis
**Team Rankings in {stat_category.title()}**
- How teams rank in traditional stat vs underlying metrics
- Organizational approaches and trends
- Impact on team success and correlation with wins

### Historical Context
- How {season_text} compares to recent seasons
- Trends in the statistic over time
- Impact of rule changes or environmental factors

### Predictive Insights
**Rest-of-Season Projections**
- Players likely to improve based on underlying metrics
- Regression candidates with unsustainable performance
- Breakout candidates with strong Statcast profiles

### Key Takeaways for Scouts and Analysts
**What Statcast Tells Us About {stat_category.title()}**
- Most important underlying metrics for predicting {stat_category}
- Players to watch based on advanced metrics
- Market inefficiencies and undervalued performers

**Actionable Insights**
- Player development focus areas
- Talent evaluation considerations
- Strategic implications for team building

**CRITICAL REQUIREMENTS:**
- **Always verify parameter validity** using get_meta() before API calls
- **Prioritize Statcast analysis** over traditional statistics alone
- **Include statGroup parameter** in all get_league_leader_data() calls
- **Focus on contact quality metrics** for comprehensive evaluation
- **Use expected statistics** to identify over/under-performers
- Structure as analytical scouting report with actionable insights"""


# ============================================================================
# WEB UI PROMPTS - STRUCTURED PLOT GENERATION
# ============================================================================


def web_ui_player_plots(player_name: str, season: Optional[int] = None) -> str:
    """
    Generate structured plot data for player analysis in Web UI format.

    Args:
        player_name: Full name or partial name of the player
        season: Season year (defaults to current season if not specified)

    Returns:
        Prompt for generating structured plot list for Web UI
    """
    season_text = f"the {season} season" if season else "the current/most recent season"

    return f"""Generate a comprehensive set of plots for {player_name} analysis for {season_text}.

Your response should be a list of plot objects in this exact JSON format:
[
  {{"plot_url": "<IMAGE_URL>", "plot_title": "<TITLE>"}},
  {{"plot_url": "<IMAGE_URL>", "plot_title": "<TITLE>"}},
  ...
]

STEP 1: CONSIDER RELEVANT PLOTS
For player analysis, determine which of these plots would be most relevant:
- create_strike_zone_plot: Strike zone heat maps for batting or pitching locations
- create_spraychart_plot: Batted ball spray charts for hit distribution
- create_bb_profile_plot: Distribution plots for exit velocity, launch angle, pitch velocity
- create_teams_plot: Team-level comparisons (less relevant for individual players)

STEP 2: ACQUIRE DATA
1. Use lookup_player("{player_name}") to find the player and get their MLB ID
2. Determine if player is primarily a batter or pitcher:
   - Use get_player_stats(player_id, group="hitting") for batting stats
   - Use get_player_stats(player_id, group="pitching") for pitching stats
3. Get Statcast data (PRIMARY SOURCE for plotting):
   - \
**get_statcast_batter_data(player_id{', start_dt="' + str(season) + '-01-01", end_dt="' + str(season) + '-12-31"' if season else ""}) for batters**
   - \
**get_statcast_pitcher_data(player_id{', start_dt="' + str(season) + '-01-01", end_dt="' + str(season) + '-12-31"' if season else ""}) for pitchers**

STEP 3: GENERATE PLOTS
Based on player type, create these specific plots:

**For BATTERS:**
1. create_strike_zone_plot(statcast_data, title="{player_name} Strike Zone Profile - \
{season_text}", colorby="events")
2. create_spraychart_plot(statcast_data, title="{player_name} Spray Chart - \
{season_text}", colorby="events")
3. create_bb_profile_plot(statcast_data, parameter="exit_velocity", title="{player_name} Exit Velocity Distribution")
4. create_bb_profile_plot(statcast_data, parameter="launch_angle", title="{player_name} Launch Angle Distribution")

**For PITCHERS:**
1. create_strike_zone_plot(statcast_data, title="{player_name} Pitch Locations - \
{season_text}", colorby="pitch_type")
2. create_bb_profile_plot(statcast_data, parameter="release_speed", title="{player_name} Pitch Velocity Distribution")
3. create_bb_profile_plot(statcast_data, parameter="exit_velocity", title="Exit Velocity Allowed - {player_name}")
4. create_spraychart_plot(statcast_data, title="Batted Balls Allowed - \
{player_name}", colorby="events")

**CRITICAL INSTRUCTIONS:**
- Each plotting function returns a response with image_url field containing a pre-signed S3 URL
- Extract the image_url value and use it as plot_url in your response
- Create descriptive, specific titles for each plot
- Only return the JSON list format specified above
- Do not include any explanatory text, only the JSON list of plot objects
- Ensure all plots use Statcast data for accuracy and detail"""


def web_ui_team_comparison_plots(
    team1: str, team2: str, focus_area: str = "overall"
) -> str:
    """
    Generate structured plot data for team comparison in Web UI format.

    Args:
        team1: First team abbreviation (e.g., "NYY", "BOS")
        team2: Second team abbreviation
        focus_area: Area to focus on ("overall", "hitting", "pitching", "recent")

    Returns:
        Prompt for generating structured plot list for Web UI
    """
    return f"""Generate a comprehensive set of comparison plots between {team1} and {team2} with focus on {focus_area}.

Your response should be a list of plot objects in this exact JSON format:
[
  {{"plot_url": "<IMAGE_URL>", "plot_title": "<TITLE>"}},
  {{"plot_url": "<IMAGE_URL>", "plot_title": "<TITLE>"}},
  ...
]

STEP 1: CONSIDER RELEVANT PLOTS
For team comparison focused on "{focus_area}", determine which plots would be most relevant:
- create_teams_plot: Primary tool for team-vs-team statistical comparisons
- create_bb_profile_plot: Team-level distributions (if aggregated data available)
- create_strike_zone_plot: Team approach comparisons (if aggregated data available)
- create_spraychart_plot: Team batted ball profiles (if aggregated data available)

STEP 2: ACQUIRE DATA
1. Get team data based on focus area:
   - get_team_batting(current_season) for hitting analysis
   - get_team_pitching(current_season) for pitching analysis
   - get_standings() for current records and context
2. Get Statcast team-level data:
   - get_statcast_batter_exitvelo_barrels(current_year) for hitting metrics
   - get_statcast_pitcher_exitvelo_barrels(current_year) for pitching metrics
3. Filter data for {team1} and {team2} specifically

STEP 3: GENERATE PLOTS
Based on focus area "{focus_area}", create relevant comparison plots:

**If focus_area is "hitting" or "overall":**
1. create_teams_plot(team_data, x_axis="runs_per_game", y_axis="ops", title="{team1} vs {team2} - Offensive Production")
2. create_teams_plot(team_data, x_axis="avg_exit_velocity", y_axis="hard_hit_rate", title="{team1} vs {team2} - Contact Quality (Statcast)")
3. create_teams_plot(team_data, x_axis="barrel_rate", y_axis="avg_distance", title="{team1} vs {team2} - Power Metrics (Statcast)")

**If focus_area is "pitching" or "overall":**
1. create_teams_plot(team_data, x_axis="era", y_axis="whip", title="{team1} vs {team2} - \
Pitching Performance")
2. create_teams_plot(team_data, x_axis="avg_exit_velocity_allowed", y_axis="hard_hit_rate_allowed", title="{team1} vs {team2} - Contact Management (Statcast)")
3. create_teams_plot(team_data, x_axis="strikeout_rate", y_axis="walk_rate", title="{team1} vs {team2} - Command Metrics")

**If focus_area is "recent":**
1. Get recent game data with get_schedule() for last 15 games
2. create_teams_plot(recent_data, x_axis="runs_scored_recent", y_axis="runs_allowed_recent", title="{team1} vs {team2} - Recent Form")
3. create_teams_plot(recent_data, x_axis="wins_last_10", y_axis="run_differential_last_10", title="{team1} vs {team2} - Last 10 Games")

**CRITICAL INSTRUCTIONS:**
- Each plotting function returns a response with image_url field containing a pre-signed S3 URL
- Extract the image_url value and use it as plot_url in your response
- Create descriptive titles that clearly indicate which teams are being compared
- Use appropriate statistical measures for the x_axis and y_axis parameters
- Only return the JSON list format specified above
- Do not include any explanatory text, only the JSON list of plot objects
- Prioritize Statcast metrics for most meaningful comparisons"""


def web_ui_game_analysis_plots(date: str, team1: str, team2: str) -> str:
    """
    Generate structured plot data for game analysis in Web UI format.

    Args:
        date: Game date in YYYY-MM-DD format
        team1: First team abbreviation (e.g., "NYY", "BOS")
        team2: Second team abbreviation

    Returns:
        Prompt for generating structured plot list for Web UI
    """
    return f"""Generate a comprehensive set of plots for the game between \
{team1} and {team2} on {date}.

Your response should be a list of plot objects in this exact JSON format:
[
  {{"plot_url": "<IMAGE_URL>", "plot_title": "<TITLE>"}},
  {{"plot_url": "<IMAGE_URL>", "plot_title": "<TITLE>"}},
  ...
]

STEP 1: FIND THE GAME
1. Use get_schedule(date="{date}", team_id=None) to get all games on {date}
2. Filter the schedule results to find the game between {team1} and \
{team2}
3. Extract the game_id (or game_pk) from the matching game
4. If no game found between these teams on this date, inform the user and \
suggest checking the date or team abbreviations

STEP 2: CONSIDER RELEVANT PLOTS
For game analysis, determine which plots would be most relevant:
- create_strike_zone_plot: Pitcher locations and batter approach for the game
- create_spraychart_plot: All batted balls from the game with outcomes
- create_bb_profile_plot: Game-specific distributions (exit velocity, launch angle, pitch velocity)
- create_teams_plot: Team performance comparison for this specific game

STEP 3: ACQUIRE DATA
1. Get basic game information:
   - get_boxscore(game_id) for final score and basic stats
   - get_linescore(game_id) for inning-by-inning breakdown
2. Get detailed game data:
   - **get_statcast_single_game(game_id) - PRIMARY source for all plotting data**
   - get_game_scoring_play_data(game_id) for key moments context

STEP 4: GENERATE PLOTS
Create these specific plots using the single-game Statcast data:

1. create_spraychart_plot(statcast_game_data, title="{team1} @ {team2} - \
All Batted Balls • {date}", colorby="events")
2. create_strike_zone_plot(statcast_game_data, title="{team1} @ {team2} - \
Pitch Locations • {date}", colorby="pitch_type")
3. create_bb_profile_plot(statcast_game_data, parameter="exit_velocity", title="{team1} @ {team2} - Exit Velocity Distribution • {date}")
4. create_bb_profile_plot(statcast_game_data, parameter="release_speed", title="{team1} @ {team2} - Pitch Velocity Distribution • {date}")
5. create_strike_zone_plot(statcast_game_data, title="{team1} @ {team2} - \
Strike Zone by Outcome • {date}", colorby="events")

**CRITICAL INSTRUCTIONS:**
- Each plotting function returns a response with image_url field containing a pre-signed S3 URL
- Extract the image_url value and use it as plot_url in your response
- Create descriptive titles that include the teams and date for context
- Only return the JSON list format specified above
- Do not include any explanatory text, only the JSON list of plot objects
- Use get_statcast_single_game() as the primary data source for all plots"""


def web_ui_statistical_analysis_plots(
    stat_category: str,
    season: Optional[int] = None,
    min_qualifier: Optional[int] = None,
) -> str:
    """
    Generate structured plot data for statistical analysis in Web UI format.

    Args:
        stat_category: Statistical category to analyze (e.g., "home_runs", "era", "steals")
        season: Season to analyze (current if not specified)
        min_qualifier: Minimum qualifying threshold

    Returns:
        Prompt for generating structured plot list for Web UI
    """
    season_text = f"{season}" if season else "current season"

    return f"""Generate a comprehensive set of plots for {stat_category} statistical analysis in the {season_text}.

Your response should be a list of plot objects in this exact JSON format:
[
  {{"plot_url": "<IMAGE_URL>", "plot_title": "<TITLE>"}},
  {{"plot_url": "<IMAGE_URL>", "plot_title": "<TITLE>"}},
  ...
]

STEP 1: CONSIDER RELEVANT PLOTS
For statistical analysis of {stat_category}, determine which plots would be most relevant:
- create_teams_plot: League-wide team comparisons in the statistical category
- create_bb_profile_plot: Distribution analysis of the statistic across players
- create_strike_zone_plot: If stat relates to plate discipline or pitch location
- create_spraychart_plot: If stat relates to batted ball outcomes

STEP 2: ACQUIRE DATA
1. Discover valid parameters:
   - get_meta(type_name="leagueLeaderTypes") to ensure {stat_category} is valid
   - get_meta(type_name="statGroups") to get proper statGroup for API calls
2. Get league-wide data:
   - \
**get_league_leader_data("{stat_category}", season={season or "current"}, statGroup="[APPROPRIATE_GROUP]", limit=50)**
3. Get supporting Statcast data:
   - get_statcast_batter_expected_stats({season or "current_year"}) for batting statistics
   - get_statcast_pitcher_expected_stats({season or "current_year"}) for pitching statistics
   - get_statcast_batter_exitvelo_barrels({season or "current_year"}) if power-related
   - get_statcast_pitcher_exitvelo_barrels({season or "current_year"}) if contact-related
4. Get team-level data:
   - get_team_batting(current_season) or get_team_pitching(current_season) as appropriate

STEP 3: GENERATE PLOTS
Create these analysis plots based on the statistical category:

**For Offensive Statistics (home_runs, batting_average, etc.):**
1. create_teams_plot(team_data, x_axis="{stat_category}", y_axis="wins", title="Team {stat_category.title()} vs Wins - {season_text}")
2. create_teams_plot(league_data, x_axis="{stat_category}", y_axis="avg_exit_velocity", title="{stat_category.title()} vs Exit Velocity (Statcast)")
3. create_bb_profile_plot(statcast_data, parameter="exit_velocity", title="Exit Velocity Distribution - Top {stat_category.title()} Leaders")
4. create_teams_plot(expected_data, x_axis="expected_{stat_category}", y_axis="actual_{stat_category}", title="Expected vs Actual {stat_category.title()} - {season_text}")

**For Pitching Statistics (era, strikeouts, etc.):**
1. create_teams_plot(team_data, x_axis="{stat_category}", y_axis="wins", title="Team {stat_category.title()} vs Wins - {season_text}")
2. create_teams_plot(league_data, x_axis="{stat_category}", y_axis="avg_exit_velocity_allowed", title="{stat_category.title()} vs Exit Velocity Allowed (Statcast)")
3. create_bb_profile_plot(statcast_data, parameter="release_speed", title="Pitch Velocity Distribution - Top {stat_category.title()} Leaders")
4. create_teams_plot(expected_data, x_axis="expected_{stat_category}", y_axis="actual_{stat_category}", title="Expected vs Actual {stat_category.title()} - {season_text}")

**For Advanced/Statcast Statistics:**
1. create_teams_plot(statcast_data, x_axis="{stat_category}", y_axis="barrel_rate", title="{stat_category.title()} vs Barrel Rate - {season_text}")
2. create_bb_profile_plot(statcast_data, parameter="{stat_category}", title="{stat_category.title()} Distribution Across League")
3. create_teams_plot(team_data, x_axis="team_{stat_category}", y_axis="run_differential", title="Team {stat_category.title()} vs Run Differential")

**CRITICAL INSTRUCTIONS:**
- Each plotting function returns a response with image_url field containing a pre-signed S3 URL
- Extract the image_url value and use it as plot_url in your response
- Create descriptive titles that clearly indicate the statistical focus and time period
- Always use get_meta() to validate parameters before making API calls
- Include statGroup parameter when using get_league_leader_data()
- Only return the JSON list format specified above
- Do not include any explanatory text, only the JSON list of plot objects
- Prioritize Statcast correlations and expected vs actual performance analysis"""


def web_ui_season_trends_plots(
    focus_area: str = "league", season: Optional[int] = None
) -> str:
    """
    Generate structured plot data for season trend analysis in Web UI format.

    Args:
        focus_area: Area to analyze ("league", "team_rankings", "individual_leaders")
        season: Season to analyze (current if not specified)

    Returns:
        Prompt for generating structured plot list for Web UI
    """
    season_text = f"{season}" if season else "current season"

    return f"""Generate a comprehensive set of plots for {focus_area} trend analysis in the {season_text}.

Your response should be a list of plot objects in this exact JSON format:
[
  {{"plot_url": "<IMAGE_URL>", "plot_title": "<TITLE>"}},
  {{"plot_url": "<IMAGE_URL>", "plot_title": "<TITLE>"}},
  ...
]

STEP 1: CONSIDER RELEVANT PLOTS
For {focus_area} trend analysis, determine which plots would be most relevant:
- create_teams_plot: Primary tool for league-wide and team comparison analysis
- create_bb_profile_plot: Distribution analysis across the league
- create_strike_zone_plot: League trends in plate discipline (if applicable)
- create_spraychart_plot: League trends in batted ball profiles (if applicable)

STEP 2: ACQUIRE DATA
Based on focus_area "{focus_area}":

**If "league":**
- get_team_batting(current_season) for all teams
- get_team_pitching(current_season) for all teams
- get_statcast_batter_exitvelo_barrels({season or "current_year"}) for league contact trends
- get_statcast_pitcher_exitvelo_barrels({season or "current_year"}) for league pitching trends
- get_standings() for context

**If "team_rankings":**
- get_team_batting(current_season) and get_team_pitching(current_season)
- get_standings() for win-loss context
- get_statcast_batter_expected_stats({season or "current_year"})
- get_statcast_pitcher_expected_stats({season or "current_year"})

**If "individual_leaders":**
- get_meta(type_name="leagueLeaderTypes") to find valid categories
- get_league_leader_data() for multiple key statistical categories with proper statGroup
- get_statcast_batter_percentile_ranks({season or "current_year"})
- get_statcast_pitcher_percentile_ranks({season or "current_year"})

STEP 3: GENERATE PLOTS
Create trend analysis plots based on focus area:

**For "league" focus:**
1. create_teams_plot(team_data, x_axis="runs_per_game", y_axis="era", title="League Offensive vs Pitching Balance - {season_text}")
2. create_teams_plot(statcast_data, x_axis="avg_exit_velocity", y_axis="barrel_rate", title="League Contact Quality Trends (Statcast) - {season_text}")
3. create_bb_profile_plot(league_statcast_data, parameter="exit_velocity", title="League-Wide Exit Velocity Distribution - {season_text}")
4. create_teams_plot(team_data, x_axis="home_runs", y_axis="strikeouts", title="League Power vs Strikeout Trends - {season_text}")

**For "team_rankings" focus:**
1. create_teams_plot(team_data, x_axis="runs_scored", y_axis="runs_allowed", title="Team Run Differential Landscape - {season_text}")
2. create_teams_plot(statcast_data, x_axis="expected_wins", y_axis="actual_wins", title="Expected vs Actual Wins (Statcast) - {season_text}")
3. create_teams_plot(team_data, x_axis="ops", y_axis="era", title="Team OPS vs ERA Rankings - {season_text}")
4. create_teams_plot(team_data, x_axis="hard_hit_rate", y_axis="hard_hit_rate_allowed", title="Team Contact Quality Matrix (Statcast) - {season_text}")

**For "individual_leaders" focus:**
1. create_teams_plot(leader_data, x_axis="home_runs", y_axis="avg_exit_velocity", title="HR Leaders vs Exit Velocity (Statcast) - {season_text}")
2. create_teams_plot(pitcher_data, x_axis="strikeouts", y_axis="avg_exit_velocity_allowed", title="Strikeout Leaders vs Contact Management - {season_text}")
3. create_bb_profile_plot(top_hitters_data, parameter="exit_velocity", title="Exit Velocity Distribution - Top Hitters {season_text}")
4. create_teams_plot(expected_data, x_axis="woba", y_axis="xwoba", title="wOBA vs xwOBA - \
Top Performers {season_text}")

**CRITICAL INSTRUCTIONS:**
- Each plotting function returns a response with image_url field containing a pre-signed S3 URL
- Extract the image_url value and use it as plot_url in your response
- Create descriptive titles that clearly indicate the trend focus and time period
- Always use get_meta() to validate parameters before making API calls
- Only return the JSON list format specified above
- Do not include any explanatory text, only the JSON list of plot objects
- Emphasize Statcast metrics and expected performance analysis for modern insights"""
