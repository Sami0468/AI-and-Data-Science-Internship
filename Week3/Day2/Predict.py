from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MATCH_MODEL_PATH = MODEL_DIR / "match_winner_model.joblib"
PLAYER_MODEL_PATH = MODEL_DIR / "top_player_model.joblib"

match_model = joblib.load(MATCH_MODEL_PATH)
player_model = joblib.load(PLAYER_MODEL_PATH)


def validate_date(date):
    try:
        return pd.to_datetime(date)
    except Exception:
        raise ValueError("Invalid date format.")


def predict_match_winner(team_a, team_b, date):

    if not team_a or not team_b:
        raise ValueError("Both team names are required.")

    if team_a == team_b:
        raise ValueError("Team A and Team B must be different.")

    prediction_date = validate_date(date)

    input_data = pd.DataFrame({
        "team": [team_a],
        "opponent": [team_b],
        "home_away": ["home"],
        "match_date": [prediction_date]
    })

    for feature in match_model.feature_names_in_:
        if feature not in input_data.columns:
            input_data[feature] = np.nan

    input_data = input_data[
        match_model.feature_names_in_
    ]

    prediction = match_model.predict(input_data)[0]

    probabilities = match_model.predict_proba(input_data)[0]

    probability_dict = {
        str(cls): float(prob)
        for cls, prob in zip(
            match_model.classes_,
            probabilities
        )
    }

    if prediction == "home_win":
        winner = team_a
    elif prediction == "away_win":
        winner = team_b
    else:
        winner = "Draw"

    return {
        "winner": winner,
        "prediction": str(prediction),
        "probability": probability_dict,
        "date": str(prediction_date.date())
    }


def predict_top_player(
    player_data,
    team,
    opponent,
    stat_type="disposals"
):

    if player_data is None or player_data.empty:
        raise ValueError("Player data is empty.")

    if not team:
        raise ValueError("Team is required.")

    if stat_type != "disposals":
        raise ValueError(
            "Currently supported stat_type: disposals"
        )

    team_players = player_data[
        player_data["team"] == team
    ].copy()

    if team_players.empty:
        raise ValueError(
            f"No player data found for team: {team}"
        )

    required_features = [
        "year",
        "round",
        "career_game_count",
        "previous_disposals",
        "recent_disposals_avg",
        "previous_goals",
        "recent_goals_avg",
        "previous_tackles",
        "recent_fantasy_avg",
        "team",
        "opponent"
    ]

    missing = [
        col
        for col in required_features
        if col not in team_players.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    X = team_players[required_features].copy()

    predictions = player_model.predict(X)

    result = team_players[
        ["player_id", "team"]
    ].copy()

    result["predicted_disposals"] = predictions

    result = result.sort_values(
        "predicted_disposals",
        ascending=False
    )

    return result.reset_index(drop=True)