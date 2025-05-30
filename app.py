import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load dataset and model
match_data = pd.read_csv('assets/dataset_matches.csv', low_memory=False)
match_data['MatchDate'] = pd.to_datetime(match_data['MatchDate'], errors='coerce')
match_data['HomeTeam'] = match_data['HomeTeam'].str.upper()
match_data['AwayTeam'] = match_data['AwayTeam'].str.upper()

model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler.joblib')

# Prepare sorted list of unique teams
teams = sorted(set(match_data['HomeTeam'].unique()) | set(match_data['AwayTeam'].unique()))


def get_latest_features(df, team_name, is_home=True):
    """
    Gets the most recent features for the team, either home or away.
    """
    prefix = 'Home' if is_home else 'Away'
    team_col = 'HomeTeam' if is_home else 'AwayTeam'

    team_matches = df[df[team_col] == team_name].sort_values('MatchDate', ascending=False)
    if team_matches.empty:
        # If no data is found, return zeros for all features
        return {f"{prefix}Elo": 0, f"{prefix}Form3": 0, f"{prefix}OddHome": 0, f"{prefix}OddDraw": 0, f"{prefix}OddAway": 0,
                f"{prefix}Shots": 0, f"{prefix}Target": 0, f"{prefix}Fouls": 0, f"{prefix}Yellow": 0, f"{prefix}Red": 0}

    last_match = team_matches.iloc[0]

    # List of features to extract (make sure they match exactly with dataset column names)
    features = {}
    features[f"{prefix}Elo"] = last_match.get(f"{prefix}Elo", 0)
    features[f"{prefix}Form3"] = last_match.get(f"{prefix}Form3", 0)
    # For odds, we use three separate odds
    features[f"{prefix}OddHome"] = last_match.get(f"{prefix}OddHome", 0)
    features[f"{prefix}OddDraw"] = last_match.get(f"{prefix}OddDraw", 0)
    features[f"{prefix}OddAway"] = last_match.get(f"{prefix}OddAway", 0)

    features[f"{prefix}Shots"] = last_match.get(f"{prefix}Shots", 0)
    features[f"{prefix}Target"] = last_match.get(f"{prefix}Target", 0)
    features[f"{prefix}Fouls"] = last_match.get(f"{prefix}Fouls", 0)
    features[f"{prefix}Yellow"] = last_match.get(f"{prefix}Yellow", 0)
    features[f"{prefix}Red"] = last_match.get(f"{prefix}Red", 0)

    return features


def encode_prediction(pred, home_team, away_team):
    """
    Converts numerical prediction to readable string with team names
    """
    mapping = {
        1: f"{home_team} wins",
        0: "Draw",
        -1: f"{away_team} wins"
    }
    return mapping.get(pred, "Unknown")


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        home_team = request.form['home_team'].upper()
        away_team = request.form['away_team'].upper()
        match_date = request.form['match_date']  # can be used for info, but not used in the model currently

        # Get real features from dataset
        home_feats = get_latest_features(match_data, home_team, is_home=True)
        away_feats = get_latest_features(match_data, away_team, is_home=False)

        # Feature order as in training set
        feature_vector = [
            home_feats.get('HomeElo', 0),
            away_feats.get('AwayElo', 0),
            home_feats.get('HomeForm3', 0),
            away_feats.get('AwayForm3', 0),
            home_feats.get('HomeOddHome', 0),
            home_feats.get('HomeOddDraw', 0),
            home_feats.get('HomeOddAway', 0),
            home_feats.get('HomeShots', 0),
            away_feats.get('AwayShots', 0),
            home_feats.get('HomeTarget', 0),
            away_feats.get('AwayTarget', 0),
            home_feats.get('HomeFouls', 0),
            away_feats.get('AwayFouls', 0),
            home_feats.get('HomeYellow', 0),
            away_feats.get('AwayYellow', 0),
            home_feats.get('HomeRed', 0),
            away_feats.get('AwayRed', 0),
        ]

        # Scale and predict
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = scaler.transform(feature_vector)

        pred = model.predict(feature_vector_scaled)[0]
        prediction = encode_prediction(pred, home_team, away_team)

        return jsonify({'prediction': prediction})

    return render_template('index.html', teams=teams)

if __name__ == "__main__":
    app.run(debug=True)
