import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

def main() -> None:
    # Load match dataset with mixed types allowed
    match_data = pd.read_csv('assets/dataset_matches.csv', low_memory=False)

    # Clean and preprocess the dataset
    match_data = clean_match_data(match_data)

    # Create target variable: match_result (1 = home win, 0 = draw, -1 = away win)
    match_data['match_result'] = match_data.apply(get_result, axis=1)

    # Drop rows with missing target
    match_data.dropna(subset=['match_result'], inplace=True)

    # Select useful features
    features = [
        'HomeElo', 'AwayElo',
        'Form3Home', 'Form3Away',
        'OddHome', 'OddDraw', 'OddAway',
        'HomeShots', 'AwayShots',
        'HomeTarget', 'AwayTarget',
        'HomeFouls', 'AwayFouls',
        'HomeYellow', 'AwayYellow',
        'HomeRed', 'AwayRed'
    ]

    # Keep only numeric columns and drop rows with NaNs
    match_data = match_data[features + ['match_result']]
    match_data.dropna(inplace=True)

    # Define feature matrix X and target vector y
    X = match_data[features]
    y = match_data['match_result']

    # Optionally scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model and scaler
    joblib.dump(model, 'rf_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    print(f'Accuracy: {accuracy:.2%}')
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Optional: show class distribution
    print("\nTarget distribution:")
    print(y.value_counts(normalize=True))

def clean_match_data(match_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the match dataset: parse dates and uppercase team names.
    """
    match_data['MatchDate'] = pd.to_datetime(match_data['MatchDate'], errors='coerce')
    match_data['HomeTeam'] = match_data['HomeTeam'].str.upper()
    match_data['AwayTeam'] = match_data['AwayTeam'].str.upper()
    return match_data


def get_result(row: pd.Series) -> int:
    """
    Compute the match result:
    - 1 if home wins
    - 0 if draw
    - -1 if away wins
    """
    try:
        home_goals = int(row['FTHome'])
        away_goals = int(row['FTAway'])
    except:
        return None  # In case of bad data

    if home_goals > away_goals:
        return 1
    elif home_goals < away_goals:
        return -1
    else:
        return 0


# Entry point
if __name__ == "__main__":
    main()