# Football AI Predictor 🏆

An AI-powered football match prediction web application that uses machine learning to forecast match outcomes based on historical data and team statistics.

## Overview

Football AI Predictor uses a Random Forest classifier trained on extensive match data to predict the outcomes of football matches (home win, draw, or away win). The model analyzes various features including:

- Team ELO ratings
- Recent form (last 3 matches)
- Pre-match betting odds
- Match statistics (shots, fouls, cards)

The model achieves a 57% accuracy on three-class predictions, which is significant in football outcome prediction.

## Features

- **Simple Interface**: Select home and away teams, match date and time
- **Real-Time Predictions**: Get instant AI-powered match predictions
- **Data-Driven**: Uses real historical match data and statistics
- **Supported Leagues**:
  - Premier League
  - Serie A
  - La Liga
  - Bundesliga
  - Ligue 1
  - UEFA Champions League

## Technical Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, JavaScript
- **Machine Learning**: scikit-learn (RandomForestClassifier)
- **Data Processing**: pandas, numpy

## Installation

1. Clone the repository:
```sh
git clone [repository-url]
cd football-ai-predictor
```

2. Create and activate a virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```sh
pip install -r requirements.txt
```

4. Run the application:
```sh
python app.py
```

## Model Training

The model can be retrained using new data:

```sh
python train.py
```

This will:
- Load and preprocess the match dataset
- Train a new Random Forest model
- Save the model and scaler as `.joblib` files
- Output performance metrics

## Project Structure

```
├── app.py              # Flask application
├── train.py            # Model training script
├── requirements.txt    # Python dependencies
├── rf_model.joblib     # Trained model
├── scaler.joblib      # Feature scaler
├── assets/
│   ├── dataset_elo_ratings.csv
│   └── dataset_matches.csv
└── templates/
    └── index.html      # Web interface
```
