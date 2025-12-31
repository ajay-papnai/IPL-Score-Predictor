import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def train_model():
    df = pd.read_csv("ipl_data.csv")

    df = df.drop(columns=[
        'mid', 'date', 'batsman', 'bowler',
        'striker', 'non-striker'
    ])
    df.dropna(inplace=True)

    bat_encoder = LabelEncoder()
    bowl_encoder = LabelEncoder()
    venue_encoder = LabelEncoder()

    df['bat_team'] = bat_encoder.fit_transform(df['bat_team'])
    df['bowl_team'] = bowl_encoder.fit_transform(df['bowl_team'])
    df['venue'] = venue_encoder.fit_transform(df['venue'])

    X = df[['bat_team', 'bowl_team', 'venue',
            'overs', 'runs', 'wickets',
            'runs_last_5', 'wickets_last_5']]

    y = df['total']

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    return model, bat_encoder, bowl_encoder, venue_encoder
