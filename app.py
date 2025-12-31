import streamlit as st
import pickle
import numpy as np

from train_model import train_model

model , bat_encoder, bowl_encoder, venue_encoder = train_model()

# Load model & encoders
# model = pickle.load(open("ipl_score_model.pkl", "rb"))
# bat_encoder = pickle.load(open("bat_team_encoder.pkl", "rb"))
# bowl_encoder = pickle.load(open("bowl_team_encoder.pkl", "rb"))
# venue_encoder = pickle.load(open("venue_encoder.pkl", "rb"))

st.set_page_config(page_title="IPL Score Predictor", layout="centered")

st.title("🏏 IPL First Innings Score Predictor")
st.write("Predict final score and score range based on match situation")

# Score range function
def score_range(score):
    if score < 150:
        return "Low Scoring (< 150)"
    elif score < 170:
        return "Average Scoring (150–169)"
    elif score < 190:
        return "Good Scoring (170–189)"
    else:
        return "High Scoring (190+)"

# Dropdowns
bat_team = st.selectbox("Batting Team", bat_encoder.classes_)
bowl_team = st.selectbox("Bowling Team", bowl_encoder.classes_)
venue = st.selectbox("Venue", venue_encoder.classes_)

# Inputs
overs = st.number_input("Completed Overs", min_value=0, max_value=20, value=10)
balls = st.slider("Balls in Current Over", 0, 5, 0)

total_overs = overs + balls / 6

runs = st.number_input("Current Runs", min_value=0, value=80)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=2)
runs_last_5 = st.number_input("Runs in Last 5 Overs", min_value=0, value=40)
wickets_last_5 = st.number_input("Wickets in Last 5 Overs", min_value=0, max_value=5, value=1)

# Predict
if st.button("Predict Score"):
    bat_enc = bat_encoder.transform([bat_team])[0]
    bowl_enc = bowl_encoder.transform([bowl_team])[0]
    venue_enc = venue_encoder.transform([venue])[0]

    input_data = np.array([[
    bat_enc,
    bowl_enc,
    venue_enc,
    total_overs,
    runs,
    wickets,
    runs_last_5,
    wickets_last_5
]])


    predicted_score = model.predict(input_data)[0]
    predicted_range = score_range(predicted_score)

    st.success(f"🏏 Predicted Final Score: **{int(predicted_score)} runs**")
    st.info(f"📊 Score Range: **{predicted_range}**")

    st.info(f"Based on current score of {runs} runs in {overs} overs with {wickets} wickets down.")