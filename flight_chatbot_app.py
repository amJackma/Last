import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load the trained Random Forest model (dict with model and threshold)
MODEL_PATH = r"D:\Last\saved_models\random_forest_delay\random_forest_delay_vwith_threshold_info.joblib"
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle['model']
threshold = model_bundle['threshold']

# Carrier codes
CARRIER_OPTIONS = ['AA', 'DL', 'UA', 'WN', 'AS', 'B6', 'F9', 'G4', 'HA', 'NK', 'SY', 'VX']

# Airport codes (all unique ORIGIN)
AIRPORT_CODES = ['ABE', 'ABQ', 'ABY', 'ACY', 'AEX', 'AGS', 'ALB', 'ANC', 'ASE', 'ATL', 'ATW', 'AUS', 'AVL', 'AVP', 'BDL', 'BHM', 'BMI', 'BNA', 'BOS', 'BQK', 'BTR', 'BTV', 'BUF', 'BWI', 'BZN', 'CAE', 'CAK', 'CHA', 'CHO', 'CHS', 'CID', 'CLE', 'CLT', 'CMH', 'COS', 'CRW', 'CSG', 'CVG', 'DAB', 'DAL', 'DAY', 'DCA', 'DEN', 'DFW', 'DHN', 'DSM', 'DTW', 'ECP', 'EGE', 'ELM', 'ELP', 'EVV', 'EWN', 'EWR', 'EYW', 'FAR', 'FAY', 'FCA', 'FLL', 'FNT', 'FSD', 'FSM', 'FWA', 'GNV', 'GPT', 'GRB', 'GRK', 'GRR', 'GSO', 'GSP', 'GTR', 'HDN', 'HNL', 'HOU', 'HPN', 'HSV', 'IAD', 'IAH', 'ICT', 'ILM', 'IND', 'ISP', 'JAC', 'JAN', 'JAX', 'JFK', 'LAS', 'LAX', 'LCK', 'LEX', 'LFT', 'LGA', 'LIT', 'LNK', 'MCI', 'MCO', 'MDT', 'MDW', 'MEM', 'MGM', 'MHT', 'MIA', 'MKE', 'MLB', 'MLI', 'MLU', 'MOB', 'MSN', 'MSO', 'MSP', 'MSY', 'MTJ', 'MYR', 'OAJ', 'OAK', 'OKC', 'OMA', 'ORD', 'ORF', 'PBI', 'PDX', 'PHF', 'PHL', 'PHX', 'PIA', 'PIT', 'PNS', 'PVD', 'PWM', 'RAP', 'RDU', 'RIC', 'RNO', 'ROA', 'ROC', 'RST', 'RSW', 'SAN', 'SAT', 'SAV', 'SBN', 'SDF', 'SEA', 'SFO', 'SGF', 'SHV', 'SJC', 'SJU', 'SLC', 'SMF', 'SNA', 'SRQ', 'STL', 'STT', 'STX', 'SYR', 'TLH', 'TPA', 'TRI', 'TTN', 'TUL', 'TUS', 'TVC', 'TYS', 'VLD', 'VPS', 'XNA']

def calculate_distance(origin, dest):
    """Placeholder for distance calculation."""
    distances = {
        ('ATL', 'LAX'): 1946,
        ('LAX', 'ATL'): 1946,
        ('ORD', 'JFK'): 740,
        ('JFK', 'ORD'): 740,
    }
    return distances.get((origin.upper(), dest.upper()), 1000)

def preprocess_input(data):
    """Convert user data into a DataFrame for the model."""
    df = pd.DataFrame([data])
    df['Quarter'] = ((df['Month'] - 1) // 3) + 1
    year = df['Year'].iloc[0]
    month = df['Month'].iloc[0]
    day = df['DayofMonth'].iloc[0]
    df['DayOfWeek'] = datetime(year, month, day).weekday() + 1
    
    df = df.rename(columns={
        'Year': 'YEAR',
        'Month': 'MONTH',
        'DayofMonth': 'DAY_OF_MONTH',
        'DayOfWeek': 'DAY_OF_WEEK',
        'UNIQUE_CARRIER': 'UNIQUE_CARRIER',
        'ORIGIN': 'ORIGIN',
        'DEST': 'DEST',
        'DEP_TIME': 'DEP_TIME',
        'DISTANCE': 'DISTANCE'
    })
    
    missing_cols = [
        'TIME_HIST_DELAY_MEAN', 'CARRIER_ROUTE_HIST_DELAY_MEAN', 'SEASON', 
        'CARRIER_HIST_FLIGHT_COUNT', 'TIME_HIST_DELAY_MEDIAN', 'ROUTE_HIST_DELAY_MEDIAN', 
        'IS_WEEKEND', 'TIME_HIST_DELAY_STD', 'HOUR', 'CARRIER_ROUTE_HIST_DELAY_MEDIAN', 
        'ROUTE_FREQUENCY', 'CARRIER_ROUTE_HIST_DELAY_STD', 'AIR_TIME', 
        'CARRIER_HIST_DELAY_MEDIAN', 'DEST_HIST_DELAY_STD', 'DEST_HIST_DELAY_MEDIAN', 
        'ORIGIN_HIST_DELAY_MEDIAN', 'TIME_HIST_FLIGHT_COUNT', 'ROUTE_HIST_DELAY_MEAN', 
        'ROUTE', 'ARR_TIME', 'CARRIER_HIST_DELAY_MEAN', 'ORIGIN_HIST_FLIGHT_COUNT', 
        'ORIGIN_HIST_DELAY_MEAN', 'CARRIER_HIST_DELAY_STD', 'CARRIER_ROUTE_HIST_FLIGHT_COUNT', 
        'TIME_PERIOD', 'ORIGIN_HIST_DELAY_STD', 'ROUTE_HIST_FLIGHT_COUNT', 
        'IS_HOLIDAY_SEASON', 'DEST_HIST_FLIGHT_COUNT', 'ROUTE_HIST_DELAY_STD', 
        'DEST_HIST_DELAY_MEAN'
    ]
    for col in missing_cols:
        df[col] = 0
    
    df['ROUTE'] = df['ORIGIN'] + '_' + df['DEST']
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([6, 7]).astype(int)
    df['HOUR'] = df['DEP_TIME'] // 100
    df['SEASON'] = df['MONTH'].map({12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall'})
    df['TIME_PERIOD'] = pd.cut(df['HOUR'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    df['IS_HOLIDAY_SEASON'] = df['MONTH'].isin([11, 12, 1]).astype(int)
    
    return df

def predict_delay(features_df):
    prediction_proba = model.predict_proba(features_df)[0][1]
    prediction = 1 if prediction_proba >= threshold else 0
    return prediction, prediction_proba

# Streamlit app
st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="wide")

st.title("✈️ Flight Delay Prediction Chatbot")
st.markdown("Enter your flight details to predict if it might be delayed.")

with st.sidebar:
    st.header("Flight Details")
    
    year = st.selectbox("Year", [2025, 2024, 2023], index=0)
    month = st.slider("Month", 1, 12, 1)
    day = st.slider("Day of Month", 1, 31, 1)
    dep_time = st.number_input("Departure Time (HHMM)", min_value=0, max_value=2359, value=1200, step=1)
    carrier = st.selectbox("Airline Carrier", CARRIER_OPTIONS)
    origin = st.selectbox("Origin Airport", AIRPORT_CODES)
    dest = st.selectbox("Destination Airport", AIRPORT_CODES)
    
    predict_button = st.button("Predict Delay")

if predict_button:
    distance = calculate_distance(origin, dest)
    flight_data = {
        'Year': year,
        'Month': month,
        'DayofMonth': day,
        'UNIQUE_CARRIER': carrier.upper(),
        'ORIGIN': origin.upper(),
        'DEST': dest.upper(),
        'DEP_TIME': dep_time,
        'DISTANCE': distance
    }
    
    features_df = preprocess_input(flight_data)
    prediction, proba = predict_delay(features_df)
    
    if prediction == 1:
        st.error(f"🚨 Your flight from {origin} to {dest} is likely delayed (probability: {proba:.2f}).")
        st.write("**Recommendations:** Consider rescheduling, choosing a different carrier, or an alternative route.")
    else:
        st.success(f"✅ Your flight is predicted to be on time (probability of delay: {proba:.2f}). Enjoy your trip!")
    
    st.write(f"**Flight Details:** {carrier} from {origin} to {dest} on {month}/{day}/{year} at {dep_time}.")

st.markdown("---")
st.markdown("Built with Streamlit | Model: Random Forest")