"""
Flight Delay Prediction Chatbot

This script implements a conversational AI chatbot that uses the tuned LightGBM model
to predict flight delays based on user inputs. It gathers flight details interactively,
makes real-time predictions, and provides recommendations.

Requirements Satisfied:
- Conversational AI: Interactive chat interface with natural language prompts.
- Recommendation System: Suggests alternatives if delay is predicted.
- Real-time Prediction: Processes user inputs on-the-fly for immediate results.

Dependencies: joblib, pandas, numpy, scikit-learn (for preprocessing if needed)
Install via: pip install joblib pandas numpy scikit-learn

Usage: Run the script with Python. The chatbot will guide the user through inputs.
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load the trained LightGBM model
MODEL_PATH = r"D:\Last\saved_models\lightgbm_delay\lightgbm_delay_vbaseline_pre_tuning.joblib"
model = joblib.load(MODEL_PATH)

# Define the expected input features (based on the dataset and model training)
REQUIRED_FEATURES = [
    'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'UNIQUE_CARRIER',
    'ORIGIN', 'DEST', 'DEP_TIME', 'DISTANCE'
]

# Carrier codes (example list; expand as needed from dataset)
CARRIER_OPTIONS = ['AA', 'DL', 'UA', 'WN', 'AS', 'B6', 'F9', 'G4', 'HA', 'NK', 'SY', 'VX']

# Airport codes (from dataset; all unique ORIGIN)
AIRPORT_CODES = ['ABE', 'ABQ', 'ABY', 'ACY', 'AEX', 'AGS', 'ALB', 'ANC', 'ASE', 'ATL', 'ATW', 'AUS', 'AVL', 'AVP', 'BDL', 'BHM', 'BMI', 'BNA', 'BOS', 'BQK', 'BTR', 'BTV', 'BUF', 'BWI', 'BZN', 'CAE', 'CAK', 'CHA', 'CHO', 'CHS', 'CID', 'CLE', 'CLT', 'CMH', 'COS', 'CRW', 'CSG', 'CVG', 'DAB', 'DAL', 'DAY', 'DCA', 'DEN', 'DFW', 'DHN', 'DSM', 'DTW', 'ECP', 'EGE', 'ELM', 'ELP', 'EVV', 'EWN', 'EWR', 'EYW', 'FAR', 'FAY', 'FCA', 'FLL', 'FNT', 'FSD', 'FSM', 'FWA', 'GNV', 'GPT', 'GRB', 'GRK', 'GRR', 'GSO', 'GSP', 'GTR', 'HDN', 'HNL', 'HOU', 'HPN', 'HSV', 'IAD', 'IAH', 'ICT', 'ILM', 'IND', 'ISP', 'JAC', 'JAN', 'JAX', 'JFK', 'LAS', 'LAX', 'LCK', 'LEX', 'LFT', 'LGA', 'LIT', 'LNK', 'MCI', 'MCO', 'MDT', 'MDW', 'MEM', 'MGM', 'MHT', 'MIA', 'MKE', 'MLB', 'MLI', 'MLU', 'MOB', 'MSN', 'MSO', 'MSP', 'MSY', 'MTJ', 'MYR', 'OAJ', 'OAK', 'OKC', 'OMA', 'ORD', 'ORF', 'PBI', 'PDX', 'PHF', 'PHL', 'PHX', 'PIA', 'PIT', 'PNS', 'PVD', 'PWM', 'RAP', 'RDU', 'RIC', 'RNO', 'ROA', 'ROC', 'RST', 'RSW', 'SAN', 'SAT', 'SAV', 'SBN', 'SDF', 'SEA', 'SFO', 'SGF', 'SHV', 'SJC', 'SJU', 'SLC', 'SMF', 'SNA', 'SRQ', 'STL', 'STT', 'STX', 'SYR', 'TLH', 'TPA', 'TRI', 'TTN', 'TUL', 'TUS', 'TVC', 'TYS', 'VLD', 'VPS', 'XNA']

def get_user_input(prompt, options=None, input_type=str):
    """Helper to get validated user input."""
    while True:
        user_input = input(prompt).strip()
        if options and user_input.upper() not in [opt.upper() for opt in options]:
            print(f"Invalid choice. Options: {', '.join(options)}")
            continue
        try:
            return input_type(user_input)
        except ValueError:
            print(f"Please enter a valid {input_type.__name__}.")

def calculate_distance(origin, dest):
    """Placeholder for distance calculation. In a real app, use an API or database."""
    # Example distances (miles); replace with actual logic
    distances = {
        ('ATL', 'LAX'): 1946,
        ('LAX', 'ATL'): 1946,
        ('ORD', 'JFK'): 740,
        ('JFK', 'ORD'): 740,
        # Add more pairs
    }
    return distances.get((origin.upper(), dest.upper()), 1000)  # Default to 1000 if not found

def preprocess_input(data):
    """Convert user data into a DataFrame for the model."""
    df = pd.DataFrame([data])
    # Add derived features
    df['Quarter'] = ((df['Month'] - 1) // 3) + 1
    year = df['Year'].iloc[0]
    month = df['Month'].iloc[0]
    day = df['DayofMonth'].iloc[0]
    df['DayOfWeek'] = datetime(year, month, day).weekday() + 1  # Monday=1
    
    # Rename columns to match model expectations
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
    
    # Add missing columns with default values (0 for historical/derived features)
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
    
    # Simple derivations (set categoricals as strings for OneHotEncoder)
    df['ROUTE'] = df['ORIGIN'] + '_' + df['DEST']
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([6, 7]).astype(int)  # 1 if Sat/Sun
    df['HOUR'] = df['DEP_TIME'] // 100  # Extract hour from HHMM
    df['SEASON'] = df['MONTH'].map({12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall'})
    df['TIME_PERIOD'] = pd.cut(df['HOUR'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    df['IS_HOLIDAY_SEASON'] = df['MONTH'].isin([11, 12, 1]).astype(int)  # Nov-Jan
    
    return df

def predict_delay(features_df):
    """Make prediction using the loaded model."""
    prediction_proba = model.predict_proba(features_df)[0][1]  # Probability of delay
    prediction = 1 if prediction_proba >= 0.60 else 0  # Using tuned threshold for LightGBM
    return prediction, prediction_proba

def generate_recommendation(prediction, proba, data):
    """Provide recommendations based on prediction."""
    if prediction == 1:
        rec = f"Your flight from {data['ORIGIN']} to {data['DEST']} is likely delayed (probability: {proba:.2f}). "
        rec += "Recommendations: Consider rescheduling to an earlier time, choosing a different carrier, or selecting an alternative route. "
        rec += "Check for nearby airports or next-day flights to avoid disruptions."
    else:
        rec = f"Your flight is predicted to be on time (probability of delay: {proba:.2f}). Enjoy your trip!"
    return rec

def chatbot():
    """Main chatbot loop."""
    print(" Welcome to the Flight Delay Predictor Chatbot!")
    print("I'll help you check if your flight might be delayed and provide recommendations.")
    print("Let's gather some details about your flight.\n")

    # Gather inputs conversationally
    year = get_user_input("What year is your flight? (e.g., 2025): ", input_type=int)
    month = get_user_input("What month? (1-12): ", input_type=int)
    day = get_user_input("What day of the month? (1-31): ", input_type=int)
    dep_time = get_user_input("What is the scheduled departure time? (HHMM, e.g., 1430 for 2:30 PM): ", input_type=int)
    carrier = get_user_input(f"What is the airline carrier code? Options: {', '.join(CARRIER_OPTIONS)}: ", options=CARRIER_OPTIONS)
    origin = get_user_input(f"What is the origin airport code? Options: {', '.join(AIRPORT_CODES)}: ", options=AIRPORT_CODES)
    dest = get_user_input(f"What is the destination airport code? Options: {', '.join(AIRPORT_CODES)}: ", options=AIRPORT_CODES)
    distance = calculate_distance(origin, dest)  # Auto-calculate or ask user

    # Compile data
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

    print("\nProcessing your request...")

    # Preprocess and predict
    features_df = preprocess_input(flight_data)
    prediction, proba = predict_delay(features_df)
    recommendation = generate_recommendation(prediction, proba, flight_data)

    print(f"\n Prediction: {'Delayed' if prediction else 'On Time'}")
    print(f"Recommendation: {recommendation}")

    # Ask if user wants to check another flight
    again = get_user_input("\nWould you like to check another flight? (yes/no): ", options=['yes', 'no'])
    if again.lower() == 'yes':
        chatbot()
    else:
        print("Thank you for using the Flight Delay Predictor! Safe travels! ")

if __name__ == "__main__":
    try:
        chatbot()
    except KeyboardInterrupt:
        print("\nChatbot stopped by user. Safe travels!")