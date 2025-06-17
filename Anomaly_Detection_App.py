from streamlit_cropper import st_cropper # type: ignore
import streamlit as st # type: ignore
from streamlit_drawable_canvas import st_canvas # type: ignore
from rapidfuzz import fuzz
import pandas as pd
from datetime import timedelta
import numpy as np
import random
import os
from ollama import Client

st.set_page_config(page_title="Sensor Analyst Chatbot")
# os.write(1,b'Something was executed.\n')
print("starting")

df_operator_logs = pd.read_pickle('Final_Operator_Log.pkl')
print("read pickles 1")

df_anomalies = pd.read_pickle('Final_Anomaly_Data.pkl')
print("read pickles 2")

# Ensure datetime format
df_anomalies['Timestamp'] = pd.to_datetime(df_anomalies['Timestamp'])
df_operator_logs['Timestamp'] = pd.to_datetime(df_operator_logs['Timestamp'])

# Matching function
def fuzzy_match_logs(anomaly_df, log_df, time_window=30, score_threshold=80):
    matched_logs = []

    for _, anomaly_row in anomaly_df.iterrows():
        a_time = anomaly_row['Timestamp']
        a_equip = anomaly_row['Equipment_ID']

        # Filter logs within time window
        nearby_logs = log_df[
            (log_df['Timestamp'] >= a_time - timedelta(minutes=time_window)) &
            (log_df['Timestamp'] <= a_time + timedelta(minutes=time_window))
        ]

        # Fuzzy match equipment ID to log text
        best_score = 0
        best_log = None
        for _, log_row in nearby_logs.iterrows():
            score = fuzz.partial_ratio(a_equip.lower(), log_row['Operator_Log'].lower())
            if score > best_score and score >= score_threshold:
                best_score = score
                best_log = log_row['Operator_Log']

        matched_logs.append(best_log)

    anomaly_df = anomaly_df.copy()
    anomaly_df['Matched_Log'] = matched_logs
    return anomaly_df

# Apply matching
df_anomaly_with_fuzzy_logs = fuzzy_match_logs(df_anomalies[df_anomalies['Timestamp'] >= '2025-05-24 00:00:00'], df_operator_logs[df_operator_logs['Timestamp'] >= '2025-05-24 00:00:00'])

equipment_profiles = {
    'Mud_Pump': {
        'Temperature': (65, 10, 25, 90, 15, 95, 5),
        'Pressure': (5000, 500, 3000, 7000, 1000, 9000,400),
        'Vibration': (4, 1.5, 1, 10, 2,15,2.5),
        'Flow_rate': (2000, 300, 1000, 3000, 750,4000,150),
        'Current': (70, 10, 50, 90, 30,100,5),
        'Offshore': 'binary'
    },
    'Centrifugal_Pump': {
        'Temperature': (50, 6, 35, 75, 15,90,5),
        'Pressure': (3500, 600, 2000, 6000, 900,8500,300),
        'Vibration': (3, 1, 0.5, 8, 0.7,10,1.5 ),
        'Current': (60, 8, 40, 85, 25,100,10),
        'Offshore': 'binary'
    },
    'Seawater_Lift_Pump': {
        'Temperature': (30, 5, 10, 45, 5,60,4),
        'Vibration': (2.5, 0.8, 0.5, 5, 0,8,0.05),
        'Fluid_level': (70, 10, 20, 100, 10,180,10),
        'Gas-Detection': (5, 10, 5, 50, 0,80,7),
        'Offshore': 'binary'
    }
}

def generate_sensor_prompt(sensor_type, value, equipment_id):
    sensor_type = sensor_type.strip()

    # Extract equipment base type from ID (e.g., 'Mud_Pump' from 'Mud_Pump-1')
    base_type = equipment_id.split("-")[0]

    # Get sensor config for this equipment type
    sensor_config = equipment_profiles.get(base_type, {})
    sensor_params = sensor_config.get(sensor_type)

    if not sensor_params or sensor_params == 'binary':
        return f"No thresholds defined for sensor '{sensor_type}' in '{base_type}'."

    # Unpack the thresholds (as per your tuple structure)
    mean, std, min_val, max_val, mean_low, mean_high, outlier_delta = sensor_params

    # Define warning and anomaly bands
    low_warn, high_warn = min_val, max_val
    low_anom, high_anom = min_val - min_val*0.2, max_val + max_val*0.2

    # Determine severity level
    if value >= high_anom:
        level = "ANOMALY"
        message = f"{sensor_type} reading of {value} is too high for {equipment_id}, possible anomaly. Should not exceed {high_anom}."
    elif value >= high_warn:
        level = "WARNING"
        message = f"{sensor_type} reading of {value} is above normal range for {equipment_id}. Should typically stay below {high_warn}."
    elif value <= low_anom:
        level = "ANOMALY"
        message = f"{sensor_type} reading of {value} is too low for {equipment_id}, possible anomaly. Should not go below {low_anom}."
    elif value <= low_warn:
        level = "WARNING"
        message = f"{sensor_type} reading of {value} is lower than expected for {equipment_id}. Should typically stay above {low_warn}."
    else:
        level = "Normal"
        message = f"{sensor_type} reading of {value} is within normal range for {equipment_id}."

    return f"""
Status: {level}  
Insight: {message}
""".strip()


# Customize as per real-world sensor ranges
thresholds = {
    'Temperature': {'warning': (60, 80), 'anomaly': (80, 100)},
    'Pressure': {'warning': (30, 50), 'anomaly': (50, 70)},
    'Vibration': {'warning': (5, 7), 'anomaly': (7, 10)},
    'Flow_rate': {'warning': (100, 150), 'anomaly': (150, 200)},
    'Force': {'warning': (500, 700), 'anomaly': (700, 1000)},
    'Fluid_level': {'warning': (30, 60), 'anomaly': (60, 90)},
    'Current': {'warning': (10, 15), 'anomaly': (15, 20)},
}


st.title('Anomaly Detection in Oil Rig Operations')
st.subheader('A system that detects anomalies in oil rig operations')

Insight_Generator,View_Data = st.tabs(['Insight Generator','View Sensor Data'])

with Insight_Generator:
    equipment = st.selectbox(
        'Select the Type of the Equipment',
        ('Mud Pump', 'Motor', 'Seawater Lift Pump', 'Centrifugal Pump'))

    st.write('You selected:', equipment)

    if equipment == 'Mud Pump':
        equipment_id = st.selectbox(
            'Select the ID of the Equipment: ',
            (
                'Mud_Pump-1', 'Mud_Pump-2', 'Mud_Pump-3', 'Mud_Pump-4'
            )
        )
    elif equipment == 'Seawater Lift Pump':
        equipment_id = st.selectbox(
            'Select the ID of the Equipment: ',
            (
                'Seawater_Lift_Pump-2', 'Seawater_Lift_Pump-3', 'Seawater_Lift_Pump-4', 'Seawater_Lift_Pump-5'
            )
        )
    else:
        equipment_id = st.selectbox(
            'Select the ID of the Equipment: ',
            (
                'Centrifugal_Pump-1', 'Centrifugal_Pump-2', 'Centrifugal_Pump-3'
            )
        )


    st.write('You selected:', equipment_id)

    if equipment == 'Mud Pump' or equipment == 'Centrifugal Pump':
        sensor = st.selectbox(
            'Select the Type of the Sensor: ',
            (
                'Pressure', 'Flow Rate', 'Current', 'Temperature', 'Vibration'
            )
        )
    else:
        sensor = st.selectbox(
            'Select the Type of the Sensor: ',
            (
                'Pressure', 'Flow Rate', 'Current'
            )
        )
    st.write('You have selected: ',sensor)

    temp_value = st.text_input(f"Enter the {sensor} reading:",-1)


    st.write((temp_value))
    if temp_value != '-1': 
        temp_value = int(temp_value)
        st.write(generate_sensor_prompt(sensor,temp_value,equipment_id))
    
    df = pd.read_pickle('RAG_INPUT.PKL')
    # 👉 Replace with your actual context generation (e.g. from df)
    @st.cache_data
    def get_context():
        # Or load preprocessed chunks
        return df.to_markdown(index=False)  # convert full or partial df to markdown

    # 💬 Generate response with Ollama
    def query_llm(context, question, model="mistral"):
        client = Client()
        prompt = f"""
    You are a helpful analyst. Use the following oil rig sensor data to answer the question.

    Sensor Data:
    {context}

    Question: {question}

    Answer:"""
        response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

    # --- Streamlit App UI ---



    # Cache or re-use preloaded DataFrame context
    context = get_context()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.chat_input("Ask something about the sensor data...")

    if user_query:
        with st.spinner("Analyzing..."):
            response = query_llm(context, user_query)
            st.session_state.chat_history.append(("user", user_query))
            st.session_state.chat_history.append(("ai", response))

    # Show conversation
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)


with View_Data:
    st.subheader("Sensors having Anomalies")
    st.write(df_anomalies.head(10))
    st.subheader("Operator Logs")
    st.write(df_operator_logs.head(10))
    st.subheader("Correlated Anomalies")
    st.write(df_anomaly_with_fuzzy_logs[(df_anomaly_with_fuzzy_logs['Matched_Log'].isnull()==False) & (df_anomaly_with_fuzzy_logs['is_anomaly']== True)][['Timestamp','Equipment_ID','Current','Temperature','Pressure','Flow_rate', 'Offshore', 'Vibration','Matched_Log', 'is_anomaly']])
