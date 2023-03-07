import streamlit as st
import pandas as pd
import random
import time

# Set the number of machines to monitor
num_machines = 5

# Define function to generate random machine states
def generate_states(num_machines):
    states = []
    for i in range(num_machines):
        state = random.choice(["OK", "Warning", "Error"])
        states.append(state)
    return states

# Define function to generate random machine alerts
def generate_alerts(num_machines):
    alerts = []
    for i in range(num_machines):
        alert = random.choice(["None", "Low", "Medium", "High"])
        alerts.append(alert)
    return alerts

# Generate initial machine states and alerts
states = generate_states(num_machines)
alerts = generate_alerts(num_machines)

# Define color scheme for states and alerts
state_colors = {"OK": "green", "Warning": "orange", "Error": "red"}
alert_colors = {"None": "green", "Low": "yellow", "Medium": "orange", "High": "red"}

# Display machine states and alerts in a table with colors
st.write("Machine States and Alerts:")
df = pd.DataFrame({"State": states, "Alert": alerts})
df["State"] = df["State"].apply(lambda x: f'<span style="color: {state_colors[x]}">{x}</span>')
df["Alert"] = df["Alert"].apply(lambda x: f'<span style="color: {alert_colors[x]}">{x}</span>')
st.write(df, unsafe_allow_html=True)

# Update machine states and alerts every second
while True:
    states = generate_states(num_machines)
    alerts = generate_alerts(num_machines)
    df = pd.DataFrame({"State": states, "Alert": alerts})
    df["State"] = df["State"].apply(lambda x: f'<span style="color: {state_colors[x]}">{x}</span>')
    df["Alert"] = df["Alert"].apply(lambda x: f'<span style="color: {alert_colors[x]}">{x}</span>')
    st.write(df, unsafe_allow_html=True)
    st.write("---")
    time.sleep(1)
