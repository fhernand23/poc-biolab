import streamlit as st
import pandas as pd
import datetime
import numpy as np
import json
import altair as alt

f = open("pages/kiwi_db.json")
results = json.load(f)
f.close()

mbrs = len(results)
try:
    states = list(results[list(results.keys())[0]]["measurements_aggregated"].keys())
except:
    states = ["Glucose", "OD600", "Acetate", "DOT"]

params = ["qs", "qa_p_max", "qa_c_max", "Ks", "Kap", "Kac", "Ksi", "Kai", "Yas", "Yxsox", "Yxsof", "Yos", "Yoa", "Yxa", "qm", "cx", "cs", "ca"]

seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])


# # get json files
# f = open("dags/scripts/matlab/J_options.json")
# results = json.load(f)
# f.close()

# f = open("dags/scripts/matlab/J_simul.json")
# measures = json.load(f)
# f.close()

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('pages/kiwi_style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('KIWI-Experiment Monitoring')

st.sidebar.subheader('Minibioreactors (MBRs)')
selected_mbrs = st.sidebar.multiselect('Select MBRs', [i+1 for i in range(24)], [1])

st.sidebar.subheader('States')
selected_states = st.sidebar.multiselect('Select states', states, ["Glucose"])
plot_height = st.sidebar.slider('Specify plot height', 200, 500, 350)

st.sidebar.subheader('Model parameters')
selected_params = st.sidebar.selectbox('Select parameter', params)

st.sidebar.markdown('''---
Open [Airflow Platform](http://localhost:8082).''')

# Row 1
with st.container():
    st.markdown('### Metrics')
    col1, col2, col3 = st.columns(3)
    col1.metric("Start Time", datetime.datetime.now().strftime("%H:%M:%S"))
    col2.metric("MBRs", mbrs)
    col3.metric("Strain", "ABC123")

# Row 2
if len(selected_mbrs):
    cols = st.columns(len(selected_states))
    for index, container in enumerate(cols):
        with container:
            st.markdown(f'### {selected_states[index]}')

            # Concat differents MBRs states. Add ts as column and mbr label.
            pd_mbrs_states = []
            for mbr in selected_mbrs:
                pd_mbr = pd.DataFrame.from_dict(results[list(results.keys())[mbr - 1]]["measurements_aggregated"])
                pd_mbr["ts"] = pd_mbr.index.astype(float)
                pd_mbr["mbr"] = mbr
                pd_mbrs_states.append(pd_mbr)
            
            pd_state = pd.concat(pd_mbrs_states, ignore_index=True)
            # delete None values
            pd_state_wout_na = pd_state.dropna(axis=0, subset=selected_states[index])

            # plot series from MBR selector and State selector. Order x-axes 
            chart = alt.Chart(pd_state).mark_line(point=True).encode(
                x=alt.X('ts', sort=alt.EncodingSortField(field='ts', order='descending')),
                y=selected_states[index],
                color="mbr:N"
            ).interactive()           

            st.altair_chart(chart, use_container_width=True)


# Row 3
# st.markdown(f'### Parameter distribution "{selected_params}"')
# st.line_chart(seattle_weather, x = 'date', y = ["precipitation", "wind"], height = plot_height)
