import streamlit as st

st.set_page_config(
    page_title="App0",
    page_icon="👋",
)

st.write("# Welcome to App0 INGAR Experimental Home 👋")

st.sidebar.success("Select an App to visualize.")

st.markdown(
    """
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### See other complex demos from streamlit
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)