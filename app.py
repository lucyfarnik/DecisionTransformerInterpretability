import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import torch as t

from streamlit_app.visualizations import action_string_to_id
from streamlit_app.setup import initialize_playground
from streamlit_app.components import reset_button, show_attention_pattern, hyperpar_side_bar, record_keypresses
from streamlit_app.components import show_residual_stream_contributions, render_trajectory_details, render_game_screen
from streamlit_app.components import render_observation_view, show_ov_circuit, show_qk_circuit
start = time.time()

with st.sidebar:
    st.title("Decision Transformer Interpretability")

initial_rtg = hyperpar_side_bar()



model_path = "models/demo_model.pt"
action_string_to_id = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "drop": 4, "toggle": 5, "done": 6}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}

st.session_state.max_len = 1
env, dt = initialize_playground(model_path, initial_rtg)
x, cache, tokens = render_game_screen(dt, env)

with st.sidebar:
    st.subheader("Directional Analysis")
    comparing = st.checkbox("comparing directions", value=True)
    if comparing:
        positive_action_direction = st.selectbox("Positive Action Direction", ["left", "right", "forward", "pickup", "drop", "toggle", "done"], index = 2)
        negative_action_direction = st.selectbox("Negative Action Direction", ["left", "right", "forward", "pickup", "drop", "toggle", "done"], index = 1)
        positive_action_direction = action_string_to_id[positive_action_direction]
        negative_action_direction = action_string_to_id[negative_action_direction]
        logit_dir =  dt.predict_actions.weight[positive_action_direction] - dt.predict_actions.weight[negative_action_direction]
    else: 
        selected_action_direction = st.selectbox("Selected Action Direction", ["left", "right", "forward", "pickup", "drop", "toggle", "done"], index = 2)
        selected_action_direction = action_string_to_id[selected_action_direction]
        logit_dir = dt.predict_actions.weight[selected_action_direction]

    analyses =  st.multiselect("Select Analyses", ["Residual Stream Contributions", "Attention Pattern", "Observation View", "OV Circuit", "QK Circuit"])

if "Residual Stream Contributions" in analyses:
    show_residual_stream_contributions(dt, cache, logit_dir=logit_dir)
if "Attention Pattern" in analyses:
    show_attention_pattern(dt, cache)
if "Observation View" in analyses:
    render_observation_view(dt, env, tokens, logit_dir)
if "OV Circuit" in analyses:
    show_ov_circuit(dt)
if "QK Circuit" in analyses:
    show_qk_circuit(dt)


st.markdown("""---""")

st.session_state.env = env
st.session_state.dt = dt

with st.sidebar:
    render_trajectory_details()
    reset_button()
    end = time.time()
    st.write(f"Time taken: {end - start}")

record_keypresses()