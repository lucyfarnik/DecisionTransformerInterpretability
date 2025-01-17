import pandas as pd
import streamlit as st

from .constants import (
    IDX_TO_ACTION,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    IDX_TO_STATE,
    STATE_TO_IDX,
)


def help_page():
    with st.expander("Basic help"):
        st.markdown(
            """

            #### Welcome to the Decision Transformer Interpretability App!

            A brief tour of the app:

            - The left side bar contains hyperparameters that you can use to configure the agent and analyses.
            - At the top of the screen, you can see the current game state and the model's preferences over actions. 
            - Underneath the game state, you can see whichever analyses are currently selected.

            - Use the arrow keys to move the agent.
                - The agent will move forward if you press the up key.
                - The agent will rotate left if you press the left key.
                - The agent will rotate right if you press the right key.
                - The agent will pickup if you press p.
                - The agent will drop if you press d. 
                - The agent will toggle if you press d.
                - The agent will trigger the done action if you press shift-D.
            - Reward
                - The agent receives a positive reward if it reaches the goal.
                - The agent receives a negative reward if it hits a wall or obstacle.
                - Please press reset if either of these happens as the RTG will be incorrect if you terminate the episode and keep playing.
            - Use the sidebar hyperparameters to configure the agent
                - Use the RTG hyperparameter to select the RTG you want to use. This will determine whether the Decision Transformer will simulate a trajectory that achieves high or low reward.
            - Attribution
                - Many of the analysis features will depend on the attribution configuration.
                - You can set it to logit difference or to single logit.
            - Use the sidebar to select the analysis you want to see.
                - Static analyses interpret the agents weights and don't involve any forward passes.
                - Dynamic analyses interpret the agents activations and therefore may involve one or more forward passes.
                - Causal analyses involve interventions in the forward pass.
            - Reload the model to start a new trajectory.
            - Click on the trajectory details to see the trajectory details.
            - Please use *dark* mode as I haven't made all the plots look good in light mode yet.

            See analysis help for more details and references for each analytical method.
            """
        )


def analysis_help():
    with st.expander("Analysis Help"):
        st.markdown(
            """
            # Analysis Help
            
            *please note that the app is under active development and may not always include the latest features*.
            
            Analyses appear under the main game screen after being selected in the sidebar.

            - Static analyses interpret the agents weights and don't involve any forward passes.
            - Dynamic analyses interpret the agents activations and therefore may involve one or more forward passes.
            - Causal analyses involve interventions in the forward pass.

            
            ## Static Analyses

            - RTG Embeddings: This shows the dot product of the RTG embedding onto the defined direction. The dot product with any action direction or logit is a linear function of RTG since the embedding is linear.

            - Time Embeddings: This shows the dot product of the time embedding onto the defined direction. It is not linear since it is a learned embedding.

            - QK Circuit: For each head in each layer, we show components of the state embedding increase attention paid to the RTG token from the state token. I interpret higher values in the resulting channels/positions as indicating that activation in those regions of the observation increases the attention paid to the previous RTG (key) by the state token (query).

            $$ QK_{Circuit}=W_{E(state)}^TW_Q^TW_KW_{E(RTG)} $$

            - OV Circuit: For each head, in each layer and each output action we show how components of the state embedding increase the logit of that action. I interpret higher values in the resulting channels/positions as indicating that activation in the observation regions increases/decreases that logit via the residual stream.

            $$ OV_{Circuit}=W_{U(Action)}W_OW_VW_{E(State)} $$

            ## Dynamic Analyses

            - Residual Stream Contributions: This shows the dot product of the components of the transformer with the selected direction in directional analysis. Attention Bias should be minimal and is an artefact of the transformer architecture, which is not meaningful. The input tokens are decomposed further in the observation view.

            - Observation View: An amalgam of visualisations relating to the input tokens, you can choose to show the state input channels as the transformer “sees” them, the weights associated with each in the state encoding and the position-wise/channel-wise projection of the current state embedding activations into the direction selected. Lastly, a bar chart shows the sum of the projections from each channel (remember distributivity!). Histograms of weight/activation projection magnitudes are also provided.

            - Attention View: Not super useful with a context length of two, but this panel shows a traditional attention map. Attention paid to RTG varies with the initial RTG, which is better visualised in the RTG scan.

            - RTG Scan: The RTG scan analysis shows the output logits, component contributions to the selected direction, and attention paid to the RTG token by the state token, as the RTG value is varied within the range selected. Timestep noise can be added to see how this relationship changes as a function of the set time, and correlations can be calculated between the component contributions.

            ## Causal Analyses

            - Ablation Experiment: The ablation experiment view enables you to run a counterfactual forward pass in which either attention head or
        """
        )


def reference_tables():
    with st.expander("Reference Tables"):
        st.markdown(
            """
            # Reference Tables
            """
        )
        a, b, c, d = st.columns(4)

        with a:
            st.markdown(
                """
                ## Object Space
                """
            )
            st.table(
                pd.DataFrame.from_dict(
                    IDX_TO_OBJECT, orient="index", columns=["Object"]
                )
            )

        with b:
            st.markdown(
                """
                ## Color Space
                """
            )
            st.table(
                pd.DataFrame.from_dict(
                    IDX_TO_COLOR, orient="index", columns=["Color"]
                )
            )
        with c:
            st.markdown(
                """
                ## State Space
                """
            )
            st.table(
                pd.DataFrame.from_dict(
                    IDX_TO_STATE, orient="index", columns=["State"]
                )
            )

        with d:
            st.markdown(
                """
                ## Action Space
                """
            )
            st.table(
                pd.DataFrame.from_dict(
                    IDX_TO_ACTION, orient="index", columns=["Action"]
                )
            )
