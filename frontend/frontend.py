import json
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image

## PAGE CONFIGURATION
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Attribution de prêt",
    page_icon="🟢",
)


# Titre de l'application
st.markdown(
    "<h1 style='text-align: center; color: black;'> Attribution de prêt </h1>",
    unsafe_allow_html=True,
)


# créer la sidebar avec image
st.sidebar.header("Prédiction de prêt automatique")
image = Image.open("./static/img.png")
st.sidebar.image(image, use_column_width=True)
st.sidebar.info(
    "Cette application web est créee pour prédire le fait qu'un client soit à risque ou sûr pour l'attribution d'un prêt"
)
st.sidebar.success("Conçue par Jessy PATRICE ❤️")

st.markdown(
    """
            <p style="color:Gray;font-family:'Roboto Condensed';">
            
            Ce Dashboard est destiné aux chargés de relation client afin de les aider dans leur prise de décision d'un accord de prêt. 
            L'objectif était de développer un modèle de scoring de la probabilité de défaut de paiement d'un client pour étayer la décision d'accorder ou non un prêt à un client potentiel en s'appuyant sur des sources de données variées (données comportementales, familiales, données provenant d'autres institutions financières, ...).
            
            Il répond au besoin de transparence des vis-à-vis des décisions d'attribution de crédit. </p>
            """,
    unsafe_allow_html=True,
)

# First section (row)
st.markdown("#### 1 - Que voulez vous afficher ?")


# Make page
option = st.selectbox(
    "Choisir une action:",
    (
        "---",
        "Décision d'attribution d'un prêt",
        "les graphiques de l'importance globale",
    ),
)


# Saisie
if option == "Décision d'attribution d'un prêt":

    st.markdown("#### 2 - Saisir un numéro client pour la prédiction")

    client_id_input = st.text_input("Indiquez le numéro du client:", value="")

    inputs = {"numero_client": client_id_input}

    # clients_response = requests.get(url="http://localhost:8000/database")
    clients_response = requests.get(
        url="https://mybackendapp.azurewebsites.net/database"
    )
    all_clients = json.loads(clients_response.text)
    all_clients = pd.DataFrame(all_clients)
    all_clients = all_clients["values"].to_list()

    # bouton prédire
    st.caption("NB: Il s'agit d'un numéro à 6 chiffres")
    st.markdown("#### 3 - Exécuter la prédiction")
    if st.button("Prédire"):
        if client_id_input == "" or client_id_input.isalpha():
            st.error("Saisir un numéro client")

        elif int(client_id_input) not in all_clients:
            st.error("Veuillez saisir un numéro client valide svp", icon="🚨")

        else:
            # CONVOQUER L'ENDPOINT DE LA PREDICTION
            # pred = requests.post(url="http://localhost:8000/predict", params=inputs)
            pred = requests.post(
                url="https://mybackendapp.azurewebsites.net/predict", params=inputs
            )

            # Turn JSON to dict data then grab prediction
            pred_str_format = json.loads(pred.text)
            pred_str_format = pred_str_format["probabilite"]

            # Logiques pour la prédiction
            if pred_str_format == "Prêt accordé, client sûr":
                st.success(pred_str_format, icon="✅")
                st.write("Il s'agit de la décision automatique obtenue en fonction du croisement de plusieurs variables détaillées ci-dessous.")

                # CONVOQUER L'ENDPOINT DE LA FEATURE IMPORTANCE
                # importance = requests.post(url="http://localhost:8000/importance_locale", params=inputs)
                importance = requests.post(
                    url="https://mybackendapp.azurewebsites.net/importance_locale",
                    params=inputs,
                )

                json_data = json.loads(importance.text)
                plot = pd.read_html(json_data["data"])
                plot_importance = plot[0]

                st.write(
                    "<h3 style='text-align: center; color: black;'> Importance locale des variables qui ont contribué à la prédiction pour ce client </h3>",
                    unsafe_allow_html=True,
                )
                # Utilisation de plotly
                fig = px.bar(
                    plot_importance, x="Feature", y="Contribution?", title="", color = "Contribution?",
                    color_continuous_scale=["red", "green"]
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    "<h6 style='text-align: center; color: black;'> Figure 1 : Importance des variables sur la prédiction </h6>",
                    unsafe_allow_html=True,
                )

                # Affichage du dataframe de l'importance locale
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(
                    "Vous pouvez filtrer et agrandir en plein écran le tableau ci-dessous pour connaître en détail l'importance de chaque variable sur la prédiction effectuée pour ce client"
                )
                st.dataframe(plot_importance.iloc[:, :2], use_container_width=True)
                st.markdown(
                    "<h6 style='text-align: center; color: black;'> Tableau 1 : Tableau de l'importance locale des variables sur la prédiction </h6>",
                    unsafe_allow_html=True,
                )

            elif pred_str_format == "Prêt non accordé, risque de défaut":
                st.warning(pred_str_format, icon="⚠️")
                st.write("Il s'agit de la décision automatique obtenue en fonction du croisement de plusieurs variables détaillées ci-dessous.")
                

                # CONVOQUER L'ENDPOINT DE LA FEATURE IMPORTANCE
                # importance = requests.post(url="http://localhost:8000/importance_locale", params=inputs)
                importance = requests.post(
                    url="https://mybackendapp.azurewebsites.net/importance_locale",
                    params=inputs,
                )

                json_data = json.loads(importance.text)
                plot = pd.read_html(json_data["data"])
                plot_importance = plot[0]

                st.write(
                    "<h3 style='text-align: center; color: black;'> Importance locale des variables qui ont contribué à la prédiction pour ce client</h3>",
                    unsafe_allow_html=True,
                )
                # Utilisation de plotly
                fig = px.bar(plot_importance, x="Feature", y="Contribution?", title="", color = "Contribution?",
                    color_continuous_scale=["red", "green"])
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    "<h6 style='text-align: center; color: black;'> Figure 1 : Affichage graphique de l'importance locale des variables sur la prédiction </h6>",
                    unsafe_allow_html=True,
                )

                # Affichage du dataframe de l'importance locale
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(
                    "Vous pouvez filtrer et agrandir en plein écran le tableau ci-dessous pour connaître en détail l'importance de chaque variable sur la prédiction effectuée pour ce client"
                )
                st.dataframe(plot_importance.iloc[:, :2], use_container_width=True)
                st.markdown(
                    "<h6 style='text-align: center; color: black;'> Tableau 1: Affichage de l'importance locale des variables sur la prédiction </h6>",
                    unsafe_allow_html=True,
                )

            else:
                st.error("Veuillez saisir un numéro client valide", icon="🚨")


if option == "les graphiques de l'importance globale":
    # globale = requests.get(url='http://localhost:8000/globale_importance')
    globale = requests.get(
        url="https://mybackendapp.azurewebsites.net/globale_importance"
    )

    json_data = json.loads(globale.text)
    plot = pd.read_html(json_data["data"])
    plot_importance = plot[0]

    # Feature importance

    st.write(
        "<h3 style='text-align: center; color: black;'> Importance globale des variables qui ont contribué à la prédiction </h3>",
        unsafe_allow_html=True,
    )

    fig = px.bar(plot_importance, x="Feature", y="Weight", color = "Weight",
                    color_continuous_scale=["red", "green"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<h6 style='text-align: center; color: black;'> Figure 2 : Affichage graphique de l'importance globale des variables sur la prédiction </h6>",
        unsafe_allow_html=True,
    )
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(
        "Vous pouvez filtrer et agrandir en plein écran le tableau ci-dessous pour connaître en détail l'importance de chaque variable"
    )
    st.dataframe(plot_importance, use_container_width=True)
    
    st.markdown(
        "<h6 style='text-align: center; color: black;'> Tableau 2 : Tableau de l'importance globale des variables sur la prédiction </h6>",
        unsafe_allow_html=True,
    )

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'Fait par Jessy PATRICE avec Streamlit'; 
                visibility: visible;
                display: block;
                text-align: center;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# streamlit run myapp.py
