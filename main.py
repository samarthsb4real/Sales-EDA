import streamlit as st
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

#pip freeze > requirements.txt

with open('rfr.pkl', 'rb') as file:
    rfr = pickle.load(file)

rfc = joblib.load('rfc.joblib')

df_ad = pd.read_csv('Advertising.csv')

def main():

    st.header("EDA for Advertising.csv")
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(df_ad.iloc[:,1:].corr(), annot=True)
    plt.title("Heatmap")
    st.pyplot(plt)
    if st.button("Inferences on Advertising Heatmap"):
        st.write("Input inference here")

    plt.figure(figsize=(10, 5))
    plt.scatter(df_ad['Radio'], df_ad['Sales'])
    plt.title("Radio vs Sales")
    plt.xlabel("Radio")
    plt.ylabel("Sales")
    st.pyplot(plt)
    if st.button("Inferences on Radio vs Sales"):
        st.write("Input inference here")

    plt.figure(figsize=(10, 5))
    plt.scatter(df_ad['Newspaper'], df_ad['Sales'])
    plt.title("Newspaper vs Sales")
    plt.xlabel("Newspaper")
    plt.ylabel("Sales")
    st.pyplot(plt)
    if st.button("Inferences on Newspaper vs Sales"):
        st.write("Input inference here")

    plt.figure(figsize=(10, 5))
    plt.scatter(df_ad['TV'], df_ad['Sales'])
    plt.title("TV vs Sales")
    plt.xlabel("TV")
    plt.ylabel("Sales")
    st.pyplot(plt)
    if st.button("Inferences on TV vs Sales"):
        st.write("Input inference here")


    plt.figure(figsize=(10, 5))
    plt.bar(['TV', 'Newspaper', 'Radio'], df_ad[['TV', 'Newspaper', 'Radio']].mean())
    plt.title("Average Sales for TV, Newspaper, Radio")
    plt.ylabel("Sales")
    st.pyplot(plt)
    if st.button("Inferences on Average Sales for TV, Newspaper, Radio"):
        st.write("Input inference here")

    st.header("Sales Prediction")

    tv = st.slider("TV Sales", min_value = 0.7, max_value = 296.4, step = 0.1)
    radio = st.slider("Radio Sales", min_value = 0.0, max_value = 49.6, step = 0.1)
    newspaper = st.slider("Newspaper", min_value = 0.3, max_value = 114.4, step = 0.1)

    rfr_dict = {
        "TV": tv,
        "Radio": radio,
        "Newspaper": newspaper
    }

    rfr_test_df = pd.DataFrame(rfr_dict, index = [0])
    rfr_pred = rfr.predict(rfr_test_df)

    st.header(f"Sales: {rfr_pred[0]:.1f}")

if __name__ == '__main__':
    main()