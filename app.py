
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="ChainSense AI", layout="wide")

st.title("🚚 ChainSense AI - Professional Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["xlsx","csv"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Diagnostics","Prediction","Clustering","Upload New Data"])

    # Encoding
    le = LabelEncoder()
    df['Adoption_Encoded'] = le.fit_transform(df['Adoption_Interest'])

    X = df.drop(['Adoption_Interest','Adoption_Encoded'], axis=1)
    X = pd.get_dummies(X)
    y = df['Adoption_Encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # TAB 1: Overview
    with tab1:
        st.subheader("Dataset Overview")
        st.write(df.describe())
        fig = px.histogram(df, x="Monthly_Logistics_Cost")
        st.plotly_chart(fig)

    # TAB 2: Diagnostics
    with tab2:
        st.subheader("Correlation Analysis")
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig)

    # TAB 3: Prediction
    with tab3:
        st.subheader("Model Performance")

        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
        st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
        st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob[:,1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random'))
        fig.update_layout(title="ROC Curve")
        st.plotly_chart(fig)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig2 = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig2)

        # Feature Importance
        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
        fig3 = px.bar(importance, title="Feature Importance")
        st.plotly_chart(fig3)

    # TAB 4: Clustering
    with tab4:
        st.subheader("Customer Segmentation")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df['Cluster'] = clusters

        fig = px.scatter(df, x='Monthly_Logistics_Cost', y='Revenue_Loss_%', color='Cluster')
        st.plotly_chart(fig)

        # Association Rules
        st.subheader("Association Rules")
        df_bin = pd.get_dummies(df[['Industry']])
        freq = apriori(df_bin, min_support=0.1, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=0.5)
        st.write(rules[['antecedents','consequents','confidence','lift']].head())

    # TAB 5: Upload New Data
    with tab5:
        st.subheader("Predict New Customers")
        new_file = st.file_uploader("Upload New Data", type=["xlsx","csv"], key="new")

        if new_file:
            new_df = pd.read_excel(new_file) if new_file.name.endswith("xlsx") else pd.read_csv(new_file)
            new_df = pd.get_dummies(new_df)
            new_df = new_df.reindex(columns=X.columns, fill_value=0)
            preds = model.predict(new_df)
            new_df['Prediction'] = le.inverse_transform(preds)
            st.write(new_df)
