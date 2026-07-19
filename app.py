import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Ford Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("../output/final_bike_data.csv")

df = load_data()

@st.cache_resource
def load_model():
    return joblib.load("../models/random_forest_model.pkl")

model = load_model()

st.sidebar.title("🚲 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "EDA Dashboard",
        "Prediction",
        "Business Insights"
    ]
)
if page == "Home":

    st.title("🚲 Ford Bike Sharing Analytics Dashboard")

    st.markdown(
        """
        This dashboard provides complete analysis of the Ford Bike Sharing Dataset.
        It includes:
        - Exploratory Data Analysis
        - Machine Learning Prediction
        - Business Insights
        - Recommendations
        """
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Trips", len(df))

    c2.metric(
        "Average Duration (min)",
        round(df["duration_min"].mean(),2)
    )

    c3.metric(
        "Subscribers",
        df[df["user_type"]=="Subscriber"].shape[0]
    )

    st.success("Dashboard Loaded Successfully ✅")

elif page == "EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("User Type Distribution")

        fig, ax = plt.subplots(figsize=(5,4))
        df["user_type"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("User Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.subheader("Ride Duration Distribution")

        fig, ax = plt.subplots(figsize=(5,4))
        ax.hist(df["duration_min"], bins=30)
        ax.set_xlabel("Duration (Minutes)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Monthly Ride Count")

        fig, ax = plt.subplots(figsize=(5,4))
        df["month"].value_counts().sort_index().plot(kind="line", marker="o", ax=ax)
        ax.set_xlabel("Month")
        ax.set_ylabel("Ride Count")
        st.pyplot(fig)

    with col4:
        st.subheader("Gender Distribution")

        fig, ax = plt.subplots(figsize=(5,4))
        df["member_gender"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

if page == "Prediction":

    st.title("🤖 User Type Prediction")

    duration = st.number_input("Ride Duration (Minutes)", min_value=1.0)
    age = st.number_input("Age", min_value=18)
    hour = st.slider("Hour", 0, 23, 12)
    month = st.selectbox("Month", sorted(df["month"].unique()))

    if st.button("Predict"):

        input_data = pd.DataFrame({
            "duration_min": [duration],
            "age": [age],
            "hour": [hour],
            "month": [month]
        })

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("Predicted User Type: Subscriber")
        else:
            st.success("Predicted User Type: Customer")

if page == "Business Insights":

    st.title("💼 Business Insights")

    st.markdown("## 📌 Key Business Insights")

    st.success("""
    ✅ Subscribers are significantly higher than Customers.

    ✅ Most bike rides last between 10–15 minutes.

    ✅ Bike demand is highest during peak commuting hours.

    ✅ Riders aged 25–35 use the bike-sharing service the most.

    ✅ Monthly ride patterns indicate seasonal variation in demand.
    """)

    st.markdown("---")

    st.markdown("## 🚀 Business Recommendations")

    st.info("""
 • Increase bike availability during peak hours.

 • Offer promotional discounts to casual customers.

 • Improve bike availability at high-demand stations.

 • Perform preventive maintenance on frequently used bikes.

 • Launch seasonal marketing campaigns to increase ridership.
""")

    st.markdown("---")

    st.markdown("## ✅ Conclusion")

    st.write("""
The Ford Bike Sharing Analytics Dashboard provides valuable insights into
customer behaviour, ride duration, seasonal demand, and user patterns.

The Machine Learning model successfully predicts user type based on ride
characteristics. These insights can help improve operational efficiency,
customer satisfaction, and business decision-making.
""")

    st.balloons() 