import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

# ==============================
# Title & Description
# ==============================
st.title("🛒 Customer Segmentation App")
st.markdown(
    """
    This app helps businesses **segment customers** using KMeans.  
    Upload your dataset or try the **sample dataset (Mall_Customers.csv)** to discover hidden customer groups for targeted marketing.  
    """
)

# ==============================
# Sidebar Controls
# ==============================
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")
use_sample = st.sidebar.checkbox("Use Sample Dataset")

# ==============================
# Load Dataset
# ==============================
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = pd.read_csv("Mall_Customers.csv")

if df is not None:
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    features = st.sidebar.multiselect(
        "Select features for segmentation",
        options=df.columns.tolist(),
        default=["Annual Income (k$)", "Spending Score (1-100)"]
    )

    X = df[features].select_dtypes(include=['int64', 'float64'])

    if X.shape[1] >= 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_segments = st.sidebar.slider("Number of segments (k)", 2, 10, 5)

        kmeans = KMeans(n_clusters=n_segments, init="k-means++", random_state=42)
        y_segments = kmeans.fit_predict(X_scaled)
        df["Segment"] = y_segments

        # ==============================
        # Assign Business-Friendly Labels
        # ==============================
        segment_labels = {}
        if "Annual Income (k$)" in features and "Spending Score (1-100)" in features:
            segment_summary = df.groupby("Segment")[features].mean()

            for i in range(n_segments):
                income = segment_summary.iloc[i]["Annual Income (k$)"]
                score = segment_summary.iloc[i]["Spending Score (1-100)"]

                if income > 60 and score > 60:
                    segment_labels[i] = "VIP Customers"
                elif income < 40 and score > 60:
                    segment_labels[i] = "Impulsive Buyers"
                elif income > 60 and score < 40:
                    segment_labels[i] = "Potential Customers"
                elif income < 40 and score < 40:
                    segment_labels[i] = "Budget Customers"
                else:
                    segment_labels[i] = "Regular Customers"
        else:
            for i in range(n_segments):
                segment_labels[i] = f"Group {i}"

        df["Customer Segment"] = df["Segment"].map(segment_labels)

        # ==============================
        # Show Data
        # ==============================
        st.subheader("📋 Segmented Customer Data")
        st.write(df.head())

        # ==============================
        # Visualizations
        # ==============================
        st.subheader("📈 Visualizations")

        if len(features) == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                x=df[features[0]],
                y=df[features[1]],
                hue=df["Customer Segment"],
                palette="Set2",
                s=100,
                ax=ax
            )
            plt.title("Customer Segments")
            st.pyplot(fig)

        segment_counts = df["Customer Segment"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(segment_counts, labels=segment_counts.index,
               autopct="%1.1f%%", startangle=90, colors=sns.color_palette("Set2"))
        ax.set_title("Customer Distribution by Segment")
        st.pyplot(fig)

        # ==============================
        # Business Insights
        # ==============================
        st.subheader("💡 Customer Segment Insights")
        for label in segment_counts.index:
            st.info(f"**{label}:** Customers with similar behavior grouped together.")

        # ==============================
        # Download Segmented Data
        # ==============================
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Segmented Data (CSV)",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv",
        )
    else:
        st.warning("⚠️ Please select at least 2 **numeric** features (e.g., Income, Spending Score, Gender).")
else:
    st.info("👈 Upload a dataset or select 'Use Sample Dataset' to get started.")
