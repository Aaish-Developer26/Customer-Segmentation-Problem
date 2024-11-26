import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Title
st.title("Customer Segmentation Using K-Means")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())
    
    # Select Features for Clustering
    features = st.multiselect("Select features for clustering:", data.columns)
    if len(features) < 2:
        st.warning("Please select at least two features.")
    else:
        st.write("Selected Features: ", features)
        
        # Specify the number of clusters
        n_clusters = st.slider("Select Number of Clusters (K):", min_value=2, max_value=10, value=3)
        
        # Perform K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data[features])
        data["Cluster"] = clusters
        
        st.write("Clustered Data Preview:")
        st.dataframe(data.head())
        
        # Visualize Clusters (for 2D features only)
        if len(features) == 2:
            st.write("Cluster Visualization:")
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=data, x=features[0], y=features[1], hue="Cluster", palette="viridis", s=100)
            plt.title("Customer Segments")
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            st.pyplot(plt.gcf())
        
        # Display Cluster Centers
        st.write("Cluster Centers:")
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
        st.dataframe(centers)
else:
    st.info("Awaiting CSV file upload. Please upload your dataset to begin.")
