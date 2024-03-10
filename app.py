
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import  KMeans
from sklearn.preprocessing import  StandardScaler
import warnings

import streamlit as st
from io import StringIO



df = pd.read_csv("penguins.csv")
df.head(5)
print("Hello World")

st.title("Penguin Species Classification")
spectra = st.file_uploader("upload file", type={"csv", "txt"})
if spectra is not None:
    # print("Spectra", spectra)
    spectra_df = pd.read_csv(spectra)
st.write(spectra_df.head(5))



# dataset 
headers = spectra_df.columns.tolist()
total_rows = spectra_df.shape[0]

st.write("Headers", headers)
st.write("Total Rows", total_rows)


"""fig = plt.figure(figsize=(15,6))
df.boxplot()
plt.show()

df1 = df[~((df['flipper_length_mm']>4000) | (df['flipper_length_mm']<0)) ]
df1



df2 = pd.get_dummies(df1).drop("sex_.", axis=1)
df2

# perform preprocessing steps on the dataset - scaling

scalar = StandardScaler()

X = scalar.fit_transform(df2)


df_preprocessed = pd.DataFrame(data=X, columns=df2.columns)
df_preprocessed

import numpy as np
df_preprocessed = df_preprocessed.replace(np.nan, 0)

# apply PCA 

pca = PCA(n_components= None)

dfx_pca = pca.fit(df_preprocessed)
dfx_pca.explained_variance_ratio_
n_components = sum(dfx_pca.explained_variance_ratio_>0.1)
print("Components :", n_components)
pca = PCA(n_components= n_components)
df_pca = pca.fit_transform(df_preprocessed)

dfx_pca.explained_variance_ratio_


inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df_preprocessed)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 10), inertia, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
n_clusters = 4

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(df_preprocessed)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans.labels_, cmap="viridis")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title(f"K-means Clustering (K={n_clusters})")
plt.show()"""



