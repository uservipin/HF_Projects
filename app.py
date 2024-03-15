
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import  KMeans
from sklearn.preprocessing import  StandardScaler
import warnings
import streamlit as st
from io import StringIO

from classification import ClassificationModels

warnings.filterwarnings("ignore")
import uuid


# data cleaning: https://bank-performance.streamlit.app/


# Define function for each page
# def classification():
#     st.title("Home Page")
#     st.write("Welcome to the Home Page")

def regressor():
    st.title("About Page")
    st.write("This is the About Page")

def NLP():
    st.title("Contact Page")
    st.write("You can reach us at example@example.com")

def Image():
    st.title("Home Page")
    st.write("Welcome to the Home Page")

def Voice():
    st.title("Home Page")
    st.write("Welcome to the Home Page")

def Video():
    st.title("Home Page")
    st.write("Welcome to the Home Page")

def LLMs():
    st.title("About Page")
    st.write("This is the About Page")

def resume():
    st.title("Contact Page")
    st.write("You can reach us at example@example.com")

# Main function to run the app
def main():
    st.sidebar.title("Navigation")
    page_options = ["Classification", "Regressor", "NLP", "Image", "Voice", "Video", "LLMs"]
    choice = st.sidebar.radio("Go to", page_options)

    if choice == "Classification":
        st.title("Classification")
        spectra = st.file_uploader("Upload file", type={"csv", "txt"})
        
        if spectra is not None:
            spectra_df = pd.read_csv(spectra)
            
            st.write(spectra_df.head(5))
            st.write("Headers", spectra_df.columns.tolist())
            st.write("Total Rows", spectra_df.shape[0])

            option = st.text_input("Enter your text here:")
            if option:
                st.write("You entered:", option)

                y = spectra_df[option] 
                X= spectra_df.drop(option, axis=1)

                st.write("X",X.head(5) )
                st.write("y", y.head(5))


                clf = ClassificationModels(X,y)

                # Split the data
                clf.split_data()

                # Train the models
                naive_bayes_model = clf.naive_bayes_classifier()
                logistic_regression_model = clf.logistic_regression()
                decision_tree_model = clf.decision_tree()
                random_forests_model = clf.random_forests()
                svm_model = clf.support_vector_machines()
                knn_model = clf.k_nearest_neighbour()

                # Evaluate the models
                naive_bayes_accuracy = clf.evaluate_model(naive_bayes_model)
                logistic_regression_accuracy = clf.evaluate_model(logistic_regression_model)
                decision_tree_accuracy = clf.evaluate_model(decision_tree_model)
                random_forests_accuracy = clf.evaluate_model(random_forests_model)
                svm_accuracy = clf.evaluate_model(svm_model)
                knn_accuracy = clf.evaluate_model(knn_model)

                # Evaluate classification model
                naive_bayes_classification_report = clf.evaluate_classification_report(naive_bayes_model)
                logistic_regression_classification_report = clf.evaluate_classification_report(logistic_regression_model)
                decision_tree_classification_report = clf.evaluate_classification_report(decision_tree_model)
                random_forest_classification_report = clf.evaluate_classification_report(random_forests_model)
                svm_classification_report = clf.evaluate_classification_report(svm_model)
                knn_classification_report = clf.evaluate_classification_report(knn_model)

                # Display the model prediction

                # st.write("Naive Bayes Model Prediction:", clf.predict_model(naive_bayes_model)) 

                # Display the accuracies
                st.write("Naive Bayes Accuracy:", naive_bayes_accuracy)
                st.write("Logistic Regression Accuracy:", logistic_regression_accuracy)
                st.write("Decision Tree Accuracy:", decision_tree_accuracy)
                st.write("Random Forests Accuracy:", random_forests_accuracy)
                st.write("Support Vector Machines Accuracy:", svm_accuracy)
                st.write("K-Nearest Neighbors Accuracy:", knn_accuracy)

                # Display classification reports
                st.write("Naive Bayes Classification Report:", pd.DataFrame(naive_bayes_classification_report))
                st.write("Logistic Regression Classification Report:", pd.DataFrame(logistic_regression_classification_report))
                st.write("Decision Tree Classification Report:", pd.DataFrame(decision_tree_classification_report))
                st.write("Random Forests Classification Report:", pd.DataFrame(random_forest_classification_report))
                st.write("Support Vector Machines Classification Report:", pd.DataFrame(svm_classification_report))
                st.write("K-Nearest Neighbors Classification Report:", pd.DataFrame(knn_classification_report))

                # Display the confusion matrix

                    # Add a multiple selection dropdown
                
                list_of_classifier_models = [
                                                "naive_bayes_classifier",
                                                "logistic_regression",
                                                "decision_tree",
                                                "random_forests",
                                                "support_vector_machines",
                                                "k_nearest_neighbour",
                                                "k_means_clustering"
                                                ]

                selected_models = st.multiselect("Select Models",list_of_classifier_models)

                # Execute further code based on selected models
                if selected_models:
                    # Further code execution based on selected models
                    st.write("Selected Models:", selected_models)
                    st.write("Selected Models type:", len(selected_models))

                    # Toggle to add hyperparameters
                    add_hyperparameters = st.checkbox("Add Hyperparameters")

                    # If hyperparameters should be added
                    if add_hyperparameters:
                        # For each selected model, create a space for hyperparameters
                        for model in selected_models:
                            st.write(f"Hyperparameters for {model}:")
                            # Here you can add text inputs, sliders, or any other widgets to input hyperparameters


        




        
    elif choice == "Regressor":
        regressor()
    elif choice == "NLP":
        NLP()

    if choice == "Image":
        Image() 
    
    if choice == "Voice":
        Voice()

    if choice == "Video":
        Video()
    
    if choice == "LLMs": 
        LLMs()

if __name__ == "__main__":
    main()



# fig = plt.figure(figsize=(15,6))
# df.boxplot()
# plt.show()

# df1 = df[~((df['flipper_length_mm']>4000) | (df['flipper_length_mm']<0)) ]
# df1



# df2 = pd.get_dummies(df1).drop("sex_.", axis=1)
# df2

# # perform preprocessing steps on the dataset - scaling

# scalar = StandardScaler()

# X = scalar.fit_transform(df2)


# df_preprocessed = pd.DataFrame(data=X, columns=df2.columns)
# df_preprocessed

# import numpy as np
# df_preprocessed = df_preprocessed.replace(np.nan, 0)

# # apply PCA 

# pca = PCA(n_components= None)

# dfx_pca = pca.fit(df_preprocessed)
# dfx_pca.explained_variance_ratio_
# n_components = sum(dfx_pca.explained_variance_ratio_>0.1)
# print("Components :", n_components)
# pca = PCA(n_components= n_components)
# df_pca = pca.fit_transform(df_preprocessed)

# dfx_pca.explained_variance_ratio_


# inertia = []
# for k in range(1, 10):
#     kmeans = KMeans(n_clusters=k, random_state=42).fit(df_preprocessed)
#     inertia.append(kmeans.inertia_)
# plt.plot(range(1, 10), inertia, marker="o")
# plt.xlabel("Number of clusters")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.show()
# n_clusters = 4

# n_clusters = 4
# kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(df_preprocessed)
# plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans.labels_, cmap="viridis")
# plt.xlabel("First Principal Component")
# plt.ylabel("Second Principal Component")
# plt.title(f"K-means Clustering (K={n_clusters})")
# plt.show()"""



