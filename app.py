
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
# https://docs.streamlit.io/library/api-reference/layout


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
            # st.write("Headers", spectra_df.columns.tolist())
            st.write("Total Rows", spectra_df.shape[0])

            option = st.text_input("Enter your text here:")
            if option:
                st.write("You have selected output column: ", option)

                y = spectra_df[option] 
                X= spectra_df.drop(option, axis=1)

    
                # Define the columns with your content
                col1, col2 = st.columns([4,1], gap="small")

                # Add content to col1
                with col1:
                    st.subheader("Train data excluding output")
                    st.write(X.head(5))

                # Add content to col2
                with col2:
                    st.subheader("Output")
                    st.write(y.head(5))

                # st.write("X",X.head(5) )
                # st.write("y", y.head(5))


                clf = ClassificationModels(X,y)

                # Split the data
                clf.split_data()
                # select model to perform classification

                # Add a multiple selection dropdown
                
                list_of_classifier_models = [
                                                "Naive Bayes Classifier",
                                                "Logistic Regression",
                                                "Decision Tree",
                                                "Random Forests",
                                                "SVM",
                                                "KNN",
                                                "K-Means Clustering"
                                                ]

                selected_models = st.multiselect("Select Models",list_of_classifier_models)

                # Execute further code based on selected models
                if selected_models:
                    # Further code execution based on selected models
                    st.write("Selected Models:", selected_models)
                    # st.write("Selected Models type:", len(selected_models))

                # Toggle to add hyperparameters
                add_hyperparameters = st.toggle("Add Hyperparameters")

                models_hyperparameters = {
                    "Naive Bayes Classifier": [],
                    "Logistic Regression": ["C", "max_iter"],
                    "Decision Tree": ["max_depth", "criterion"],
                    "Random Forests": ["n_estimators", "max_depth", "criterion"],
                    "SVM": ["C", "kernel"],
                    "KNN": ["n_neighbors", "algorithm"],
                    "K-Means Clustering": ["n_clusters", "init"]
                }

                # If hyperparameters should be added
                if add_hyperparameters:
                    num_models = len(selected_models)
                    max_items_per_row = 4
                    num_rows = (num_models + max_items_per_row - 1) // max_items_per_row  # Calculate number of rows

                    #Dictionary to store selected hyperparameters for each model

                    hyperparameters_values = {}

                    for row in range(num_rows):
                        cols = st.columns(min(num_models - row * max_items_per_row, max_items_per_row))  # Calculate number of columns for this row
                        for i, col in enumerate(cols):
                            model_index = row * max_items_per_row + i
                            with col:
                                if model_index < num_models:

                                    selected_model = selected_models[model_index]
                                    st.write(f"Selected Model: {selected_model}")  # Display selected model name
                   
                                    # initializing 
                                    if selected_model not in hyperparameters_values:
                                        hyperparameters_values[selected_model] = {}

                                    # selected_model = st.selectbox(f"Select Model {row}-{i}", selected_models, index=model_index)
                                    selected_hyperparameters = models_hyperparameters[selected_models[model_index]]


                                    for hyperparameter in selected_hyperparameters:
                                        if hyperparameter == "max_depth":
                                            max_depth = st.slider(f"Max Depth {selected_model} {hyperparameter}", min_value=1, max_value=20, value=5)
                                            hyperparameters_values[selected_model][hyperparameter] = max_depth
                                            st.write("Selected Max Depth:", max_depth)
                                        elif hyperparameter == "criterion":
                                            criterion = st.selectbox(f"Criterion {selected_model} {hyperparameter}", ["gini", "entropy"])
                                            hyperparameters_values[selected_model][hyperparameter] = criterion
                                            st.write("Selected Criterion:", criterion)
                                        elif hyperparameter == "C":
                                            C = st.slider(f"C {selected_model} {hyperparameter}", min_value=0.01, max_value=10.0, value=1.0)
                                            hyperparameters_values[selected_model][hyperparameter] = C
                                            st.write("Selected C:", C)
                                        elif hyperparameter == "max_iter":
                                            max_iter = st.slider(f"Max Iterations {selected_model} {hyperparameter}", min_value=100, max_value=10000, step=100, value=1000)
                                            hyperparameters_values[selected_model][hyperparameter] = max_iter
                                            st.write("Selected Max Iterations:", max_iter)
                                        elif hyperparameter == "n_estimators":
                                            n_estimators = st.slider(f"Number of Estimators {selected_model} {hyperparameter}", min_value=1, max_value=100, value=10)
                                            hyperparameters_values[selected_model][hyperparameter] = n_estimators
                                            st.write("Selected Number of Estimators:", n_estimators)
                                        elif hyperparameter == "kernel":
                                            kernel = st.selectbox(f"Kernel {selected_model} {hyperparameter}", ["linear", "poly", "rbf", "sigmoid"])
                                            hyperparameters_values[selected_model][hyperparameter] = kernel
                                            st.write("Selected Kernel:", kernel)
                                        elif hyperparameter == "n_neighbors":
                                            n_neighbors = st.slider(f"Number of Neighbors {selected_model} {hyperparameter}", min_value=1, max_value=50, value=5)
                                            hyperparameters_values[selected_model][hyperparameter] = n_neighbors
                                            st.write("Selected Number of Neighbors:", n_neighbors)
                                        elif hyperparameter == "algorithm":
                                            algorithm = st.selectbox(f"Algorithm {selected_model} {hyperparameter}", ["auto", "ball_tree", "kd_tree", "brute"])
                                            hyperparameters_values[selected_model][hyperparameter] = algorithm
                                            st.write("Selected Algorithm:", algorithm)
                                        elif hyperparameter == "n_clusters":
                                            n_clusters = st.slider(f"Number of Clusters {selected_model} {hyperparameter}", min_value=2, max_value=20, value=5)
                                            hyperparameters_values[selected_model][hyperparameter] = n_clusters
                                            st.write("Selected Number of Clusters:", n_clusters)
                                        elif hyperparameter == "init":
                                            init = st.selectbox(f"Initialization Method {selected_model} {hyperparameter}", ["k-means++", "random"])
                                            hyperparameters_values[selected_model][hyperparameter] = init
                                            st.write("Selected Initialization Method:", init)                                        # Add more hyperparameters as needed for each model
                    st.write("Hyperparameters:", hyperparameters_values)    

                for models in selected_models:
                    if models == "Naive Bayes Classifier":
                        naive_bayes_model = clf.naive_bayes_classifier()
                        naive_bayes_accuracy = clf.evaluate_model(naive_bayes_model)
                        naive_bayes_classification_report = clf.evaluate_classification_report(naive_bayes_model)
                        st.write("Naive Bayes Accuracy:", naive_bayes_accuracy)
                        # st.write("Naive Bayes Classification Report:", pd.DataFrame(naive_bayes_classification_report))
                    if models == "Logistic Regression":
                        
                        logistic_regression_model = clf.logistic_regression()
                        logistic_regression_accuracy = clf.evaluate_model(logistic_regression_model)
                        logistic_regression_classification_report = clf.evaluate_classification_report(logistic_regression_model)
                        st.write("Logistic Regression Accuracy:", logistic_regression_accuracy)
                        # st.write("Logistic Regression Classification Report:", pd.DataFrame(logistic_regression_classification_report))

                    if models == "Decision Tree":
                        decision_tree_model = clf.decision_tree()
                        decision_tree_accuracy = clf.evaluate_model(decision_tree_model)
                        decision_tree_classification_report = clf.evaluate_classification_report(decision_tree_model)
                        st.write("Decision Tree Accuracy:", decision_tree_accuracy)
                        # st.write("Decision Tree Classification Report:", pd.DataFrame(decision_tree_classification_report))

                    if models == "Random Forests":
                        random_forests_model = clf.random_forests()
                        random_forests_accuracy = clf.evaluate_model(random_forests_model)
                        random_forest_classification_report = clf.evaluate_classification_report(random_forests_model)
                        st.write("Random Forests Accuracy:", random_forests_accuracy)
                        # st.write("Random Forests Classification Report:", pd.DataFrame(random_forest_classification_report))

                    if models == "SVM":
                        svm_model = clf.support_vector_machines()
                        svm_accuracy = clf.evaluate_model(svm_model)
                        svm_classification_report = clf.evaluate_classification_report(svm_model)
                        st.write("Support Vector Machines Accuracy:", svm_accuracy)
                        # st.write("Support Vector Machines Classification Report:", pd.DataFrame(svm_classification_report))

                        
                    if models == "KNN":
                        knn_model = clf.k_nearest_neighbour()
                        knn_accuracy = clf.evaluate_model(knn_model)
                        knn_classification_report = clf.evaluate_classification_report(knn_model)
                        st.write("K-Nearest Neighbors Accuracy:", knn_accuracy)
                        # st.write("K-Nearest Neighbors Classification Report:", pd.DataFrame(knn_classification_report))
                    
                    if models == "K- Means Clustering":
                        knn_model = clf.k_means_clustering()
                        knn_accuracy = clf.evaluate_model(knn_model)
                        knn_classification_report = clf.evaluate_classification_report(knn_model)
                        st.write("K-Nearest Neighbors Accuracy:", knn_accuracy)
                        # st.write("K-Nearest Neighbors Classification Report:", pd.DataFrame(knn_classification_report))



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

