
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

                # If hyperparameters should be added
                if add_hyperparameters:
                    for model in selected_models:
                        st.write(f"Hyperparameters for {model}:")
                        # If Decision Tree is selected, show hyperparameters for Decision Tree
                        if model == "decision_tree":
                            max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5)
                            criterion = st.selectbox("Criterion", ["gini", "entropy"])
                            # You can add more hyperparameters as needed
                            st.write("Selected Max Depth:", max_depth)
                            st.write("Selected Criterion:", criterion)



                    num_containers = 3  # Change this to the desired number of containers

                    # Create a list to store the columns
                     

                    # Create dynamic number of columns
                    columns = st.columns(num_containers)
           
                    # Add content to each column
                    for i, col in enumerate(columns):
                        with col:
                            if i == 0:
                                st.header("A cat")
                                st.image("https://static.streamlit.io/examples/cat.jpg")
                            elif i == 1:
                                st.header("A dog")
                                st.image("https://static.streamlit.io/examples/dog.jpg")
                            elif i == 2:
                                st.header("An owl")
                                st.image("https://static.streamlit.io/examples/owl.jpg")
                            # Add more conditions or modify based on your needs for additional containers






                for models in selected_models:
                    if models == "Naive Bayes Classifier":
                        naive_bayes_model = clf.naive_bayes_classifier()
                        naive_bayes_accuracy = clf.evaluate_model(naive_bayes_model)
                        naive_bayes_classification_report = clf.evaluate_classification_report(naive_bayes_model)
                        st.write("Naive Bayes Classification Report:", pd.DataFrame(naive_bayes_classification_report))
                        st.write("Naive Bayes Accuracy:", naive_bayes_accuracy)
                    
                    if models == "Logistic Regression":
                        
                        logistic_regression_model = clf.logistic_regression()
                        logistic_regression_accuracy = clf.evaluate_model(logistic_regression_model)
                        logistic_regression_classification_report = clf.evaluate_classification_report(logistic_regression_model)
                        st.write("Logistic Regression Accuracy:", logistic_regression_accuracy)
                        st.write("Logistic Regression Classification Report:", pd.DataFrame(logistic_regression_classification_report))

                    if models == "Decision Tree":
                        decision_tree_model = clf.decision_tree()
                        decision_tree_accuracy = clf.evaluate_model(decision_tree_model)
                        decision_tree_classification_report = clf.evaluate_classification_report(decision_tree_model)
                        st.write("Decision Tree Accuracy:", decision_tree_accuracy)
                        st.write("Decision Tree Classification Report:", pd.DataFrame(decision_tree_classification_report))

                    if models == "Random Forests":
                        random_forests_model = clf.random_forests()
                        random_forests_accuracy = clf.evaluate_model(random_forests_model)
                        random_forest_classification_report = clf.evaluate_classification_report(random_forests_model)
                        st.write("Random Forests Accuracy:", random_forests_accuracy)
                        st.write("Random Forests Classification Report:", pd.DataFrame(random_forest_classification_report))

                    if models == "SVM":
                        svm_model = clf.support_vector_machines()
                        svm_accuracy = clf.evaluate_model(svm_model)
                        svm_classification_report = clf.evaluate_classification_report(svm_model)
                        st.write("Support Vector Machines Accuracy:", svm_accuracy)
                        st.write("Support Vector Machines Classification Report:", pd.DataFrame(svm_classification_report))

                        
                    if models == "KNN":
                        knn_model = clf.k_nearest_neighbour()
                        knn_accuracy = clf.evaluate_model(knn_model)
                        knn_classification_report = clf.evaluate_classification_report(knn_model)
                        st.write("K-Nearest Neighbors Accuracy:", knn_accuracy)
                        st.write("K-Nearest Neighbors Classification Report:", pd.DataFrame(knn_classification_report))
                    
                    if models == "K- Means Clustering":
                        knn_model = clf.k_means_clustering()
                        knn_accuracy = clf.evaluate_model(knn_model)
                        knn_classification_report = clf.evaluate_classification_report(knn_model)
                        st.write("K-Nearest Neighbors Accuracy:", knn_accuracy)
                        st.write("K-Nearest Neighbors Classification Report:", pd.DataFrame(knn_classification_report))



                

                # # Train the models
                # naive_bayes_model = clf.naive_bayes_classifier()
                # logistic_regression_model = clf.logistic_regression()
                # decision_tree_model = clf.decision_tree()
                # random_forests_model = clf.random_forests()
                # svm_model = clf.support_vector_machines()
                

                # # Evaluate the models
                # naive_bayes_accuracy = clf.evaluate_model(naive_bayes_model)
                # logistic_regression_accuracy = clf.evaluate_model(logistic_regression_model)
                # decision_tree_accuracy = clf.evaluate_model(decision_tree_model)
                # random_forests_accuracy = clf.evaluate_model(random_forests_model)
                # svm_accuracy = clf.evaluate_model(svm_model)
                # knn_accuracy = clf.evaluate_model(knn_model)

                # # Evaluate classification model
                # naive_bayes_classification_report = clf.evaluate_classification_report(naive_bayes_model)
                # logistic_regression_classification_report = clf.evaluate_classification_report(logistic_regression_model)
                # decision_tree_classification_report = clf.evaluate_classification_report(decision_tree_model)
                # random_forest_classification_report = clf.evaluate_classification_report(random_forests_model)
                # svm_classification_report = clf.evaluate_classification_report(svm_model)
                # knn_classification_report = clf.evaluate_classification_report(knn_model)

                # # Display the model prediction

                # # st.write("Naive Bayes Model Prediction:", clf.predict_model(naive_bayes_model)) 

                # # Display the accuracies
                # st.write("Naive Bayes Accuracy:", naive_bayes_accuracy)
                # st.write("Logistic Regression Accuracy:", logistic_regression_accuracy)
                # st.write("Decision Tree Accuracy:", decision_tree_accuracy)
                # st.write("Random Forests Accuracy:", random_forests_accuracy)
                # st.write("Support Vector Machines Accuracy:", svm_accuracy)
                # st.write("K-Nearest Neighbors Accuracy:", knn_accuracy)

                # # Display classification reports
                # st.write("Naive Bayes Classification Report:", pd.DataFrame(naive_bayes_classification_report))
                # st.write("Logistic Regression Classification Report:", pd.DataFrame(logistic_regression_classification_report))
                # st.write("Decision Tree Classification Report:", pd.DataFrame(decision_tree_classification_report))
                # st.write("Random Forests Classification Report:", pd.DataFrame(random_forest_classification_report))
                # st.write("Support Vector Machines Classification Report:", pd.DataFrame(svm_classification_report))
                # st.write("K-Nearest Neighbors Classification Report:", pd.DataFrame(knn_classification_report))

                # Display the confusion matrix



        
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

