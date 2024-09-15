from classification import ClassificationModels
from regression import RegressionModels 
from resume import Resume
from chat_pdf_openai import chatpdf

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

import pandas as pd
import warnings
import streamlit as st
import uuid
import time
import os
import io
import pathlib
import textwrap

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
warnings.filterwarnings("ignore")

# data cleaning: https://bank-performance.streamlit.app/
# https://docs.streamlit.io/library/api-reference/layout


load_dotenv()  # take environment variables from .env.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



## Function to load OpenAI model and get respones
model_chat = genai.GenerativeModel('gemini-pro')
chat = model_chat.start_chat(history=[])

## Function to load OpenAI model and get respones
model_vision = genai.GenerativeModel('gemini-pro-vision')

def get_gemini_response(question):
    response =chat.send_message(question,stream=True)
    return response

## Function to load OpenAI model and get respones

def get_gemini_response_vision(input,image):
    if input!="":
       response = model_vision.generate_content([input,image])
    else:
       response = model_vision.generate_content(image)
    return response.text

def gemini_model():
    ##initialize our streamlit app
    # st.set_page_config(page_title="Q&A Demo")
    st.header("Gemini Application")
    input=st.text_input("Input: ",key="input")
    submit=st.button("Ask the question")
    ## If ask button is clicked
    if submit:
        response=get_gemini_response(input)
        st.subheader("The Response is")
        for chunk in response:
            print(st.write(chunk.text))
            print("_"*80)
        
        # st.write(chat.history)

# Define function for each page

def classification():
        train, test = st.tabs(['Train','Test'])

        with train:
            st.title("Classification / Train data")
            spectra = st.file_uploader("**Upload file**", type={"csv", "txt"})
            
            if spectra is not None:
                spectra_df = pd.read_csv(spectra)
                
                st.write(spectra_df.head(5))
                # st.write("Headers", spectra_df.columns.tolist())
                st.write("**Total Rows**", spectra_df.shape[0])

                st.divider()

                option = st.text_input("**Select Output Column**:")
                st.divider()

                if option:
                    st.write("**You have selected output column**: ", option)

                    X= spectra_df.drop(option, axis=1)
                    y = spectra_df[option] 

                    # Define the columns with your content
                    col1, col2 = st.columns([4,1], gap="small")

                    # Add content to col1
                    with col1:
                        st.write("Train data excluding output")
                        st.write(X.head(5))

                    # Add content to col2
                    with col2:
                        st.write("Output")
                        st.write(y.head(5))

                    st.divider()
                    
                    list_of_classifier_models = [
                                                    "Naive Bayes Classifier",
                                                    "Logistic Regression",
                                                    "Decision Tree",
                                                    "Random Forests",
                                                    "SVM",
                                                    "KNN",
                                                    "K-Means Clustering"
                                                    ]

                    models_hyperparameters = {
                                                "Naive Bayes Classifier": [],
                                                "Logistic Regression": ["C", "max_iter"],
                                                "Decision Tree": ["max_depth", "criterion"],
                                                "Random Forests": ["n_estimators", "max_depth", "criterion"],
                                                "SVM": ["C", "kernel"],
                                                "KNN": ["n_neighbors", "algorithm"],
                                                "K-Means Clustering": ["n_clusters", "init"]
                    }

                    selected_models = st.multiselect("**Select Models**:",list_of_classifier_models)

                    # Execute further code based on selected models
                    if selected_models:
                        # st.write("Selected Models:", selected_models)
                        # Toggle to add hyperparameters
                        add_hyperparameters = st.toggle("Add Hyperparameters")
                        
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
                            # st.write("Hyperparameters:", hyperparameters_values)    
                        
                        
                            clf = ClassificationModels(X,y,hyperparameters_values)
                            # model_accuracy = {}
                            # Split the data
                            clf.split_data()

                            accuracy_dict= {}

                            for models in selected_models:

                                model_hyperparameters = hyperparameters_values.get(models, {})  # Get selected hyperparameters for this model
                                
                                if models not in accuracy_dict:
                                    accuracy_dict[models] = 0

                                # st.write("trained param",trained_models)
                                # for model_name in model_hyperparameters
                                
                                if models == "Naive Bayes Classifier":
                                    # Pipeline to implement model

                                    naive_bayes_model = clf.naive_bayes_classifier(model_hyperparameters)

                                    naive_bayes_accuracy = clf.evaluate_model(naive_bayes_model)
                                    # naive_bayes_classification_report = clf.evaluate_classification_report(naive_bayes_model)
                                    # st.write("Naive Bayes Accuracy:", naive_bayes_accuracy)
                                    accuracy_dict[models] = naive_bayes_accuracy
                                    # st.write("Naive Bayes Classification Report:", pd.DataFrame(naive_bayes_classification_report))
                                if models == "Logistic Regression":
                                    # st.write("Logistic Regression Model:", model_hyperparameters)
                                    logistic_regression_model = clf.logistic_regression(model_hyperparameters)
                                    logistic_regression_accuracy = clf.evaluate_model(logistic_regression_model)
                                    # logistic_regression_classification_report = clf.evaluate_classification_report(logistic_regression_model)
                                    # st.write("Logistic Regression Accuracy:", logistic_regression_accuracy)
                                    accuracy_dict[models] = logistic_regression_accuracy
                                    # st.write("Logistic Regression Classification Report:", pd.DataFrame(logistic_regression_classification_report))

                                if models == "Decision Tree":
                                    decision_tree_model = clf.decision_tree(model_hyperparameters)
                                    decision_tree_accuracy = clf.evaluate_model(decision_tree_model)
                                    # decision_tree_classification_report = clf.evaluate_classification_report(decision_tree_model)
                                    # st.write("Decision Tree Accuracy:", decision_tree_accuracy)
                                    accuracy_dict[models] = decision_tree_accuracy
                                    # st.write("Decision Tree Classification Report:", pd.DataFrame(decision_tree_classification_report))

                                if models == "Random Forests":
                                    random_forests_model = clf.random_forests(model_hyperparameters)
                                    random_forests_accuracy = clf.evaluate_model(random_forests_model)
                                    accuracy_dict[models] = random_forests_accuracy
                                    # random_forest_classification_report = clf.evaluate_classification_report(random_forests_model)
                                    # st.write("Random Forests Accuracy:", random_forests_accuracy)
                                    # st.write("Random Forests Classification Report:", pd.DataFrame(random_forest_classification_report))

                                if models == "SVM":
                                    svm_model = clf.support_vector_machines(model_hyperparameters)
                                    svm_accuracy = clf.evaluate_model(svm_model)
                                    accuracy_dict[models] = svm_accuracy
                                    # svm_classification_report = clf.evaluate_classification_report(svm_model)
                                    # st.write("Support Vector Machines Accuracy:", svm_accuracy)
                                    # st.write("Support Vector Machines Classification Report:", pd.DataFrame(svm_classification_report))

                                    
                                if models == "KNN":
                                    knn_model = clf.k_nearest_neighbour(model_hyperparameters)
                                    knn_accuracy = clf.evaluate_model(knn_model)
                                    accuracy_dict[models] = knn_accuracy
                                    # knn_classification_report = clf.evaluate_classification_report(knn_model)
                                    # st.write("K-Nearest Neighbors Accuracy:", knn_accuracy)
                                    # st.write("K-Nearest Neighbors Classification Report:", pd.DataFrame(knn_classification_report))
                                
                                if models == "K- Means Clustering":
                                    kmeans_model = clf.k_means_clustering(model_hyperparameters)
                                    kmeans_accuracy = clf.evaluate_model(kmeans_model)
                                    accuracy_dict[models] = kmeans_accuracy
                                    # knn_classification_report = clf.evaluate_classification_report(knn_model)
                                    # st.write("K-Nearest Neighbors Accuracy:", kmeans_accuracy)
                                    # st.write("K-Nearest Neighbors Classification Report:", pd.DataFrame(knn_classification_report))
                            
                            st.divider()
                            st.write("Models Accuracy:", accuracy_dict)
                            max_key = ''
                            max_value = 0
                            for i in accuracy_dict:
                                if accuracy_dict[i] > max_value:
                                    max_key = i
                                    max_value = accuracy_dict[i]
                        
                            st.write("Efficient Model is :",max_key, accuracy_dict[max_key])
                            st.divider()
                            st.write("Scroll up and Click on <**Test**> tab to test Model performance")
    
        with test: 
                st.title("Classification / Test")       
                spectra_1 = st.file_uploader("Upload file test the model", type={"csv", "txt"})

                if spectra_1 is not None:
                    spectra_df1 = pd.read_csv(spectra_1)
                   # Actual = spectra_df1['Disease']
                    #spectra_df1 = spectra_df1.drop(columns=['Disease'])
                    st.write(spectra_df1.head(5))
                    st.divider()
                    
                    X= spectra_df1
                    if max_key == "Naive Bayes Classifier":
                        # naive_bayes_model = clf.naive_bayes_classifier(model_hyperparameters)
                        naive_bayes_model =naive_bayes_model.predict()
                        X['Predict'] = naive_bayes_model
                        st.write("Output : ", X)
                        st.write("Model used for Prediction is: Naive Bayes Model", naive_bayes_model)
                    
                    if max_key == "Logistic Regression":
                        logistic_regression_model_ = logistic_regression_model.predict(X)
                        X['Predict'] = logistic_regression_model_
                        st.write("Output : ", X)
                        st.write("Model used for Prediction is: Logistic Regression")

                    if max_key == "Decision Tree":
                        decision_tree_model_ = decision_tree_model.predict(X)
                        X['Predict'] = decision_tree_model_
                        #X['Actual'] = Actual
                        st.write("Model used for Prediction is: Decision Tree ", X)
                    
                    if max_key == "Random Forests":
                        random_forests_model = random_forests_model.predict(X)
                        X['Predict'] = random_forests_model
                        st.write("Model used for Prediction is: Random Forests Model:", random_forests_model)
                    
                    if max_key == "SVM":
                        svm_model = svm_model.predict(X)
                        X['Predict'] = random_forests_model
                        st.write("Model used for Prediction is: Support Vector Machines Model:", svm_model)
                    
                    if max_key == "KNN":
                        knn_model = knn_model.predict(X)
                        X['Predict'] = random_forests_model
                        st.write("Model used for Prediction is: K-Nearest Neighbors Model:", knn_model)
                    
                    if max_key == "K- Means Clustering":
                        kmeans_model =kmeans_model.predict(X)
                        X['Predict'] = random_forests_model
                        st.write("Model used for Prediction is: K-Means Clustering Model:", kmeans_model)

                    st.divider()

                    data_frame = pd.DataFrame(X).to_csv().encode('utf-8')
                    st.download_button(
                        label="Download data as CSV",
                        data=data_frame,
                        file_name='classifier_tagging_df.csv',
                        mime='text/csv',
                    )

                    st.divider()


def regressor():
    train, test = st.tabs(['Train','Test'])

    with train:
            st.title("Regression / Train data")
            spectra = st.file_uploader("**Upload file**", type={"csv", "txt"})
            
            if spectra is not None:
                spectra_df = pd.read_csv(spectra)
                
                st.write(spectra_df.head(5))
                # st.write("Headers", spectra_df.columns.tolist())
                st.write("**Total Rows**", spectra_df.shape[0])

                st.divider()

                option = st.text_input("**Select Output Column**:")
                st.divider()

                if option:
                    st.write("**You have selected output column**: ", option)

                    y = spectra_df[option] 
                    X= spectra_df.drop(option, axis=1)

                                        # Define the columns with your content
                    col1, col2 = st.columns([4,1], gap="small")

                    # Add content to col1
                    with col1:
                        st.write("Train data excluding output")
                        st.write(X.head(5))

                    # Add content to col2
                    with col2:
                        st.write("Output")
                        st.write(y.head(5))

                    st.divider()

                    # Select models
                    # models_list = [
                    #     'Linear Regression', 'Polynomial Regression', 'Ridge Regression',
                    #     'Lasso Regression', 'ElasticNet Regression', 'Logistic Regression',
                    #     'Decision Tree Regression', 'Random Forest Regression',
                    #     'Gradient Boosting Regression', 'Support Vector Regression (SVR)',
                    #     'XGBoost', 'LightGBM'
                    # ]

                    models_list = [
                                   'Linear Regression',
                                    'Polynomial Regression',
                                    'Ridge Regression',
                                    'Lasso Regression',
                                    'ElasticNet Regression',
                                    'Logistic Regression',
                                    'Decision Tree Regression',
                                    'Random Forest Regression',
                                    'Gradient Boosting Regression',
                                    'Support Vector Regression (SVR)',
                                    'XGBoost',
                                    'LightGBM'
                                    ]

                    selected_models = st.multiselect('Select Regression Models', models_list)

                    if selected_models:
                        # Initialize RegressionModels class
                        models = RegressionModels()
                        
                        # Add data
                        models.add_data(X, y)
                        
                        # Split data into training and testing sets
                        models.split_data()

                        # Train and evaluate selected models
                        best_model = None
                        best_metric = float('inf')  # Initialize with a high value for MSE (lower is better)
                        for model_name in selected_models:
                            # st.subheader(f"Model: {model_name}")
                            models.fit(model_name)
                            y_pred = models.train(model_name)
                            mse, r2 = models.evaluate(model_name)
                            # st.write(f"MSE: {mse}")
                            # st.write(f"R-squared: {r2}")
                            
                            # Update best model based on MSE
                            if r2 < best_metric:
                                best_model = model_name
                                best_metric = r2


                        # Perform testing based on the best model
                        if best_model:
                            st.subheader(f"Best Model: {best_model}")
                            test_mse, test_r2 = models.evaluate(best_model)
                            st.write(f"Test MSE: {test_mse}")
                            st.write(f"Test R-squared: {test_r2}")
                            # You can also visualize the predictions vs. true values, residual plots, etc. here
                        else:
                            st.write("No best model selected.")                                



    with test:
        st.title("Regression / Test")       
        spectra_1 = st.file_uploader("Upload file test the model", type={"csv", "txt"})
        if spectra_1 is not None:
            spectra_df1 = pd.read_csv(spectra_1)
            st.write(spectra_df1.head(5))
            st.divider()
            st.write("models",models)
            # models = RegressionModels()
            if best_model:
                # st.subheader(f"Best Model: {best_model}")
                st.write("best model", best_model)
                y_pred= models.predict(model_name = best_model,X = spectra_df1)
                # st.write(f"Test MSE: {test_mse}")
                st.write(f"Y pred is : {max(y_pred)}")
                # You can also visualize the predictions vs. true values, residual plots, etc. here
            else:
                st.write("No best model selected.")


def NLP():
    Gemini_Chat,Gemini_Vision, OpenAiDocChat, Bert, = st.tabs(['Gemini-Chat','Gemini-Vision',"OpenAi Docs Chat",'ChatBot'])

    with Gemini_Chat:
            st.title("Chat with Gemini Pro")
            st.write("Note: ask basic question from LLMs")
            gemini_model()

    with Gemini_Vision:

        st.header("Chat with Image using Gemini ")
        st.write("Note: upload single image and ask question related to Image, and Input the relative prompt to ask question:")
        input=st.text_input("Input Prompt: ",key="input_prompt")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        image=""  

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            #image = Image.open(io.BytesIO(uploaded_file.read()))

            st.image(image, caption="Uploaded Image.", use_column_width=True) 
       
        submit=st.button("Tell me about the image")
        ## If ask button is clicked
        if submit:
            response=get_gemini_response_vision(input,image)
            st.subheader("The Response is")
            st.write(response)

    # with Gemini_PDF:
    #     st.title(" Working on the model, will add soon.")

    with OpenAiDocChat:
   
        gpt_model = "gpt-3.5-turbo"
        st.title("Document Question-Answering App")
        password = st.text_input("Enter a password", type="password")
        if 'asdfghjkl' == password:
            # st.write("Login success")
            chatpdf().qa_pdf(gpt_model)

        else:
            st.write("Please enter correct password")
        

    with Bert:     
            st.title(" Working on the model, will add soon.")


def deep_learning():
    st.title("Deep Learning Models")
    st.write("Needs to add projects of deep learning")


def resume():
    st.title("Resume")
    st.write("")
    About, Work_Experience,Skills_Tools, Education_Certification = st.tabs(["About", "Work Experience","Skills & Tools", "Education & Certificates"])

    with About:
        Resume().display_information()
    
    with Work_Experience:
        Resume().display_work_experience()

    with Skills_Tools:
        Resume().skills_tools()
    
    with Education_Certification:
        Resume().display_education_certificate()

# Main function to run the app
def main():

    st.sidebar.title("Deep Learning/ Data Science/ AI Models")
    # page_options = ["Classification", "Regressor", "NLP", "Image", "Voice", "Video", "LLMs"]
    page_options = ["Chatbot & NLP" ,"Classification", "Regressor","Deep Learning", "Resume"]
    choice = st.sidebar.radio("Select", page_options)

    if choice == "Classification":
        classification()

    elif choice == "Regressor":
        regressor()
    elif choice == "Chatbot & NLP":
        NLP()

    if choice == "Deep Learning": 
        deep_learning()

    if choice == 'Resume':
        resume()

if __name__ == "__main__":
    main()
