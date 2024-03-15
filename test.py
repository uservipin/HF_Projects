import pandas as pd
import streamlit as st
from classification import ClassificationModels


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

                






if __name__ == "__main__":
    main()
