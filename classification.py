from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class ClassificationModels:
    def __init__(self, X, y= None,hyperparameters=None):
        self.X = X
        self.y = y
        self.hyperparameters = hyperparameters

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )


    def naive_bayes_classifier(self, param = None):
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        return model

    def logistic_regression(self, params=None):
        model = LogisticRegression()
        if self.hyperparameters and 'logistic_regression' in self.hyperparameters:
            model = GridSearchCV(model, params, cv=5)
        model.fit(self.X_train, self.y_train)
        return model

    def decision_tree(self, params=None):
        model = DecisionTreeClassifier()
        if self.hyperparameters and 'decision_tree' in self.hyperparameters:
            model = GridSearchCV(model, params =self.hyperparameters['decision_tree'], cv=5)
        model.fit(self.X_train, self.y_train)
        return model

    def random_forests(self, params=None):
        model = RandomForestClassifier()
        if self.hyperparameters and 'random_forests' in self.hyperparameters:
            model = GridSearchCV(model, params= self.hyperparameters['random_forests'], cv=5)
        model.fit(self.X_train, self.y_train)
        return model

    def support_vector_machines(self, params=None):
        model = SVC()
        if self.hyperparameters and 'support_vector_machines' in self.hyperparameters:
            model = GridSearchCV(model,   params= self.hyperparameters['support_vector_machines'], cv=5)
        model.fit(self.X_train, self.y_train)
        return model

    def k_nearest_neighbour(self, params=None):
        model = KNeighborsClassifier()
        if self.hyperparameters and 'k_nearest_neighbour' in self.hyperparameters:
            st.write(self.hyperparameters['k_nearest_neighbour'])
            model = GridSearchCV(model, params = self.hyperparameters['k_nearest_neighbour'], cv=5)
        model.fit(self.X_train, self.y_train)
        return model


    def k_means_clustering(self, n_clusters):
        model = KMeans(n_clusters=n_clusters)

        model.fit(self.X_train)
        return model

    def evaluate_model(self, model):
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy
    
    def evaluate_classification_report(self,model):
        y_pred = model.predict(self.X_test)
        return classification_report(self.y_test, y_pred, output_dict=True)
    
    def predict_output(self, model):
        y_pred = model.predict(self.X_test)
        return y_pred
