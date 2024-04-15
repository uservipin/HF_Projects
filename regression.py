from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st






class RegressionModels:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.column_transformer = None  # Initialize as None
        self.models = {
            'Linear Regression': LinearRegression(),
            'Polynomial Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'ElasticNet Regression': ElasticNet(),
            'Logistic Regression': LogisticRegression(),
            'Decision Tree Regression': DecisionTreeRegressor(),
            'Random Forest Regression': RandomForestRegressor(),
            'Gradient Boosting Regression': GradientBoostingRegressor(),
            'Support Vector Regression (SVR)': SVR(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor()
        }
        
    def add_data(self, X, y):
        self.data = (X, y)
        
    def split_data(self, test_size=0.2, random_state=None):
        if self.data is None:
            raise ValueError("No data provided. Use add_data method to add data first.")
        X, y = self.data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def build_preprocessor(self):
        if self.column_transformer is not None:
            return self.column_transformer  # Return the existing fitted ColumnTransformer
        else:
            # Separate numerical and categorical columns
            numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = self.X_train.select_dtypes(include=['object']).columns

            # Define transformers for numerical and categorical data
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine transformers using ColumnTransformer
            self.column_transformer = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            return self.column_transformer
        
    def fit(self, model_name):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split. Use split_data method to split data into training and testing sets.")
        model = self.models[model_name]
        preprocessor = self.build_preprocessor()
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        model_pipeline.fit(self.X_train, self.y_train)
        
    def train(self, model_name):
        if self.X_train is None or self.y_train is None or self.X_test is None:
            raise ValueError("Data not split. Use split_data method to split data into training and testing sets.")
        model = self.models[model_name]
        preprocessor = self.build_preprocessor()
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        model_pipeline.fit(self.X_train, self.y_train)
        y_pred = model_pipeline.predict(self.X_test)
        return y_pred
        
    def evaluate(self, model_name):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not split. Use split_data method to split data into training and testing sets.")
        model = self.models[model_name]
        preprocessor = self.build_preprocessor()
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        model_pipeline.fit(self.X_train, self.y_train)
        y_pred = model_pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, r2
        
    def predict(self, model_name, X):
        model = self.models[model_name]
        preprocessor = self.build_preprocessor()  # Ensure that the ColumnTransformer is fitted
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        return model_pipeline.predict(X)





'''


class RegressionModels:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {
            'Linear Regression': LinearRegression(),
            'Polynomial Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'ElasticNet Regression': ElasticNet(),
            'Logistic Regression': LogisticRegression(),
            'Decision Tree Regression': DecisionTreeRegressor(),
            'Random Forest Regression': RandomForestRegressor(),
            'Gradient Boosting Regression': GradientBoostingRegressor(),
            'Support Vector Regression (SVR)': SVR(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor()
        }
        
    def add_data(self, X, y):
        self.data = (X, y)
        
    def split_data(self, test_size=0.2, random_state=None):
        if self.data is None:
            raise ValueError("No data provided. Use add_data method to add data first.")
        X, y = self.data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def build_preprocessor(self):
        # Separate numerical and categorical columns
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X_train.select_dtypes(include=['object']).columns

        # Define transformers for numerical and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        return preprocessor
        
    def fit(self, model_name):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split. Use split_data method to split data into training and testing sets.")
        model = self.models[model_name]
        preprocessor = self.build_preprocessor()
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        model_pipeline.fit(self.X_train, self.y_train)
        
    def train(self, model_name):
        if self.X_train is None or self.y_train is None or self.X_test is None:
            raise ValueError("Data not split. Use split_data method to split data into training and testing sets.")
        model = self.models[model_name]
        preprocessor = self.build_preprocessor()
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        model_pipeline.fit(self.X_train, self.y_train)
        y_pred = model_pipeline.predict(self.X_test)
        return y_pred
        
    def evaluate(self, model_name):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not split. Use split_data method to split data into training and testing sets.")
        model = self.models[model_name]
        preprocessor = self.build_preprocessor()
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        model_pipeline.fit(self.X_train, self.y_train)
        y_pred = model_pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, r2
        
    def predict(self, model_name, X):
        model = self.models[model_name]
        preprocessor = self.build_preprocessor()
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
                                ])
        
        st.write("Model", model)
        st.write(X.head(4))
        return model_pipeline.predict(X)

'''



'''
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class RegressionModels:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {
            'Linear Regression': LinearRegression(),
            'Polynomial Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'ElasticNet Regression': ElasticNet(),
            'Logistic Regression': LogisticRegression(),
            'Decision Tree Regression': DecisionTreeRegressor(),
            'Random Forest Regression': RandomForestRegressor(),
            'Gradient Boosting Regression': GradientBoostingRegressor(),
            'Support Vector Regression (SVR)': SVR(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor()
        }
        
    def add_data(self, X, y):
        self.data = (X, y)
        
    def split_data(self, test_size=0.2, random_state=None):
        if self.data is None:
            raise ValueError("No data provided. Use add_data method to add data first.")
        X, y = self.data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def fit(self, model_name):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split. Use split_data method to split data into training and testing sets.")
        model = self.models[model_name]
        model.fit(self.X_train, self.y_train)
        
    def train(self, model_name):
        if self.X_train is None or self.y_train is None or self.X_test is None:
            raise ValueError("Data not split. Use split_data method to split data into training and testing sets.")
        model = self.models[model_name]
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return y_pred
        
    def evaluate(self, model_name):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not split. Use split_data method to split data into training and testing sets.")
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, r2
        
    def predict(self, model_name, X):
        model = self.models[model_name]
        return model.predict(X)
'''