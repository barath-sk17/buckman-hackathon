import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Check if the 'visualizations' folder exists, if not, create it
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Load data
@st.cache_data
def load_data():
    return pd.read_excel("Sample Data for shortlisting.xlsx")
data = load_data()

# Data Preprocessing

# Convert categorical variables into numerical representation
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# Define features and target variables
demographic_features = ['Gender', 'Marital Status', 'Age']
employment_features = ['Gender', 'Role', 'Education', 'Household Income']
investment_features = ['Percentage of Investment', 'Source of Awareness about Investment', 
                       'Knowledge level about different investment product', 'Knowledge level about sharemarket', 'Knowledge about Govt. Schemes', 'Investment Influencer', 'Investment Experience', 'Risk Level', 'Reason for Investment']

# Function to split data and perform scaling
def split_data_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

# Function to train MLP classifier
def train_mlp_classifier(X_train, y_train):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,50))
    mlp_classifier.fit(X_train, y_train)
    return mlp_classifier

# Function to evaluate model
def evaluate_model(y_test, y_pred):
    return accuracy_score(y_test, y_pred)

# Train and evaluate models
X_demographic = data[demographic_features].drop(columns='Gender', axis=1)
y_demographic = data['Gender']
X_demo_train_scaled, X_demo_test_scaled, y_demo_train, y_demo_test = split_data_and_scale(X_demographic, y_demographic)
mlp_demo_classifier = train_mlp_classifier(X_demo_train_scaled, y_demo_train)
demo_pred = mlp_demo_classifier.predict(X_demo_test_scaled)
demo_accuracy = evaluate_model(y_demo_test, demo_pred)

X_employment = data[employment_features].drop(columns='Education', axis=1)
y_employment = data['Education']
X_emp_train_scaled, X_emp_test_scaled, y_emp_train, y_emp_test = split_data_and_scale(X_employment, y_employment)
mlp_emp_classifier = train_mlp_classifier(X_emp_train_scaled, y_emp_train)
emp_pred = mlp_emp_classifier.predict(X_emp_test_scaled)
emp_accuracy = evaluate_model(y_emp_test, emp_pred)

X_investment = data[investment_features].drop(columns='Risk Level', axis=1)
y_investment = data['Risk Level']
X_inv_train_scaled, X_inv_test_scaled, y_inv_train, y_inv_test = split_data_and_scale(X_investment, y_investment)
mlp_inv_classifier = train_mlp_classifier(X_inv_train_scaled, y_inv_train)
inv_pred = mlp_inv_classifier.predict(X_inv_test_scaled)
inv_accuracy = evaluate_model(y_inv_test, inv_pred)

# Create navigation sidebar
page = st.sidebar.radio("Navigation", ["Demographic Distribution Analysis", "Employment Details Exploration", "Investment Behavior Insights", "Recommendation System"])

if page == "Demographic Distribution Analysis":
    st.title("Demographic Distribution Analysis ")
    st.write("")
    st.write("##### Accuracy:", demo_accuracy)

    # Visualize and save for Demographic Distribution Analysis
    plt.figure(figsize=(10, 6))
    cm_demo = confusion_matrix(y_demo_test, demo_pred)
    sns.heatmap(cm_demo, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['Gender'].classes_, yticklabels=label_encoders['Gender'].classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Demographic Distribution Analysis')
    st.pyplot(plt)

    # Demographic Distribution Analysis
    plt.figure(figsize=(10, 6))
    weights_demo = mlp_demo_classifier.coefs_[0].T  # Transpose weights matrix
    plt.barh(range(len(X_demographic.columns)), weights_demo.mean(axis=0), color='skyblue')
    plt.yticks(range(len(X_demographic.columns)), X_demographic.columns)
    plt.xlabel('Average Weight')
    plt.ylabel('Feature')
    plt.title('Weights from Input Features to First Hidden Layer (Demographic Analysis)')
    st.pyplot(plt)

elif page == "Employment Details Exploration":
    st.title("Employment Details Exploration ")
    st.write("")
    st.write("##### Accuracy:", emp_accuracy)

    # Visualize and save for Employment Details Exploration
    plt.figure(figsize=(10, 6))
    cm_emp = confusion_matrix(y_emp_test, emp_pred)
    sns.heatmap(cm_emp, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['Education'].classes_, yticklabels=label_encoders['Education'].classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Employment Details Exploration')
    st.pyplot(plt)

    # Employment Details Exploration
    plt.figure(figsize=(10, 6))
    weights_emp = mlp_emp_classifier.coefs_[0].T  # Transpose weights matrix
    plt.barh(range(len(X_employment.columns)), weights_emp.mean(axis=0), color='lightgreen')
    plt.yticks(range(len(X_employment.columns)), X_employment.columns)
    plt.xlabel('Average Weight')
    plt.ylabel('Feature')
    plt.title('Weights from Input Features to First Hidden Layer (Employment Details)')
    st.pyplot(plt)

elif page == "Investment Behavior Insights":
    st.title("Investment Behavior Insights ")
    st.write("")
    st.write("##### Accuracy:", inv_accuracy)

    # Visualize and save for Investment Behavior Insights
    plt.figure(figsize=(10, 6))
    cm_inv = confusion_matrix(y_inv_test, inv_pred)
    sns.heatmap(cm_inv, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['Risk Level'].classes_, yticklabels=label_encoders['Risk Level'].classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Investment Behavior Insights')
    st.pyplot(plt)

    # Investment Behavior Insights
    plt.figure(figsize=(10, 6))
    weights_inv = mlp_inv_classifier.coefs_[0].T  # Transpose weights matrix
    plt.barh(range(len(X_investment.columns)), weights_inv.mean(axis=0), color='lightcoral')
    plt.yticks(range(len(X_investment.columns)), X_investment.columns)
    plt.xlabel('Average Weight')
    plt.ylabel('Feature')
    plt.title('Weights from Input Features to First Hidden Layer (Investment Behavior)')
    st.pyplot(plt)

else:

    def generate_recommendation(input_data, dataset, scaler, label_encoders, nn_model, xgb_model, nn_accuracy, xgb_accuracy):
        input_data_encoded = input_data.copy()
        for column in input_data_encoded.columns:
            if column in label_encoders:
                input_data_encoded[column] = label_encoders[column].transform(input_data_encoded[column])
        input_data_scaled = scaler.transform(input_data_encoded)
        
        nn_recommendation = nn_model.predict(input_data_scaled.reshape(1, -1))[0]
        xgb_recommendation = xgb_model.predict(input_data_scaled.reshape(1, -1))[0]
        
        if nn_recommendation == xgb_recommendation:
            return nn_recommendation
        else:
            # You can implement logic here for combining recommendations from both models
            # For simplicity, returning recommendation from the model with higher accuracy
            if nn_accuracy > xgb_accuracy:
                return nn_recommendation
            else:
                return xgb_recommendation

    st.title("Recommendation System")

    X = data.drop(columns=['S. No.','City','Gender','Marital Status','Age','Education','Role','Return Earned'], axis=1)
    y = data['Return Earned']

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)

    # Neural Network
    nn_model = MLPClassifier(hidden_layer_sizes=(100,50))
    nn_model.fit(X_train, y_train)

    # XGBoost
    xgb_model = XGBClassifier(scale_pos_weight=5, objective='binary:logistic')
    xgb_model.fit(X_train, y_train)

    # Neural Network
    nn_pred_prob = nn_model.predict_proba(X_test)
    nn_pred = np.argmax(nn_pred_prob, axis=1)
    nn_accuracy = accuracy_score(y_test, nn_pred)
    nn_precision = precision_score(y_test, nn_pred, average='weighted')
    nn_recall = recall_score(y_test, nn_pred, average='weighted')
    nn_f1 = f1_score(y_test, nn_pred, average='weighted')
    nn_roc_auc = roc_auc_score(y_test, nn_pred_prob, average='weighted', multi_class='ovr')

    st.write("### Neural Network Metrics:")
    details1 = {"Accuracy": nn_accuracy * 100, "Precision": nn_precision * 100, "Recall": nn_recall * 100, "F1-score": nn_f1 * 100, "ROC-AUC": nn_roc_auc * 100}
    st.write(details1)

    # XGBoost
    xgb_pred_prob = xgb_model.predict_proba(X_test)
    xgb_pred = np.argmax(xgb_pred_prob, axis=1)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_precision = precision_score(y_test, xgb_pred, average='weighted')
    xgb_recall = recall_score(y_test, xgb_pred, average='weighted')
    xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
    xgb_roc_auc = roc_auc_score(y_test, xgb_pred_prob, average='weighted', multi_class='ovr')

    st.write("### XGBoost Metrics:")
    details2 = {"Accuracy": xgb_accuracy * 100, "Precision": xgb_precision * 100, "Recall": xgb_recall * 100, "F1-score": xgb_f1 * 100, "ROC-AUC": xgb_roc_auc * 100}
    st.write(details2)

    st.write("### Data Preview:")
    st.write(data.head())

    num_investors = st.number_input("Number of investors in family", 2)
    household_income = st.text_input("Household Income", 'US$ 8206 to US$ 13675')
    percentage_investment = st.text_input("Percentage of Investment", '6% to 10%')
    source_awareness = st.text_input("Source of Awareness about Investment", 'Workers')
    knowledge_investment = st.number_input("Knowledge level about different investment product", 7)
    knowledge_sharemarket = st.number_input("Knowledge level about sharemarket", 4)
    knowledge_schemes = st.number_input("Knowledge about Govt. Schemes", 5)
    investment_influencer = st.text_input("Investment Influencer", 'Friends Reference')
    investment_experience = st.text_input("Investment Experience", '4 Years to 6 Years')
    risk_level = st.text_input("Risk Level", 'Medium')
    reason_investment = st.text_input("Reason for Investment", 'Fun and Exitement')

    # Create a DataFrame with the input data
    new_data = pd.DataFrame({
        'Number of investors in family': [num_investors],
        'Household Income': [str(household_income)],
        'Percentage of Investment': [percentage_investment],
        'Source of Awareness about Investment': [source_awareness],
        'Knowledge level about different investment product': [knowledge_investment],
        'Knowledge level about sharemarket': [knowledge_sharemarket],
        'Knowledge about Govt. Schemes': [knowledge_schemes],
        'Investment Influencer': [investment_influencer],
        'Investment Experience': [investment_experience],
        'Risk Level': [risk_level],
        'Reason for Investment': [reason_investment]
    })

    # Generate recommendation
    recommendation = generate_recommendation(new_data, data, scaler, label_encoders, nn_model, xgb_model, nn_accuracy, xgb_accuracy)

    # Display recommendation
    st.write("### Return Earned :", recommendation)


    def plot_confusion_matrix(model, X_test, y_test, model_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix for {model_name}')
        st.pyplot(fig)

    plot_confusion_matrix(nn_model, X_test, y_test, 'Neural Network')
    plot_confusion_matrix(xgb_model, X_test, y_test, 'XGBoost')

    # Feature Importance Visualization for Neural Network
    def plot_nn_feature_importance(model, features):
        # Extracting weights from the input layer to the first hidden layer
        weights_input_hidden = model.coefs_[0]
        
        # Transpose the weights array to match the features
        weights_input_hidden = weights_input_hidden.T
        
        # Calculating mean importance for each feature
        feature_importance = np.abs(weights_input_hidden).mean(axis=0)
        
        # Sorting features by importance
        sorted_idx = np.argsort(feature_importance)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([features[i] for i in sorted_idx])
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importance for Neural Network')
        st.pyplot(fig)

    # Call the function with the correct number of features
    plot_nn_feature_importance(nn_model, X.columns)


    # Feature Importance Visualization for XGBoost
    def plot_feature_importance(model, features):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([features[i] for i in sorted_idx])
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importance for XGBoost')
        st.pyplot(fig)

    plot_feature_importance(xgb_model, X.columns)


    
