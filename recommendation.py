
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Assuming you have your data stored in a pandas DataFrame named 'data'
# Load data into DataFrame

dataset = pd.read_excel("Sample Data for shortlisting.xlsx")

# Data Preprocessing

# Convert categorical variables into numerical representation
label_encoders = {}
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        dataset[column] = label_encoders[column].fit_transform(dataset[column])

# Handling missing values
dataset.fillna(dataset.mean(), inplace=True)

# Splitting data into features and target
X = dataset.drop(columns=['S. No.','City','Gender','Marital Status','Age','Education','Role',"Knowledge about Govt. Schemes",'Return Earned'], axis=1)
y = dataset['Return Earned']


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)

# Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(100,50))
nn_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBClassifier(scale_pos_weight = 5, objective = 'binary:logistic')
xgb_model.fit(X_train, y_train)


# Neural Network
nn_pred_prob = nn_model.predict_proba(X_test)
nn_pred = np.argmax(nn_pred_prob, axis=1)
nn_accuracy = accuracy_score(y_test, nn_pred)
nn_precision = precision_score(y_test, nn_pred, average='weighted')
nn_recall = recall_score(y_test, nn_pred, average='weighted')
nn_f1 = f1_score(y_test, nn_pred, average='weighted')
nn_roc_auc = roc_auc_score(y_test, nn_pred_prob, average='weighted', multi_class='ovr')

print("Neural Network Metrics:")
print("Accuracy:", nn_accuracy * 100)
print("Precision:", nn_precision * 100)
print("Recall:", nn_recall * 100)
print("F1-score:", nn_f1 * 100)
print("ROC-AUC:", nn_roc_auc * 100)

# XGBoost
xgb_pred_prob = xgb_model.predict_proba(X_test)
xgb_pred = np.argmax(xgb_pred_prob, axis=1)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred, average='weighted')
xgb_recall = recall_score(y_test, xgb_pred, average='weighted')
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
xgb_roc_auc = roc_auc_score(y_test, xgb_pred_prob, average='weighted', multi_class='ovr')

print("\nXGBoost Metrics:")
print("Accuracy:", xgb_accuracy * 100)
print("Precision:", xgb_precision * 100)
print("Recall:", xgb_recall* 100)
print("F1-score:", xgb_f1* 100)
print("ROC-AUC:", xgb_roc_auc* 100)




print("\n\n")


# Recommendation Generation

def generate_recommendation(input_data):
    input_data_encoded = input_data.copy()
    for column in input_data_encoded.columns:
        if column in label_encoders:
            input_data_encoded[column] = label_encoders[column].transform([input_data_encoded[column]])[0]
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

# Example of generating recommendation for new data
new_data = pd.DataFrame({
                         'Number of investors in family': [2],
                         'Household Income': ['US$ 8206 to US$ 13675'],
                         'Percentage of Investment': ['6% to 10%'],
                         'Source of Awareness about Investment': ['Workers'],
                         'Knowledge level about different investment product': [7],
                         'Knowledge level about sharemarket':[4],
                         'Knowledge about Govt. Schemes':[5],
                         'Investment Influencer':['Friends Reference'],
                         'Investment Experience': ['4 Years to 6 Years'],
                         'Risk Level': ['Medium'],
                         'Reason for Investment':['Fun and Exitement']
                         })

recommendation = generate_recommendation(new_data)
print("Investment Recommendation:", recommendation)


