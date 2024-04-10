import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Check if the 'visualizations' folder exists, if not, create it
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

data = pd.read_excel("Sample Data for shortlisting.xlsx")

# Data Preprocessing

# Convert categorical variables into numerical representation
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])


# Define features for demographic distribution analysis
demographic_features = ['Gender', 'Marital Status', 'Age']

# Define features for employment details exploration
employment_features = ['Gender', 'Role', 'Education', 'Household Income']

# Define features for investment behavior insights
investment_features = ['Percentage of Investment', 'Source of Awareness about Investment', 
                       'Knowledge level about different investment product', 'Knowledge level about sharemarket', 'Knowledge about Govt. Schemes', 'Investment Influencer', 'Investment Experience', 'Risk Level', 'Reason for Investment']

# Prepare data for each analysis
X_demographic = data[demographic_features].drop(columns='Gender', axis=1)
X_employment = data[employment_features].drop(columns='Education', axis=1)
X_investment = data[investment_features].drop(columns='Risk Level', axis=1)

# Define target variables for each analysis
y_demographic = data['Gender']
y_employment = data['Education']
y_investment = data['Risk Level']

# Split the data into training and testing sets for each analysis
scaler = StandardScaler()

X_demo_train, X_demo_test, y_demo_train, y_demo_test = train_test_split(X_demographic, y_demographic, test_size=0.2, random_state=42)
X_demo_train_scaled, X_demo_test_scaled = scaler.fit_transform(X_demo_train), scaler.transform(X_demo_test)

X_emp_train, X_emp_test, y_emp_train, y_emp_test = train_test_split(X_employment, y_employment, test_size=0.2, random_state=42)
X_emp_train_scaled, X_emp_test_scaled = scaler.fit_transform(X_emp_train), scaler.transform(X_emp_test)

X_inv_train, X_inv_test, y_inv_train, y_inv_test = train_test_split(X_investment, y_investment, test_size=0.2, random_state=42)
X_inv_train_scaled, X_inv_test_scaled = scaler.fit_transform(X_inv_train), scaler.transform(X_inv_test)

# Train MLP classifiers for each analysis
mlp_demo_classifier = MLPClassifier(hidden_layer_sizes=(100,50))
mlp_demo_classifier.fit(X_demo_train_scaled, y_demo_train)

mlp_emp_classifier = MLPClassifier(hidden_layer_sizes=(100,50))
mlp_emp_classifier.fit(X_emp_train_scaled, y_emp_train)

mlp_inv_classifier = MLPClassifier(hidden_layer_sizes=(100,50))
mlp_inv_classifier.fit(X_inv_train_scaled, y_inv_train)

# Predictions for each analysis
demo_pred = mlp_demo_classifier.predict(X_demo_test_scaled)
emp_pred = mlp_emp_classifier.predict(X_emp_test_scaled)
inv_pred = mlp_inv_classifier.predict(X_inv_test_scaled)

# Evaluate the models
demo_accuracy = accuracy_score(y_demo_test, demo_pred)
emp_accuracy = accuracy_score(y_emp_test, emp_pred)
inv_accuracy = accuracy_score(y_inv_test, inv_pred)

print("Demographic Distribution Analysis Accuracy:", demo_accuracy)
print("Employment Details Exploration Accuracy:", emp_accuracy)
print("Investment Behavior Insights Accuracy:", inv_accuracy)

# Visualization folder path
visualization_folder = 'visualizations/'

# Visualize and save for Demographic Distribution Analysis
plt.figure(figsize=(10, 6))
cm_demo = confusion_matrix(y_demo_test, demo_pred)
sns.heatmap(cm_demo, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['Gender'].classes_, yticklabels=label_encoders['Gender'].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Demographic Distribution Analysis')
plt.savefig(visualization_folder + 'demo_confusion_matrix.png')
plt.close()

# Visualize and save for Employment Details Exploration
plt.figure(figsize=(10, 6))
cm_emp = confusion_matrix(y_emp_test, emp_pred)
sns.heatmap(cm_emp, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['Education'].classes_, yticklabels=label_encoders['Education'].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Employment Details Exploration')
plt.savefig(visualization_folder + 'emp_confusion_matrix.png')
plt.close()

# Visualize and save for Investment Behavior Insights
plt.figure(figsize=(10, 6))
cm_inv = confusion_matrix(y_inv_test, inv_pred)
sns.heatmap(cm_inv, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['Risk Level'].classes_, yticklabels=label_encoders['Risk Level'].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Investment Behavior Insights')
plt.savefig(visualization_folder + 'inv_confusion_matrix.png')
plt.close()

# For MLP, let's visualize the weights connecting the input features to the first hidden layer

# Demographic Distribution Analysis
plt.figure(figsize=(10, 6))
weights_demo = mlp_demo_classifier.coefs_[0].T  # Transpose weights matrix
plt.barh(range(len(X_demographic.columns)), weights_demo.mean(axis=0))
plt.yticks(range(len(X_demographic.columns)), X_demographic.columns)
plt.xlabel('Average Weight')
plt.ylabel('Feature')
plt.title('Weights from Input Features to First Hidden Layer (Demographic Analysis)')
plt.savefig(visualization_folder + 'demo_weights_first_hidden_layer.png')
plt.close()

# Employment Details Exploration
plt.figure(figsize=(10, 6))
weights_emp = mlp_emp_classifier.coefs_[0].T  # Transpose weights matrix
plt.barh(range(len(X_employment.columns)), weights_emp.mean(axis=0))
plt.yticks(range(len(X_employment.columns)), X_employment.columns)
plt.xlabel('Average Weight')
plt.ylabel('Feature')
plt.title('Weights from Input Features to First Hidden Layer (Employment Details)')
plt.savefig(visualization_folder + 'emp_weights_first_hidden_layer.png')
plt.close()

# Investment Behavior Insights
plt.figure(figsize=(10, 6))
weights_inv = mlp_inv_classifier.coefs_[0].T  # Transpose weights matrix
plt.barh(range(len(X_investment.columns)), weights_inv.mean(axis=0))
plt.yticks(range(len(X_investment.columns)), X_investment.columns)
plt.xlabel('Average Weight')
plt.ylabel('Feature')
plt.title('Weights from Input Features to First Hidden Layer (Investment Behavior)')
plt.savefig(visualization_folder + 'inv_weights_first_hidden_layer.png')
plt.close()
