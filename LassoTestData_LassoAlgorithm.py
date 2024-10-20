import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

# Corrected Dataset Path and Sheet Name
data_path = '/Users/mayank/Downloads/SemesterMohanLab/Processed_LassoTestData.xlsx'
sheet_name = 'Sheet1'  # Update this if the sheet name is different

# Load Dataset from Spreadsheet
data = pd.read_excel(data_path, sheet_name=sheet_name)

# Drop rows with missing 'Groups'
data = data.dropna(subset=['Group'])

# Define Explanatory Variables (biomarkers) and Response variable
biomarker_columns = data.columns[1:]  # All columns except the first one
X = data.loc[:, biomarker_columns].values  # Only Biomarker Columns
y = data['Group'].values

# Split the data into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform LassoCV for feature selection on entire training set
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, random_state=0, max_iter=10000, tol=0.01).fit(X_train_scaled, y_train)
best_alpha = lasso_cv.alpha_

# Get coefficients and their indices
coefficients = lasso_cv.coef_
sorted_indices = np.argsort(np.abs(coefficients))[::-1]  # Sort indices by absolute value of coefficients

# Define top-N ranges
top_n_ranges = [3, 5, 10]

# Initialize lists to store results
results = {}

# Extract feature names and coefficients
feature_names = np.array(biomarker_columns)

# Store Lasso coefficients for all features
lasso_coefficients = pd.Series(coefficients, index=feature_names)

# Function to create and print the logistic regression equation for Lasso coefficients > 0
def print_lasso_equation(top_n, selected_features):
    # Filter for non-zero Lasso coefficients
    non_zero_coefs = coefficients[selected_features]
    non_zero_indices = np.where(non_zero_coefs != 0)[0]
    
    # Construct the equation using only non-zero coefficients
    equation_terms = [f"{non_zero_coefs[i]:.4f} * {feature_names[selected_features[i]]}" for i in non_zero_indices]
    equation = " + ".join(equation_terms)
    
    return f"\nLasso Regression Equation for Top {top_n} Features:\n" \
           f"Logit(Probability of Active LN) = {equation}" if equation_terms else \
           f"No non-zero coefficients for Top {top_n} features."

# Train and evaluate models using top-N features
with open('/Users/mayank/Downloads/SemesterMohanLab/TrialLassoResults/Trial_Lasso_Equations.txt', 'w') as file:
    for top_n in top_n_ranges:
        # Select top-N features based on coefficients
        selected_features = sorted_indices[:top_n]
        X_train_top_n = X_train_scaled[:, selected_features]
        X_test_top_n = X_test_scaled[:, selected_features]
        
        # Train Logistic Regression model with selected features
        log_reg = LogisticRegression(penalty='l1', solver='saga', max_iter=1000000, C=1/best_alpha, random_state=0)
        log_reg.fit(X_train_top_n, y_train)
        
        # Predict and evaluate
        y_pred = log_reg.predict(X_test_top_n)
        y_pred_proba = log_reg.predict_proba(X_test_top_n)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[top_n] = {
            'accuracy': accuracy,
            'auc': auc_score,
            'num_zero_coefficients': np.sum(coefficients[selected_features] == 0),
            'coefficients': lasso_coefficients.loc[feature_names[sorted_indices[:top_n]]]
        }
        
        # Print Lasso regression equation based on non-zero coefficients
        file.write(print_lasso_equation(top_n, selected_features))
        
        # Print additional metrics
        file.write(f"\nTop {top_n} Features:\n")
        file.write(f"Accuracy: {results[top_n]['accuracy']:.2f}\n")
        file.write(f"AUC: {results[top_n]['auc']:.2f}\n")
        file.write(f"Number of zero coefficients: {results[top_n]['num_zero_coefficients']}\n")
        
        # Load additional AUC data
        auc_data_path = '/Users/mayank/Downloads/SemesterMohanLab/Processed_LassoTestData_test_1_2_output_plottable.xlsx'
        auc_data = pd.read_excel(auc_data_path, sheet_name='Sheet1')

        # Ensure 'Protein' and 'AUC value' columns are correctly named
        auc_data = auc_data.rename(columns=lambda x: x.strip())  # Strip any leading/trailing whitespace from column names
        individual_auc_values = auc_data['AUC value'].values

        # Append AUC comparison results
        combined_auc = results[top_n]['auc']
        greater_than_combined_auc = np.sum(individual_auc_values > combined_auc)
        
        file.write(f"Number of individual AUCs greater than the combined AUC for Top {top_n}: {greater_than_combined_auc}\n")
        file.write("\n")

# Plot the results
fig, axes = plt.subplots(len(top_n_ranges), 2, figsize=(12, 6 * len(top_n_ranges)))

for i, top_n in enumerate(top_n_ranges):
    # Confusion Matrix for top-N features
    cm = confusion_matrix(y_test, log_reg.predict(X_test_top_n))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues, ax=axes[i, 0], colorbar=False)
    axes[i, 0].set_title(f'Confusion Matrix (Top {top_n} Features)')
    
    # Coefficients for top-N features
    result = results[top_n]
    result['coefficients'].plot(kind='bar', ax=axes[i, 1])
    axes[i, 1].set_title(f'Logistic Regression Coefficients (Top {top_n} Features)')
    axes[i, 1].set_xlabel('Biomarkers')
    axes[i, 1].set_ylabel('Coefficient Value')

plt.tight_layout()

# Save the plot to a file
plt.savefig('/Users/mayank/Downloads/SemesterMohanLab/TrialLassoResults/Trial_Lasso_Binary_Classification_Top_N.png')

# Export coefficients to Excel
with pd.ExcelWriter('/Users/mayank/Downloads/SemesterMohanLab/TrialLassoResults/Trial_Lasso_Coefficients_Top_N.xlsx') as writer:
    for top_n, result in results.items():
        result['coefficients'].to_excel(writer, sheet_name=f'Top_{top_n}_Features')

plt.close(fig)

print("Plots saved to '/Users/mayank/Downloads/SemesterMohanLab/TrialLassoResults/Trial_Lasso_Binary_Classification_Top_N.png'.")
print("Coefficients saved to '/Users/mayank/Downloads/SemesterMohanLab/TrialLassoResults/Trial_Lasso_Coefficients_Top_N.xlsx'.")
print("Equations saved to '/Users/mayank/Downloads/SemesterMohanLab/TrialLassoResults/Trial_Lasso_Equations.txt'.")

