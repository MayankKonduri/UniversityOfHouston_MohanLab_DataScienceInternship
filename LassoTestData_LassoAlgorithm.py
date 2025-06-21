import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

data_path = '/Users/mayank/Downloads/SemesterMohanLab/Processed_LassoTestData.xlsx'
sheet_name = 'Sheet1'

data = pd.read_excel(data_path, sheet_name=sheet_name)
data = data.dropna(subset=['Group'])

biomarker_columns = data.columns[1:]
X = data.loc[:, biomarker_columns].values
y = data['Group'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, random_state=0, max_iter=10000, tol=0.01).fit(X_train_scaled, y_train)
best_alpha = lasso_cv.alpha_

coefficients = lasso_cv.coef_
sorted_indices = np.argsort(np.abs(coefficients))[::-1]

top_n_ranges = [3, 5, 10]
results = {}
feature_names = np.array(biomarker_columns)
lasso_coefficients = pd.Series(coefficients, index=feature_names)

def print_lasso_equation(top_n, selected_features):
    non_zero_coefs = coefficients[selected_features]
    non_zero_indices = np.where(non_zero_coefs != 0)[0]
    equation_terms = [f"{non_zero_coefs[i]:.4f} * {feature_names[selected_features[i]]}" for i in non_zero_indices]
    equation = " + ".join(equation_terms)
    return f"\nLasso Regression Equation for Top {top_n} Features:\nLogit(Probability of Active LN) = {equation}" if equation_terms else f"No non-zero coefficients for Top {top_n} features."

with open('/Users/mayank/Downloads/SemesterMohanLab/TrialLassoResults/Trial_Lasso_Equations.txt', 'w') as file:
    for top_n in top_n_ranges:
        selected_features = sorted_indices[:top_n]
        X_train_top_n = X_train_scaled[:, selected_features]
        X_test_top_n = X_test_scaled[:, selected_features]
        log_reg = LogisticRegression(penalty='l1', solver='saga', max_iter=1000000, C=1/best_alpha, random_state=0)
        log_reg.fit(X_train_top_n, y_train)
        y_pred = log_reg.predict(X_test_top_n)
        y_pred_proba = log_reg.predict_proba(X_test_top_n)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        results[top_n] = {
            'accuracy': accuracy,
            'auc': auc_score,
            'num_zero_coefficients': np.sum(coefficients[selected_features] == 0),
            'coefficients': lasso_coefficients.loc[feature_names[sorted_indices[:top_n]]]
        }
        file.write(print_lasso_equation(top_n, selected_features))
        file.write(f"\nTop {top_n} Features:\n")
        file.write(f"Accuracy: {results[top_n]['accuracy']:.2f}\n")
        file.write(f"AUC: {results[top_n]['auc']:.2f}\n")
        file.write(f"Number of zero coefficients: {results[top_n]['num_zero_coefficients']}\n")
        auc_data_path = '/Users/mayank/Downloads/SemesterMohanLab/Processed_LassoTestData_test_1_2_output_plottable.xlsx'
        auc_data = pd.read_excel(auc_data_path, sheet_name='Sheet1')
        auc_data = auc_data.rename(columns=lambda x: x.strip())
        individual_auc_values = auc_data['AUC value'].values
        combined_auc = results[top_n]['auc']
        greater_than_combined_auc = np.sum(individual_auc_values > combined_auc)
        file.write(f"Number of individual AUCs greater than the combined AUC for Top {top_n}: {greater_than_combined_auc}\n")
        file.write("\n")

fig, axes = plt.subplots(len(top_n_ranges), 2, figsize=(12, 6 * len(top_n_ranges)))

for i, top_n in enumerate(top_n_ranges):
    cm = confusion_matrix(y_test, log_reg.predict(X_test_top_n))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues, ax=axes[i, 0], colorbar=False)
    axes[i, 0].set_title(f'Confusion Matrix (Top {top_n} Features)')
    result = results[top_n]
    result['coefficients'].plot(kind='bar', ax=axes[i, 1])
    axes[i, 1].set_title(f'Logistic Regression Coefficients (Top {top_n} Features)')
    axes[i, 1].set_xlabel('Biomarkers')
    axes[i, 1].set_ylabel('Coefficient Value')

plt.tight_layout()
plt.savefig('/Users/mayank/Downloads/SemesterMohanLab/TrialLassoResults/Trial_Lasso_Binary_Classification_Top_N.png')

with pd.ExcelWriter('/Users/mayank/Downloads/SemesterMohanLab/TrialLassoResults/Trial_Lasso_Coefficients_Top_N.xlsx') as writer:
    for top_n, result in results.items():
        result['coefficients'].to_excel(writer, sheet_name=f'Top_{top_n}_Features')

plt.close(fig)

print("Plots saved to '/Users/mayank/Downloads/SemesterMohanLab/TrialLassoResults/Trial_Lasso_Binary_Classification_Top_N.png'.")
print("Coefficients saved to '/Users/mayank/Downloads/SemesterMohanLab/TrialLassoResults/Trial_Lasso_Coefficients_Top_N.xlsx'.")
print("Equations saved to '/Users/mayank/Downloads/SemesterMohanLab/TrialLassoResults/Trial_Lasso_Equations.txt'.")
