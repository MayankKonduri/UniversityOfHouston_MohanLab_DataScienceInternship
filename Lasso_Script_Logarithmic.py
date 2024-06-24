import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load Dataset from Spreadsheet
data_path = '/Users/mayank/Downloads/PracticeLasso.xlsx'
data = pd.read_excel(data_path, sheet_name='Samples w 20')

# Drop rows with missing 'Disease Group'
data = data.dropna(subset=['Disease Group'])

# Define Explanatory Variables (biomarkers) and Response variable
X = data.iloc[:, 2:].values  # Only Biomarker Columns
y = data['Disease Group'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.37, random_state=1)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with L1 regularization using One-vs-Rest scheme
log_reg = LogisticRegression(penalty='l1', solver='saga', multi_class='ovr', max_iter=100000000, random_state=0)
log_reg.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = log_reg.predict(X_test_scaled)

# Calculate accuracy directly from confusion matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))

# Coefficients
coefficients = pd.Series(log_reg.coef_[0], index=data.columns[2:]).sort_values()

# Perform LassoCV for best alpha value selection with increased max_iter and tol
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, random_state=0, max_iter=10000, tol=0.01).fit(X_train_scaled, y_train)
best_alpha = lasso_cv.alpha_

# Plot the cross-validation curve
mse_path = lasso_cv.mse_path_

# Create a scrollable output widget
output = widgets.Output(layout={'border': '1px solid black', 'width': '100%', 'height': '800px', 'overflow_y': 'scroll'})

with output:
    # Visualize all plots together
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    # Confusion Matrix
    disp.plot(cmap=plt.cm.Blues, ax=axes[0, 0], colorbar=False)
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].text(0.5, -0.3, f"Accuracy: {accuracy:.2f}", ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)

    # All Coefficients
    coefficients.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Logistic Regression Coefficients (L1 Regularization)')
    axes[0, 1].set_xlabel('Biomarkers')
    axes[0, 1].set_ylabel('Coefficient Value')

    # Non-zero Coefficients
    non_zero_coefficients = coefficients[coefficients != 0]
    non_zero_coefficients.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Non-zero Coefficients')
    axes[1, 0].set_xlabel('Biomarkers')
    axes[1, 0].set_ylabel('Coefficient Value')

    # Actual vs Predicted
    axes[1, 1].scatter(y_test, y_pred, alpha=0.3)
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    axes[1, 1].set_xlabel('Actual Disease Group')
    axes[1, 1].set_ylabel('Predicted Disease Group')
    axes[1, 1].set_title('Actual vs Predicted Disease Group')

    # Cross-Validation Curve for Lasso Regression
    axes[2, 0].plot(lasso_cv.alphas_, mse_path.mean(axis=1), marker='o')
    axes[2, 0].set_xscale('log')
    axes[2, 0].set_xlabel('Alpha (log scale)')
    axes[2, 0].set_ylabel('Mean Squared Error')
    axes[2, 0].set_title('Cross-Validation Curve for Lasso Regression')
    axes[2, 0].grid(True)

    # Display best alpha from LassoCV
    axes[2, 1].text(0.5, 0.5, f"Best Alpha from LassoCV:\n{best_alpha}", ha='center', va='center', fontsize=12)
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Display the scrollable output
display(output)

