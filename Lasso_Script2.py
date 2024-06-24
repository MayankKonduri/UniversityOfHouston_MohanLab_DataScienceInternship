import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error

# Load Dataset from Spreadsheet
data_path = '/Users/mayank/Downloads/PracticeLasso.xlsx'
data = pd.read_excel(data_path, sheet_name='Samples w 20')

# Drop rows with missing 'Disease Group'
data = data.dropna(subset=['Disease Group'])

# Define Explanatory Variables (biomarkers) and Response variable
X = data.iloc[:, 2:].values  # Only Biomarker Columns
y = data['Disease Group'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso Cross-validation to find the best alpha
lasso_cv = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10], random_state=0).fit(X_train_scaled, y_train)
best_alpha = lasso_cv.alpha_
print(f"Best alpha from LassoCV: {best_alpha}")

# Fit Lasso regression with the best alpha
lasso = Lasso(alpha=best_alpha)
lasso.fit(X_train_scaled, y_train)

# Train and test scores
train_score = lasso.score(X_train_scaled, y_train)
test_score = lasso.score(X_test_scaled, y_test)
print(f"Lasso train score: {train_score}")
print(f"Lasso test score: {test_score}")

# Coefficients
coefficients = pd.Series(lasso.coef_, index=data.columns[2:]).sort_values()
print(coefficients)

# Visualize Coefficients as a bar plot
plt.figure(figsize=(12, 8))
plt.subplot(221)
coefficients.plot(kind='bar')
plt.title('Lasso Regression Coefficients')
plt.xlabel('Biomarkers')
plt.ylabel('Coefficient Value')

# Residual Plot
y_pred = lasso.predict(X_test_scaled)
residuals = y_test - y_pred
plt.subplot(222)
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

# Actual vs Predicted plot
plt.subplot(223)
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')

# Cross-Validation Curve
alphas = lasso_cv.alphas_
cv_scores = lasso_cv.mse_path_.mean(axis=1)
plt.subplot(224)
plt.plot(alphas, cv_scores, marker='o')
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation Curve for Lasso Regression')
plt.grid(True)

plt.tight_layout()
plt.savefig('/Users/mayank/Downloads/lasso_plots.png')  # Save the plot as an image file
plt.show()

