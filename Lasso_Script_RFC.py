import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LassoCV

# Load Dataset from Spreadsheet
data_path = '/Users/mayank/Downloads/PracticeLasso.xlsx'
data = pd.read_excel(data_path, sheet_name='Samples w 20')

# Drop rows with missing 'Disease Group'
data = data.dropna(subset=['Disease Group'])

# Define Explanatory Variables (biomarkers) and Response variable
X = data.iloc[:, 2:].values  # Only Biomarker Columns
y = data['Disease Group'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform LassoCV for feature selection
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, random_state=0).fit(X_train_scaled, y_train)
selected_features = np.where(lasso_cv.coef_ != 0)[0]

# Define and train Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train_scaled[:, selected_features], y_train)

# Predict and evaluate
y_pred_rf = rf.predict(X_test_scaled[:, selected_features])
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf}")

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=np.unique(y))

# Create a scrollable output widget for plots
output_rf = widgets.Output(layout={'border': '1px solid black', 'width': '100%', 'height': '800px', 'overflow_y': 'scroll'})

with output_rf:
    # Visualize all plots together
    fig_rf, axes_rf = plt.subplots(2, 2, figsize=(12, 12))

    # Confusion Matrix
    disp_rf.plot(cmap=plt.cm.Blues, ax=axes_rf[0, 0], colorbar=False)
    axes_rf[0, 0].set_title('Random Forest Confusion Matrix')

    # Feature Importances
    importances_rf = rf.feature_importances_
    indices_rf = np.argsort(importances_rf)[::-1]
    feature_names = data.columns[2:]
    axes_rf[0, 1].bar(range(X.shape[1]), importances_rf[indices_rf], align="center")
    axes_rf[0, 1].set_title('Random Forest Feature Importances')
    axes_rf[0, 1].set_xticks(range(X.shape[1]))
    axes_rf[0, 1].set_xticklabels(feature_names[indices_rf], rotation=90)
    axes_rf[0, 1].set_xlim([-1, X.shape[1]])

    # Actual vs Predicted
    axes_rf[1, 0].scatter(y_test, y_pred_rf, alpha=0.3)
    axes_rf[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    axes_rf[1, 0].set_xlabel('Actual Disease Group')
    axes_rf[1, 0].set_ylabel('Predicted Disease Group')
    axes_rf[1, 0].set_title('Actual vs Predicted Disease Group (Random Forest)')

    plt.tight_layout()
    plt.show()

# Display the scrollable output
display(output_rf)

