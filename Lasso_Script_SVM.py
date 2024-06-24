# Import necessary libraries (already imported above)

# Define and train SVM classifier
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_scaled[:, selected_features], y_train)

# Predict and evaluate
y_pred_svm = svm.predict(X_test_scaled[:, selected_features])
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm}")

# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=np.unique(y))

# Create a scrollable output widget for plots
output_svm = widgets.Output(layout={'border': '1px solid black', 'width': '100%', 'height': '800px', 'overflow_y': 'scroll'})

with output_svm:
    # Visualize all plots together
    fig_svm, axes_svm = plt.subplots(2, 2, figsize=(12, 12))

    # Confusion Matrix
    disp_svm.plot(cmap=plt.cm.Blues, ax=axes_svm[0, 0], colorbar=False)
    axes_svm[0, 0].set_title('SVM Confusion Matrix')

    # Actual vs Predicted
    axes_svm[1, 0].scatter(y_test, y_pred_svm, alpha=0.3)
    axes_svm[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    axes_svm[1, 0].set_xlabel('Actual Disease Group')
    axes_svm[1, 0].set_ylabel('Predicted Disease Group')
    axes_svm[1, 0].set_title('Actual vs Predicted Disease Group (SVM)')

    plt.tight_layout()
    plt.show()

# Display the scrollable output
display(output_svm)

