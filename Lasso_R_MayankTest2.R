install.packages("readxl")
install.packages("dplyr")
install.packages("caret")
install.packages("glmnet")
install.packages("ggplot2")
install.packages("tibble")

# Step 1: Load and Prepare Data
library(readxl)
library(dplyr)

# Load Dataset from Spreadsheet
data <- read_excel('/Users/mayank/Downloads/PracticeLasso.xlsx', sheet = 'Samples w 20')

# Drop rows with missing 'Disease Group'
data <- na.omit(data)

# Define Explanatory Variables (biomarkers) and Response variable
X <- as.matrix(data[, 3:ncol(data)])  # Only biomarker columns
y <- data$`Disease Group`

# Step 2: Split Data into Training and Testing Sets
set.seed(123)  # for reproducibility
train_indices <- sample(1:nrow(data), 0.65 * nrow(data))  # 65% train, 35% test
X_train <- X[train_indices, ]
y_train <- y[train_indices]
X_test <- X[-train_indices, ]
y_test <- y[-train_indices]

# Step 3: Scale Data
library(caret)

# Standardize features
preproc <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preproc, X_train)
X_test_scaled <- predict(preproc, X_test)

# Step 4: Lasso Regression with Cross-Validation
library(glmnet)

# Perform cross-validation to find optimal lambda value
cv_model <- cv.glmnet(X_train_scaled, y_train, alpha = 1.0)  # alpha = 1 for Lasso

# Find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda
print(paste("Optimal lambda: ", best_lambda))


# Step 5: Fit Lasso Regression Model
best_model <- glmnet(X_train_scaled, y_train, alpha = 1, lambda = best_lambda)

# Step 6: Predict on Training and Testing Sets
train_predictions <- predict(best_model, s = best_lambda, newx = X_train_scaled)
test_predictions <- predict(best_model, s = best_lambda, newx = X_test_scaled)

# Step 7: Calculate MSE for Training and Testing Sets
train_mse <- mean((y_train - train_predictions)^2)
test_mse <- mean((y_test - test_predictions)^2)
print(paste("Training MSE: ", train_mse))
print(paste("Testing MSE: ", test_mse))

# Step 8: Calculate R-squared for Training and Testing Sets
train_r2 <- 1 - (sum((train_predictions - y_train)^2) / sum((y_train - mean(y_train))^2))
test_r2 <- 1 - (sum((test_predictions - y_test)^2) / sum((y_test - mean(y_test))^2))
print(paste("Training R-squared: ", train_r2))
print(paste("Testing R-squared: ", test_r2))

# Step 9: Extract Coefficients and Plot
library(ggplot2)
library(tibble)

# Convert coefficients to a regular matrix
coef_matrix <- as.matrix(coef(best_model, s = best_lambda))
# Create a tibble and add biomarker names
coef_df <- as_tibble(coef_matrix, rownames = "Biomarker")
colnames(coef_df)[2] <- "value"

# Match the order of biomarkers plotted according to '.py' Code
biomarker_order <- c("SLEDAI", "L-selectin ng/mg", "TWEAK", "IL-2rβ pg/mg", "NCAM1 ng/mg", "MCSFR ng/mg",
                     "LAIR2 ng/mg", "Ferritin ng/mg", "FCLR5 ng/mg", "FCGR2a ng/mg", "MCP-1", "Hemopexin",
                     "PF-4", "Timp1", "VCAM-1", "KIM-1", "ALCAM", "rSLEDAI", "Cystatin-C", "CD163 pg/mg",
                     "TGFβ pg/mg", "CD36 ng/mg", "NGAL (lipocalin2)")

coef_df <- coef_df %>%
  mutate(Biomarker = factor(Biomarker, levels = biomarker_order)) %>%
  arrange(Biomarker)

# Plot coefficients
plot <- ggplot(coef_df, aes(x = Biomarker, y = value)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Lasso Regression Coefficients for Biomarkers",
       x = "Biomarker", y = "Coefficient")

# Save the plot as PNG
ggsave("/Users/mayank/Downloads/R_Mayank_LassoTask#1.png", plot, width = 10, height = 6, units = "in")
