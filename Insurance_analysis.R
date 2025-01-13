# --------------------------
# 1. Install and Load Necessary Libraries
# --------------------------

install.packages("tidyverse") # For data manipulation and visualization
install.packages("caret") # For data splitting
install.packages("randomForest") # For Random Forest model
install.packages("gbm") # For Gradient Boosting model
install.packages("ggplot2") # For data visualization

library(tidyverse)  # For data manipulation and visualization
library(caret)      # For data splitting
library(randomForest)  # For Random Forest model
library(gbm)        # For Gradient Boosting model
library(ggplot2)    # For data visualization

# --------------------------
# 2. Data Loading and Preprocessing
# --------------------------

# Load the dataset
data <- read.csv("Data/Insurance.csv")

# Display the first few rows to understand the data structure
head(data)

# Check for missing values
missing_values <- colSums(is.na(data))
print("Missing values in each column:")
print(missing_values)

# Handle missing values (if any)
# Option 1: Remove rows with missing values
data <- data %>% na.omit()

# Option 2: Fill missing values with mean/mode (example for numeric columns)
# data <- data %>%
#   mutate(
#     age = ifelse(is.na(age), mean(age, na.rm = TRUE), age),
#     bmi = ifelse(is.na(bmi), mean(bmi, na.rm = TRUE), bmi),
#     charges = ifelse(is.na(charges), mean(charges, na.rm = TRUE), charges)
#   )

# Convert categorical variables to factors
data <- data %>%
  mutate(
    sex = as.factor(sex),
    smoker = as.factor(smoker),
    region = as.factor(region)
  )

# Check the structure of the updated data
str(data)

# Summary of the dataset to verify changes
summary(data)


# --------------------------
# 3. Exploratory Data Analysis (EDA)
# --------------------------

# Pairwise scatter plots
pairs(data %>% select(age, bmi, children, charges),
      main = "Pairwise Scatter Plots",
      col = ifelse(data$smoker == "yes", "red", "blue"),
      labels = c("Age", "BMI", "Children", "Charges"),  # Add axis labels
      cex.labels = 1.2,  # Increase label size
      pch = 16,          # Use solid points
      cex = 0.8,         # Adjust point size
      gap = 0.5)         # Add space between plots

# Histogram of the distribution of charges
ggplot(data, aes(x = charges)) +
  geom_histogram(binwidth = 2000, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of Insurance Charges",
       x = "Charges",
       y = "Frequency") +
  theme_minimal()

# Scatter plot of charges vs. age with linear regression line
ggplot(data, aes(x = age, y = charges)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Scatter Plot of Charges vs. Age with Regression Line",
       x = "Age",
       y = "Charges") +
  theme_minimal()

# Scatter plot of charges vs. BMI with linear regression line
ggplot(data, aes(x = bmi, y = charges)) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_smooth(method = "lm", color = "blue", se = FALSE) +
  labs(title = "Scatter Plot of Charges vs. BMI with Regression Line",
       x = "BMI",
       y = "Charges") +
  theme_minimal()

# Boxplot of charges vs. smoker
ggplot(data, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Boxplot of Charges by Smoking Status",
       x = "Smoking Status",
       y = "Charges") +
  scale_fill_manual(values = c("no" = "blue", "yes" = "red")) +
  theme_minimal()

# --------------------------
# 4. Feature Engineering
# --------------------------

# Transform categorical variables into dummy variables
data_transformed <- data %>%
  mutate(
    sex_male = ifelse(sex == "male", 1, 0),
    smoker_yes = ifelse(smoker == "yes", 1, 0),
    region_northwest = ifelse(region == "northwest", 1, 0),
    region_northeast = ifelse(region == "northeast", 1, 0),
    region_southeast = ifelse(region == "southeast", 1, 0),
    region_southwest = ifelse(region == "southwest", 1, 0)
  ) %>%
  select(-sex, -smoker, -region)  # Remove original categorical columns

# Print the transformed data
print(head(data_transformed))

# --------------------------
# 5. Predictive Model Development
# --------------------------

# Ensure reproducibility
set.seed(123)

# Split the data into training (80%) and testing (20%) sets
train_indices <- createDataPartition(data_transformed$charges, p = 0.8, list = FALSE)
train_data <- data_transformed[train_indices, ]
test_data <- data_transformed[-train_indices, ]

# Verify the dimensions of the datasets
cat("Training set dimensions: ", dim(train_data), "\n")
cat("Testing set dimensions: ", dim(test_data), "\n")

# Build Linear Regression Model
lm_model <- lm(charges ~ ., data = train_data)
cat("\nLinear Regression Model Summary:\n")
print(summary(lm_model))

# Build Random Forest Model
rf_model <- randomForest(charges ~ ., data = train_data, ntree = 500, importance = TRUE)
cat("\nRandom Forest Model Summary:\n")
print(rf_model)

# Build Gradient Boosting Model
gbm_model <- gbm(
  charges ~ .,                 # Formula for the model
  data = train_data,           # Training data
  distribution = "gaussian",   # For regression
  n.trees = 500,               # Number of trees
  interaction.depth = 3,       # Depth of each tree
  shrinkage = 0.01,            # Learning rate
  cv.folds = 5,                # Perform 5-fold cross-validation
  verbose = FALSE              # Suppress detailed output
)

# Print the best number of trees based on cross-validation
cat("\nOptimal number of trees based on cross-validation:\n")
best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = TRUE)


# Print a summary of the Gradient Boosting model
cat("\nGradient Boosting Model Summary:\n")
print(summary(gbm_model))

# --------------------------
# 6. Predictive Model Selection
# --------------------------

# Make predictions for each model
lm_predictions <- predict(lm_model, newdata = test_data)
rf_predictions <- predict(rf_model, newdata = test_data)
gbm_predictions <- predict(gbm_model, newdata = test_data, n.trees = best_iter)

# Combine actual and predicted values for visualization
lm_results <- data.frame(Actual = test_data$charges, Predicted = lm_predictions)
rf_results <- data.frame(Actual = test_data$charges, Predicted = rf_predictions)
gbm_results <- data.frame(Actual = test_data$charges, Predicted = gbm_predictions)

# Visualize predictions for Linear Regression
ggplot(lm_results, aes(x = Actual, y = Predicted)) +
  geom_point(color = "green", alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, color = "maroon") +
  labs(title = "Linear Regression Predictions vs. Actual Values",
       x = "Actual Charges",
       y = "Predicted Charges") +
  theme_minimal(base_size = 14)

# Visualize predictions for Random Forest
ggplot(rf_results, aes(x = Actual, y = Predicted)) +
  geom_point(color = "black", alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Random Forest Predictions vs. Actual Values",
       x = "Actual Charges",
       y = "Predicted Charges") +
  theme_minimal(base_size = 14)

# Visualize predictions for Gradient Boosting
ggplot(gbm_results, aes(x = Actual, y = Predicted)) +
  geom_point(color = "purple", alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Gradient Boosting Predictions vs. Actual Values",
       x = "Actual Charges",
       y = "Predicted Charges") +
  theme_minimal(base_size = 14)

# Function to calculate RMSE, MAE, and R-squared
calculate_metrics <- function(actual, predicted, model_name) {
  rmse_value <- sqrt(mean((actual - predicted)^2))
  mae_value <- mean(abs(actual - predicted))
  r_squared_value <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)

  cat("\nPerformance Metrics for", model_name, ":\n")
  cat("  RMSE      :", round(rmse_value, 2), "\n")
  cat("  MAE       :", round(mae_value, 2), "\n")
  cat("  R-squared :", round(r_squared_value, 4), "\n")
}

# Evaluate metrics for each model
calculate_metrics(test_data$charges, lm_predictions, "Linear Regression")
calculate_metrics(test_data$charges, rf_predictions, "Random Forest")
calculate_metrics(test_data$charges, gbm_predictions, "Gradient Boosting")

# --------------------------
# 7. Determine Significant Factors
# --------------------------

# 1. Analyse the coefficients from the Linear Regression model and print them
cat("\nLinear Regression Coefficients:\n")
print(summary(lm_model)$coefficients)

# 2. Analyse the feature importance from the Random Forest model and print them
cat("\nRandom Forest Feature Importance:\n")
importance_rf <- importance(rf_model)
print(importance_rf)

# 3. Analyse the feature importance scores from Gradient Boosting and print them
cat("\nGradient Boosting Feature Importance:\n")
importance_gbm <- summary(gbm_model, n.trees = best_iter)
print(importance_gbm)

# Convert to a data frame and rename columns
importance_gbm_df <- as.data.frame(importance_gbm)
colnames(importance_gbm_df) <- c("Feature", "Importance_Score")

# Visualize feature importance as a bar plot
ggplot(importance_gbm_df, aes(x = reorder(Feature, Importance_Score), y = Importance_Score)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +  # Bar plot
  coord_flip() +  # Flip axes for better readability
  labs(
    title = "Gradient Boosting Feature Importance",
    subtitle = "Features vs Importance Scores",
    x = "Features",
    y = "Importance Score"
  ) +
  theme_minimal(base_size = 12) +  # Minimal theme
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),  # Center and bold title
    plot.subtitle = element_text(hjust = 0.5, size = 10, color = "gray40"),  # Center subtitle
    axis.title.x = element_text(size = 10, face = "bold"),  # Bold x-axis label
    axis.title.y = element_text(size = 10, face = "bold"),  # Bold y-axis label
    axis.text.x = element_text(size = 8),  # Adjust x-axis text size
    axis.text.y = element_text(size = 8),  # Adjust y-axis text size
    panel.grid.major = element_line(color = "gray90"),  # Add light grid lines
    panel.grid.minor = element_blank()  # Remove minor grid lines
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))  # Add padding to y-axis


