# --------------------------
# 1. Load Necessary Libraries
# --------------------------

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
