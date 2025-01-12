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
