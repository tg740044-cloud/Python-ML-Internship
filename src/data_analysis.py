import pandas as pd

# Load the student datasets
math_data = pd.read_csv("data/student-mat.csv", sep=";")
portuguese_data = pd.read_csv("data/student-por.csv", sep=";")

# Display basic information
print("Mathematics Dataset")
print("Shape:", math_data.shape)
print(math_data.head())

print("\nPortuguese Dataset")
print("Shape:", portuguese_data.shape)
print(portuguese_data.head())

# Check missing values
print("\nMissing Values - Mathematics")
print(math_data.isnull().sum())

print("\nMissing Values - Portuguese")
print(portuguese_data.isnull().sum())
# Basic statistics

print("\nMathematics Statistics")
print(math_data.describe())

print("\nPortuguese Statistics")
print(portuguese_data.describe())

# Average final grade

print("\nAverage Final Grade - Mathematics:", math_data["G3"].mean())
print("Average Final Grade - Portuguese:", portuguese_data["G3"].mean())

# Students with passing grades

print("\nStudents with G3 >= 10")
print("Mathematics:", (math_data["G3"] >= 10).sum())
print("Portuguese:", (portuguese_data["G3"] >= 10).sum())
# Day 3 - Data Visualization

import matplotlib.pyplot as plt

# 1. Distribution of final grades
plt.figure(figsize=(8, 5))
plt.hist(math_data["G3"], bins=10)
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.title("Distribution of Final Grades - Mathematics")
plt.show()

# 2. Study time vs final grade
plt.figure(figsize=(8, 5))
plt.scatter(math_data["studytime"], math_data["G3"])
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.title("Study Time vs Final Grade")
plt.show()

# 3. Failures vs final grade
plt.figure(figsize=(8, 5))
plt.scatter(math_data["failures"], math_data["G3"])
plt.xlabel("Number of Failures")
plt.ylabel("Final Grade (G3)")
plt.title("Failures vs Final Grade")
plt.show()

# Day 4 - Feature Analysis

print("\nDay 4 - Feature Analysis")

# Average grade by study time
studytime_analysis = math_data.groupby("studytime")["G3"].mean()

print("\nAverage Final Grade by Study Time:")
print(studytime_analysis)

# Average grade by number of failures
failure_analysis = math_data.groupby("failures")["G3"].mean()

print("\nAverage Final Grade by Number of Failures:")
print(failure_analysis)

# Average grade by school
school_analysis = math_data.groupby("school")["G3"].mean()

print("\nAverage Final Grade by School:")
print(school_analysis)
# Day 5 - Feature Selection

print("\nDay 5 - Feature Selection")

# Select important features for prediction
features = [
    "studytime",
    "failures",
    "absences",
    "Medu",
    "Fedu",
    "G1",
    "G2"
]

target = "G3"

X = math_data[features]
y = math_data[target]

print("\nSelected Features:")
print(features)

print("\nFeature Data:")
print(X.head())

print("\nTarget Data:")
print(y.head())

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# Day 6 - Train Test Split

from sklearn.model_selection import train_test_split

print("\nDay 6 - Train Test Split")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)
# Day 7 - Linear Regression Model

from sklearn.linear_model import LinearRegression

print("\nDay 7 - Linear Regression Model")

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Model training completed successfully!")

# Make predictions
y_pred = model.predict(X_test)

print("\nActual Grades:")
print(y_test.head())

print("\nPredicted Grades:")
print(y_pred[:5])

# Day 8 - Model Evaluation

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\nDay 8 - Model Evaluation")

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# Day 9 - Actual vs Predicted Grades

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Final Grade")
plt.ylabel("Predicted Final Grade")
plt.title("Actual vs Predicted Student Grades")

plt.show()
# Day 10 - Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

print("\nDay 10 - Random Forest Regression")

# Create the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

print("Random Forest model training completed!")

# Make predictions
rf_pred = rf_model.predict(X_test)

print("\nActual Grades:")
print(y_test.head())

print("\nRandom Forest Predictions:")
print(rf_pred[:5])
# Day 11 - Model Comparison

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\nDay 11 - Model Comparison")

# Linear Regression metrics
linear_mae = mean_absolute_error(y_test, y_pred)
linear_mse = mean_squared_error(y_test, y_pred)
linear_r2 = r2_score(y_test, y_pred)

# Random Forest metrics
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\nLinear Regression:")
print("MAE:", linear_mae)
print("MSE:", linear_mse)
print("R2 Score:", linear_r2)

print("\nRandom Forest:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("R2 Score:", rf_r2)

# Day 12 - Model Comparison Visualization

models = ["Linear Regression", "Random Forest"]

mae_values = [linear_mae, rf_mae]
r2_values = [linear_r2, rf_r2]

# MAE comparison
plt.figure(figsize=(8, 5))
plt.bar(models, mae_values)
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error")
plt.title("MAE Comparison of Models")
plt.show()

# R2 comparison
plt.figure(figsize=(8, 5))
plt.bar(models, r2_values)
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.title("R2 Score Comparison of Models")
plt.show()

# Day 13 - Student Grade Prediction

print("\nDay 13 - Student Grade Prediction")

def predict_grade():
    print("\nEnter student details:")

    studytime = float(input("Study time (1-4): "))
    failures = float(input("Number of past failures: "))
    absences = float(input("Number of absences: "))
    Medu = float(input("Mother's education level (0-4): "))
    Fedu = float(input("Father's education level (0-4): "))
    G1 = float(input("First period grade (0-20): "))
    G2 = float(input("Second period grade (0-20): "))

    student_data = [[
        studytime,
        failures,
        absences,
        Medu,
        Fedu,
        G1,
        G2
    ]]

    prediction = model.predict(student_data)

    print("\nPredicted Final Grade (G3):", round(prediction[0], 2))


predict_grade()


# Day 14 - Input Validation

print("\nDay 14 - Input Validation")


def get_valid_input(prompt, minimum, maximum):
    while True:
        try:
            value = float(input(prompt))

            if minimum <= value <= maximum:
                return value

            print(f"Please enter a value between {minimum} and {maximum}.")

        except ValueError:
            print("Please enter a valid number.")


def predict_grade_validated():
    print("\nEnter student details:")

    studytime = get_valid_input("Study time (1-4): ", 1, 4)
    failures = get_valid_input("Number of past failures (0-4): ", 0, 4)
    absences = get_valid_input("Number of absences (0-93): ", 0, 93)
    Medu = get_valid_input("Mother's education level (0-4): ", 0, 4)
    Fedu = get_valid_input("Father's education level (0-4): ", 0, 4)
    G1 = get_valid_input("First period grade (0-20): ", 0, 20)
    G2 = get_valid_input("Second period grade (0-20): ", 0, 20)

    student_data = [[
        studytime,
        failures,
        absences,
        Medu,
        Fedu,
        G1,
        G2
    ]]

    prediction = model.predict(student_data)

    print("\nPredicted Final Grade (G3):", round(prediction[0], 2))


predict_grade_validated()