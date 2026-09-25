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
