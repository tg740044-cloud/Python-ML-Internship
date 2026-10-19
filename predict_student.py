# Student Performance Prediction
# Combined Project - Days 23 to 31

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create results folder
os.makedirs("results", exist_ok=True)

# Load dataset
data = pd.read_csv("data/student-mat.csv", sep=";")

features = [
    "studytime", "failures", "absences",
    "Medu", "Fedu", "G1", "G2"
]

X = data[features]
y = data["G3"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(
    n_estimators=100, random_state=42
)
rf_model.fit(X_train, y_train)

# Day 23 - Project Information
print("\nSTUDENT PERFORMANCE PREDICTION")
print("=" * 40)

# Day 24 - Model Evaluation
print("\nMODEL EVALUATION")

models = {
    "Linear Regression": linear_model,
    "Random Forest": rf_model
}

evaluation_results = []

for name, model in models.items():
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    evaluation_results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "R2 Score": r2
    })

evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df.round(3).to_string(index=False))

evaluation_df.to_csv(
    "results/model_evaluation.csv", index=False
)

# Day 25 - Feature Importance
print("\nFEATURE IMPORTANCE")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance.to_string(index=False))

importance.to_csv(
    "results/feature_importance.csv", index=False
)

plt.figure(figsize=(8, 5))
plt.barh(importance["Feature"], importance["Importance"])
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.close()

# Day 26 - Input Validation
def get_valid_input(prompt, minimum, maximum):
    while True:
        try:
            value = float(input(prompt))

            if minimum <= value <= maximum:
                return value

            print(f"Enter a value between {minimum} and {maximum}.")

        except ValueError:
            print("Invalid input. Enter a number.")

# Day 27 - Prediction History Statistics
history_file = "results/predictions.csv"

def show_history():
    if not os.path.exists(history_file):
        print("\nNo prediction history available.")
        return

    with open(history_file, "r", newline="") as file:
        records = list(csv.DictReader(file))

    if not records:
        print("\nNo prediction history available.")
        return

    grades = [
        float(record["PredictedGrade"])
        for record in records
    ]

    print("\nPREDICTION HISTORY")
    print("Total predictions:", len(grades))
    print("Average grade:", round(sum(grades) / len(grades), 2))
    print("Highest grade:", round(max(grades), 2))
    print("Lowest grade:", round(min(grades), 2))

    for record in records:
        print(
            record["Date"],
            "| Grade:", record["PredictedGrade"],
            "| Level:", record["PerformanceLevel"]
        )

# Day 28 - Grade Distribution
plt.figure(figsize=(8, 5))
plt.hist(y, bins=11, edgecolor="black")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.title("Student Grade Distribution")
plt.tight_layout()
plt.savefig("results/grade_distribution.png")
plt.close()

print("\nDATASET STATISTICS")
print("Total students:", len(data))
print("Average first-period grade:", round(data["G1"].mean(), 2))
print("Average second-period grade:", round(data["G2"].mean(), 2))
print("Average final grade:", round(data["G3"].mean(), 2))

# Day 29 - Save Dataset Statistics
statistics = pd.DataFrame([{
    "Total Students": len(data),
    "Average G1": data["G1"].mean(),
    "Average G2": data["G2"].mean(),
    "Average G3": data["G3"].mean()
}])

statistics.to_csv(
    "results/student_statistics.csv", index=False
)

# Day 30 - Save Trained Model
import joblib

joblib.dump(rf_model, "results/student_model.pkl")
print("\nTrained Random Forest model saved.")

# Day 31 - Prediction and Final Testing
def interpret_grade(grade):
    if grade < 10:
        return "Needs Improvement"
    elif grade < 13:
        return "Average"
    elif grade < 16:
        return "Good"
    return "Excellent"

def predict_student():
    print("\nENTER STUDENT DETAILS")

    studytime = get_valid_input("Study time (1-4): ", 1, 4)
    failures = get_valid_input("Past failures (0-4): ", 0, 4)
    absences = get_valid_input("Absences (0-93): ", 0, 93)
    Medu = get_valid_input("Mother's education (0-4): ", 0, 4)
    Fedu = get_valid_input("Father's education (0-4): ", 0, 4)
    G1 = get_valid_input("First grade (0-20): ", 0, 20)
    G2 = get_valid_input("Second grade (0-20): ", 0, 20)

    student_data = [[
        studytime, failures, absences, Medu, Fedu, G1, G2
    ]]

    grade = rf_model.predict(student_data)[0]
    grade = max(0, min(20, grade))
    level = interpret_grade(grade)

    print("\nPredicted Final Grade:", round(grade, 2))
    print("Performance Level:", level)

    file_exists = os.path.exists(history_file)

    with open(history_file, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists or os.path.getsize(history_file) == 0:
            writer.writerow([
                "Date", "StudyTime", "Failures", "Absences",
                "MotherEducation", "FatherEducation",
                "G1", "G2", "PredictedGrade", "PerformanceLevel"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            studytime, failures, absences, Medu, Fedu,
            G1, G2, round(grade, 2), level
        ])

    print("Prediction saved successfully.")

# Main menu
while True:
    print("\n" + "=" * 40)
    print("1. Predict Student Grade")
    print("2. View Prediction History")
    print("3. Exit")

    choice = input("Choose an option: ")

    if choice == "1":
        predict_student()
    elif choice == "2":
        show_history()
    elif choice == "3":
        print("Thank you for using the program!")
        break
    else:
        print("Invalid choice. Select 1, 2, or 3.")