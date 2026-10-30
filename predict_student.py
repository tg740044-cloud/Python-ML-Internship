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

    # Day 32 - Compare Actual and Predicted Grades

def compare_grades():
    actual_grades = y_test.tolist()
    predicted_grades = rf_model.predict(X_test)

    comparison = pd.DataFrame({
        "Actual Grade": actual_grades,
        "Predicted Grade": predicted_grades.round(2)
    })

    comparison["Difference"] = (
        comparison["Actual Grade"] -
        comparison["Predicted Grade"]
    ).round(2)

    print("\nACTUAL VS PREDICTED GRADES")
    print(comparison.head(10).to_string(index=False))

    comparison.to_csv(
        "results/grade_comparison.csv",
        index=False
    )

    print("\nComparison saved to results/grade_comparison.csv")
    # DAY 33 - PASS AND FAIL ANALYSIS

def pass_fail_analysis():
    passed = len(data[data["G3"] >= 10])
    failed = len(data[data["G3"] < 10])

    print("\nPASS AND FAIL ANALYSIS")
    print("----------------------")
    print("Total Students:", len(data))
    print("Passed Students:", passed)
    print("Failed Students:", failed)

    plt.figure(figsize=(6, 5))
    plt.bar(["Passed", "Failed"], [passed, failed])
    plt.title("Student Pass and Fail Analysis")
    plt.ylabel("Number of Students")
    plt.savefig("results/pass_fail_analysis.png")
    plt.close()

    print("Graph saved successfully.")


# DAY 34 - ATTENDANCE ANALYSIS

def attendance_analysis():
    average = data["absences"].mean()
    highest = data["absences"].max()
    lowest = data["absences"].min()

    print("\nATTENDANCE ANALYSIS")
    print("-------------------")
    print("Average Absences:", round(average, 2))
    print("Highest Absences:", highest)
    print("Lowest Absences:", lowest)

    plt.figure(figsize=(8, 5))
    plt.hist(data["absences"], bins=15, edgecolor="black")
    plt.title("Student Absence Distribution")
    plt.xlabel("Number of Absences")
    plt.ylabel("Number of Students")
    plt.savefig("results/attendance_analysis.png")
    plt.close()

    print("Attendance graph saved successfully.")


# DAY 35 - STUDY TIME VS FINAL GRADE

def study_time_analysis():
    average_grades = data.groupby("studytime")["G3"].mean()

    print("\nSTUDY TIME ANALYSIS")
    print("-------------------")
    print(average_grades.round(2))

    plt.figure(figsize=(8, 5))
    plt.plot(
        average_grades.index,
        average_grades.values,
        marker="o"
    )

    plt.title("Study Time vs Final Grade")
    plt.xlabel("Study Time")
    plt.ylabel("Average Final Grade")
    plt.xticks([1, 2, 3, 4])
    plt.grid(True)

    plt.savefig("results/study_time_analysis.png")
    plt.close()

    print("Study time graph saved successfully.")
    # DAY 36 - PARENTAL EDUCATION ANALYSIS

def parental_education_analysis():
    average_grades = data.groupby("Medu")["G3"].mean()

    print("\nPARENTAL EDUCATION ANALYSIS")
    print("---------------------------")
    print("Mother's Education vs Average Student Grade")
    print(average_grades.round(2))

    plt.figure(figsize=(8, 5))
    plt.bar(average_grades.index, average_grades.values)

    plt.title("Mother's Education vs Student Performance")
    plt.xlabel("Mother's Education Level")
    plt.ylabel("Average Final Grade")
    plt.savefig("results/parental_education.png")
    plt.close()

    print("Graph saved successfully.")


# DAY 37 - PAST FAILURES ANALYSIS

def failure_analysis():
    average_grades = data.groupby("failures")["G3"].mean()

    print("\nPAST FAILURES ANALYSIS")
    print("----------------------")
    print("Past Failures vs Average Final Grade")
    print(average_grades.round(2))

    plt.figure(figsize=(8, 5))
    plt.bar(average_grades.index, average_grades.values)

    plt.title("Past Failures vs Student Performance")
    plt.xlabel("Number of Past Failures")
    plt.ylabel("Average Final Grade")
    plt.savefig("results/failure_analysis.png")
    plt.close()

    print("Failure analysis graph saved successfully.")


# DAY 38 - CORRELATION ANALYSIS

def correlation_analysis():
    columns = [
        "studytime", "failures", "absences",
        "Medu", "Fedu", "G1", "G2", "G3"
    ]

    correlation = data[columns].corr()

    print("\nCORRELATION ANALYSIS")
    print("--------------------")
    print(correlation.round(2))

    plt.figure(figsize=(9, 7))

    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)

    plt.colorbar(label="Correlation")
    plt.xticks(range(len(columns)), columns, rotation=45)
    plt.yticks(range(len(columns)), columns)

    plt.title("Student Performance Correlation Matrix")
    plt.tight_layout()

    plt.savefig("results/correlation_analysis.png")
    plt.close()

    print("Correlation graph saved successfully.")
    # DAY 39 - GENDER-WISE PERFORMANCE ANALYSIS

def gender_analysis():
    average_grades = data.groupby("sex")["G3"].mean()

    print("\nGENDER-WISE PERFORMANCE")
    print("-----------------------")
    print("Average Final Grade:")
    print(average_grades.round(2))

    plt.figure(figsize=(7, 5))
    plt.bar(average_grades.index, average_grades.values)

    plt.title("Gender-wise Student Performance")
    plt.xlabel("Gender")
    plt.ylabel("Average Final Grade")

    plt.savefig("results/gender_analysis.png")
    plt.close()

    print("Gender analysis graph saved.")


# DAY 40 - GRADE CATEGORY ANALYSIS

def grade_category_analysis():
    def categorize_grade(grade):
        if grade < 10:
            return "Fail"
        elif grade < 13:
            return "Average"
        elif grade < 16:
            return "Good"
        else:
            return "Excellent"

    categories = data["G3"].apply(categorize_grade)
    counts = categories.value_counts()

    print("\nGRADE CATEGORY ANALYSIS")
    print("-----------------------")
    print(counts)

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)

    plt.title("Student Grade Categories")
    plt.xlabel("Performance Category")
    plt.ylabel("Number of Students")

    plt.savefig("results/grade_categories.png")
    plt.close()

    print("Grade category graph saved.")


# DAY 41 - AUTOMATIC PROJECT SUMMARY REPORT

def generate_project_report():
    total_students = len(data)
    average_grade = data["G3"].mean()
    highest_grade = data["G3"].max()
    lowest_grade = data["G3"].min()

    passed = len(data[data["G3"] >= 10])
    failed = len(data[data["G3"] < 10])

    report = f"""
STUDENT PERFORMANCE PREDICTION
PROJECT SUMMARY REPORT
====================================

1. DATASET INFORMATION

Total Students: {total_students}

2. GRADE STATISTICS

Average Final Grade: {average_grade:.2f}
Highest Final Grade: {highest_grade}
Lowest Final Grade: {lowest_grade}

3. PASS AND FAIL ANALYSIS

Passed Students: {passed}
Failed Students: {failed}

4. STUDY INFORMATION

Average Study Time: {data['studytime'].mean():.2f}
Average Absences: {data['absences'].mean():.2f}

5. MODEL INFORMATION

Model Used: Random Forest Regressor
Target Variable: G3
Features Used: {', '.join(features)}

====================================
END OF REPORT
"""

    with open("results/project_summary_report.txt", "w") as file:
        file.write(report)

    print("\nPROJECT SUMMARY REPORT")
    print(report)
    print("Report saved successfully.")

# Main menu
while True:
    print("\n" + "=" * 40)
    print("1. Predict Student Grade")
    print("2. View Prediction History")
    print("3. Exit")
    print("4. Compare Actual and Predicted Grades")
    print("5. Pass and Fail Analysis")
    print("6. Attendance Analysis")
    print("7. Study Time Analysis")
    print("8. Parental Education Analysis")
    print("9. Past Failures Analysis")
    print("10. Correlation Analysis")
    print("11. Gender-wise Performance Analysis")
    print("12. Grade Category Analysis")
    print("13. Generate Project Summary Report")

    choice = input("Choose an option: ")

    if choice == "1":
        predict_student()
    elif choice == "2":
        show_history()
    elif choice == "3":
        print("Thank you for using the program!")
        break
    elif choice == "4":
      compare_grades()
    elif choice == "5":
            pass_fail_analysis()

    elif choice == "6":
            attendance_analysis()

    elif choice == "7":
            study_time_analysis()
    elif choice == "8":
            parental_education_analysis()
    elif choice == "9":
            failure_analysis()

    elif choice == "10":
            correlation_analysis()
    elif choice == "11":
            gender_analysis()
    elif choice == "12":
            grade_category_analysis()

    elif choice == "13":
            generate_project_report()
    else:
        print("Invalid choice. Select from 1 to 13.")
