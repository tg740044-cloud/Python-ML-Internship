# Day 19 - Save Prediction Results

import csv
import os
from datetime import datetime

from src.prediction import train_model, predict_grade, interpret_grade


def get_valid_input(prompt, minimum, maximum):
    while True:
        try:
            value = float(input(prompt))

            if minimum <= value <= maximum:
                return value

            print(f"Enter a value between {minimum} and {maximum}.")

        except ValueError:
            print("Invalid input. Please enter a number.")


def save_prediction(student_data, grade, level):
    os.makedirs("results", exist_ok=True)
    file_path = "results/predictions.csv"
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writerow([
                "Date", "StudyTime", "Failures", "Absences",
                "MotherEducation", "FatherEducation",
                "G1", "G2", "PredictedGrade", "PerformanceLevel"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            *student_data,
            grade,
            level
        ])

    print(f"\nPrediction saved to {file_path}")


print("Student Performance Prediction")
print("------------------------------")

model = train_model()

studytime = get_valid_input("Study time (1-4): ", 1, 4)
failures = get_valid_input("Past failures (0-4): ", 0, 4)
absences = get_valid_input("Absences (0-93): ", 0, 93)
Medu = get_valid_input("Mother's education (0-4): ", 0, 4)
Fedu = get_valid_input("Father's education (0-4): ", 0, 4)
G1 = get_valid_input("First period grade (0-20): ", 0, 20)
G2 = get_valid_input("Second period grade (0-20): ", 0, 20)

student_data = [
    studytime, failures, absences, Medu, Fedu, G1, G2
]

grade = predict_grade(model, student_data)
level = interpret_grade(grade)

print("\nPredicted Final Grade:", grade)
print("Performance Level:", level)

save_prediction(student_data, grade, level)