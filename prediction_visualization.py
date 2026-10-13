# Day 22 - Prediction Visualization

import csv
import os
import matplotlib.pyplot as plt

file_path = "results/predictions.csv"

if not os.path.exists(file_path):
    print("No prediction history found. Run predict_student.py first.")
else:
    with open(file_path, "r", newline="") as file:
        records = list(csv.DictReader(file))

    if not records:
        print("No predictions available to visualize.")
    else:
        grades = [
            float(record["PredictedGrade"])
            for record in records
        ]

        prediction_numbers = list(range(1, len(grades) + 1))

        plt.figure(figsize=(8, 5))
        plt.plot(prediction_numbers, grades, marker="o")
        plt.xlabel("Prediction Number")
        plt.ylabel("Predicted Final Grade")
        plt.title("Student Grade Prediction History")
        plt.xticks(prediction_numbers)
        plt.grid(True)
        plt.tight_layout()
        plt.show()