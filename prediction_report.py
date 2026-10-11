# Day 20 - Prediction History Report

import csv
import os

file_path = "results/predictions.csv"

if not os.path.exists(file_path):
    print("No prediction history found. Run predict_student.py first.")

else:
    with open(file_path, "r", newline="") as file:
        reader = csv.DictReader(file)
        records = list(reader)

    if not records:
        print("No predictions recorded yet.")

    else:
        grades = [
            float(record["PredictedGrade"])
            for record in records
        ]

        print("\nStudent Prediction History")
        print("--------------------------")
        print("Total predictions:", len(records))
        print("Average predicted grade:", round(sum(grades) / len(grades), 2))
        print("Highest predicted grade:", round(max(grades), 2))
        print("Lowest predicted grade:", round(min(grades), 2))

        print("\nPrevious Predictions:")

        for record in records:
            print(
                record["Date"],
                "| Grade:", record["PredictedGrade"],
                "| Level:", record["PerformanceLevel"]
            )