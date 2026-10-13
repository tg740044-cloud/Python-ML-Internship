# Day 21 - Formatted Prediction History

import csv
import os

file_path = "results/predictions.csv"

if not os.path.exists(file_path):
    print("No prediction history found.")
    print("Run predict_student.py first.")
else:
    with open(file_path, "r", newline="") as file:
        records = list(csv.DictReader(file))

    if not records:
        print("No predictions available yet.")
    else:
        print("\nSTUDENT PERFORMANCE PREDICTION HISTORY")
        print("=" * 75)
        print(f"{'Date':<20} {'Grade':>10} {'Level':<20}")
        print("-" * 75)

        for record in records:
            print(
                f"{record['Date']:<20} "
                f"{float(record['PredictedGrade']):>10.2f} "
                f"{record['PerformanceLevel']:<20}"
            )

        grades = [
            float(record["PredictedGrade"])
            for record in records
        ]

        print("-" * 75)
        print("Total predictions:", len(grades))
        print("Average predicted grade:", round(sum(grades) / len(grades), 2))