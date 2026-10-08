# Day 17 - Student Prediction Program

from src.prediction import train_model, predict_grade, interpret_grade

print("Student Performance Prediction")
print("------------------------------")

model = train_model()

studytime = float(input("Study time (1-4): "))
failures = float(input("Number of past failures (0-4): "))
absences = float(input("Number of absences: "))
Medu = float(input("Mother's education level (0-4): "))
Fedu = float(input("Father's education level (0-4): "))
G1 = float(input("First period grade (0-20): "))
G2 = float(input("Second period grade (0-20): "))

student_data = [
    studytime,
    failures,
    absences,
    Medu,
    Fedu,
    G1,
    G2
]

grade = predict_grade(model, student_data)

print("\nPredicted Final Grade:", grade)
print("Performance Level:", interpret_grade(grade))