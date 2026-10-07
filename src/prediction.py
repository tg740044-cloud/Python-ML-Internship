# Day 16 - Student Prediction Module

import pandas as pd
from sklearn.linear_model import LinearRegression


def train_model():
    data = pd.read_csv("data/student-mat.csv", sep=";")

    features = [
        "studytime",
        "failures",
        "absences",
        "Medu",
        "Fedu",
        "G1",
        "G2"
    ]

    X = data[features]
    y = data["G3"]

    model = LinearRegression()
    model.fit(X, y)

    return model


def predict_grade(model, student_data):
    prediction = model.predict([student_data])
    return round(prediction[0], 2)


def interpret_grade(grade):
    if grade < 10:
        return "Needs Improvement"
    elif grade < 13:
        return "Average"
    elif grade < 16:
        return "Good"
    else:
        return "Excellent"
