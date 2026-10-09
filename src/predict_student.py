# Day 18 - Input Validation

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

print("\nPredicted Final Grade:", grade)
print("Performance Level:", interpret_grade(grade))