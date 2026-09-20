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