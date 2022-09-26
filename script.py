# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import codecademylib3

print("In this project, I analyze simulated learner data. This simulated data was supplied by Codecademy. Such data is similar to actual learner data that curriculum teams would be interested in drawing conclusions from.")

# Read in the data
codecademy = pd.read_csv('codecademy.csv')

# Print the first five rows
print(codecademy.head())
print('\n')

# Create a scatter plot of score vs completed
plt.scatter(codecademy.completed, codecademy.score)

# Show then clear plot
plt.show()
plt.clf()
print("There appears to be a linear relationship between the x-variable (content items completed) and the y-variable (quiz score) based on the scatter plot.")
print('\n')

# Fit a linear regression to predict score based on prior lessons completed
model= sm.OLS.from_formula('score ~ completed', codecademy).fit()
print(model.params)

# Intercept interpretation:
# Slope interpretation:
print("The intercept (Intercept) is 13.21. This intercept suggests that if a user completed 0 content items, they would receive a score of 13.21/100. The slope is 1.31. This slope (completed) indicates that for every 1 content item completed, the user's score can be expected to rise by 1.31 points.")
print('\n')

# Plot the scatter plot with the line on top
plt.scatter(codecademy.completed, codecademy.score)
plt.plot(codecademy.completed, model.params[0] + model.params[1] * codecademy.completed)

# Show then clear plot
plt.show()
plt.clf()

# Predict score for learner who has completed 20 prior lessons
score20= model.params[1] * 20 + model.params[0]
print(score20)
print("A user who previously completed 20 content items is expected to score 39.35 points on the quiz based on our linear regression model.")
print('\n')

# Calculate fitted values
fitted_values= model.predict(codecademy)

# Calculate residuals
residuals= codecademy.score - fitted_values

# Check normality assumption
print("We plot a histogram of the residuals to check the normality assumption for linear regression.")
plt.hist(residuals)

# Show then clear the plot
plt.show()
plt.clf()
print("The residuals histogram appears to be normally distributed")
print('\n')

# Check homoscedasticity assumption
print("We check the homoscedasticity assumption for linear regression by scatterplotting the residuals by the fitted values.")
plt.scatter(fitted_values, residuals)

# Show then clear the plot
plt.show()
plt.clf()
print("The scatter plot appears to meet the homoscedasticity assumption, as the points are randomly scattered and form no patterns.")
print('\n')

print('We now seek to answer the question: Do learners who take lesson A or B perform better on the quiz?')
print('\n')

# Create a boxplot of score vs lesson
sns.boxplot(
  data=codecademy,
  x='lesson',
  y='score'
)

# Show then clear plot
plt.show()
plt.clf()
print("Based on the boxplot, it is clear that students who took Lesson A immediately before the quiz outperform those who took Lesson B.")
print('\n')

print("We now create and fit a linear regression model that predicts score using lesson as the predictor.")
print('\n')

# Fit a linear regression to predict score based on which lesson they took
model= sm.OLS.from_formula('score ~ lesson', codecademy).fit()
print(model.params)
print("The intercept (Intercept) is 59.22. This indicates that learners who took Lesson A received a mean score of 59.22 on the quiz. The slope (lesson[T.Lesson B]) is -11.64. This indicates that learners who took Lesson B received a mean score 11.64 points LESS than 59.22 (the mean score for Lesson A takers).")
print('\n')

# Calculate and print the group means and mean difference (for comparison)
lessonA_mean= round(model.params[0], 2)
lessonB_mean= round(model.params[1] + model.params[0], 2)
lessonAB_diff= lessonA_mean - lessonB_mean
print("The mean quiz score for learners who took lesson A is {}.".format(lessonA_mean))
print("The mean quiz score for learners who took lesson B is {}.".format(lessonB_mean))
print("The mean difference is {}.".format(lessonAB_diff))

# Use `sns.lmplot()` to plot `score` vs. `completed` colored by `lesson`
print("Lastly, we plot score by number of learning items completed, colored by lesson for even more insight into the data.")
sns.lmplot(
  x='completed',
  y='score',
  hue='lesson',
  data=codecademy
)

plt.show()
plt.clf()
