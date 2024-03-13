# Linear Regression Model to Find Correlation Between Age and Insurance Cost

## Overview
This Jupyter Notebook explores the correlation between age and insurance cost using a linear regression model. The dataset used is named 'insurance.csv,' and the analysis aims to identify trends and make predictions based on the gathered information.

## Dependencies
Make sure you have the following Python libraries installed:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

You can install these dependencies using the following:
> pip install pandas numpy seaborn matplotlib scikit-learn

## Steps

### 1. Importing Libraries
The necessary libraries are imported for data manipulation, visualisation, and machine learning.

> import pandas as pd  
> import numpy as np  
> import seaborn as sns  
> from matplotlib import pyplot as plt  
> from sklearn.linear_model import LinearRegression

### 2. Importing and Processing Data
The dataset ('insurance.csv') is loaded and processed to extract relevant information for analysis.

> df = pd.read_csv('insurance.csv')  
> age_and_cost = df[['age', 'charges']]  
> age_and_cost_ordered = age_and_cost.sort_values(['age'], ascending=True).reset_index(drop=True)

### 3. Initial Data Plotting and Findings
The initial scatter plot of age against insurance cost is created, and preliminary observations are made.

> plt.figure()  
> sns.scatterplot(data=age_and_cost_ordered, x='age', y='charges')  
> plt.show()  
> plt.close()

# Summarise initial findings
> We can see from this graph that there are three clear 'groups' or 'bands' of insurance cost.  
> In order to gain a single, clear line of best fit, we will need to find the average insurance cost for each age group.

### 4. Refining Data Based on Initial Findings and Replotting
The data is grouped by age, and the mean insurance cost for each group is calculated, resulting in a refined scatter plot.

> grouped_age_and_cost = age_and_cost.groupby('age')['charges'].mean().reset_index()

> plt.figure()  
> sns.scatterplot(data=grouped_age_and_cost, x='age', y='charges')  
> plt.show()  
> plt.close()

# Summarise findings
> We can see here that the data is much better suited to finding a single line of best fit.

### 5. Applying Linear Regression Modeling
A linear regression model is applied to the refined data to establish a correlation between age and insurance cost.

> x = grouped_age_and_cost['age'].values.reshape(-1, 1)  
> y = grouped_age_and_cost['charges'].values

> insurance_model = LinearRegression()  
> insurance_model.fit(x, y)  
> y_pred = insurance_model.predict(x)

### 6. Plotting Refined Data with Linear Regression Line
The refined data is plotted along with the linear regression line to visualise the correlation.

> plt.figure()  
> sns.scatterplot(data=grouped_age_and_cost, x='age', y='charges', color='green')  
> plt.plot(x, y_pred, label='Linear Regressor', color='red')

# Customising plot
> plt.title('Comparing Average Insurance Cost Against Age', fontdict={'fontsize': 14})  
> plt.xlabel('Age')  
> plt.ylabel('Average Insurance Cost')  

# Printing plot
> plt.legend()  
> plt.show()  
> plt.close()

# Summarise findings
> From this graph, it's clear to see that there is a correlation between age and insurance cost.  
> The average cost of insurance for 30-year-olds is roughly 10,000, whereas this is doubled for those who are aged 65.  
> It would be interesting to understand what type of insurance this is. I assume it's life insurance, as the likelihood of a claim will go up as age increases.  
> It can be said with a degree of confidence that it's not car insurance.

### 7. Extending Linear Regressor to Make Predictions Beyond Gathered Data
The linear regression model is extended to predict insurance costs for ages beyond the gathered data.

# Set prediction age-points
> unknown_a = [[70]]  
> unknown_b = [[80]]  
> unknown_c = [[90]]  
> unknown_d = [[100]]

# Reshape and refit
> x_pred = np.append(x, 100).reshape(-1, 1)  
> y_pred = insurance_model.predict(x_pred)

# Plot prediction graph
> plt.figure(figsize=(9,5))  
> sns.scatterplot(data=grouped_age_and_cost, x='age', y='charges', color='green')  
> plt.plot(x_pred, y_pred, label='Linear Regressor', color='red')

# Customising plot
> plt.title('Comparing Average Insurance Cost Against Age', fontdict={'fontsize': 14})  
> plt.xlabel('Age')  
> plt.ylabel('Average Insurance Cost')  
> plt.xticks([20,30,40,50,60,70,80,90,100])

# Printing plot
> plt.legend()  
> plt.show()  
> plt.close()

# Format predictions at two decimal places
> pred_70 = format(float(insurance_model.predict(unknown_a)), '.2f')  
> pred_80 = format(float(insurance_model.predict(unknown_b)), '.2f')  
> pred_90 = format(float(insurance_model.predict(unknown_c)), '.2f')  
> pred_100 = format(float(insurance_model.predict(unknown_d)), '.2f')

> Predicted insurance costs by age:  
> print(f"70 = {pred_70}")  
> print(f"80 = {pred_80}")  
> print(f"90 = {pred_90}")  
> print(f"100 = {pred_100}")

## Conclusion
This analysis demonstrates a correlation between age and insurance cost. The linear regression model is used to make predictions beyond the gathered data, providing insights into potential future insurance costs.
