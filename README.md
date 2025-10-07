# Hypothesis Testing with Python
## One-Way ANOVA and Post Hoc Analysis

## Overview

This project explores the use of one-way ANOVA and Tukey’s HSD post hoc test to analyse how different TV promotion budget categories (Low, Medium, High) influence sales performance.
Using Python’s data analysis and statistical libraries, this activity evaluates whether sales differ significantly across promotional categories, checks model assumptions, and visualises results.
## Dataset

Each record represents an independent marketing promotion, containing:

- **TV Promotion Budget** (Low, Medium, High)
- **Radio Budget** (millions)
- **Social Media Budget** (millions)
- **Influencer Size** (Mega, Macro, Micro, Nano)
- **Sales (millions)**
Dataset source: Dummy Advertising and Sales Data (Kaggle)

## Step 1: Import Libraries
Let's import the necessary libraries.

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
```
## Step 2: Data Exploration
Now we load and display the data.
```
data = pd.read_csv('marketing_sales_data.csv')
data.head()
```
<img width="436" height="170" alt="image" src="https://github.com/user-attachments/assets/1b6f8f77-35b1-4028-9b8c-f8462e259bb0" />

Let's find out the relationship between sales and TV promotional budget by visualisation. Since TV is a categorical variable, we will use a box plot.

```
sns.boxplot(x='TV', y='Sales', data=data)
plt.title('Sales Distribution by TV Promotion Budget')
plt.show()
```
<img width="429" height="287" alt="image" src="https://github.com/user-attachments/assets/60ab7c46-e878-40c0-a4fa-317652686b20" />

There is higher variation in sales between different categories of TV Promotional budgets. Median sales for a low promotional budget are considerably lower compared to medium and high promotional budgets. The median sales for TV promotional budget categories are are little lower than 100, a little more than 200, and around 300 for low, medium and high, respectively. In short, there is higher variation in sales based on the  TV budget category.

## Step 3: Data Cleaning

Let's check if the dataset has missing values.

```
data.isnull().sum()
```
<img width="173" height="121" alt="image" src="https://github.com/user-attachments/assets/742d5062-8c7a-4719-8fee-210561a23cae" />

The dataset has three missing values. We need to remove them.

```
data.dropna(inplace=True)
data.isnull().sum()
```
<img width="180" height="110" alt="image" src="https://github.com/user-attachments/assets/ce8a4022-8206-442d-80c6-1d34f0cd0d2c" />

## Step 4: Model Building (OLS Regression)

Build Model

```
ols_formula = 'Sales ~ C(TV)'
model = ols(formula=ols_formula, data=data).fit()
print(model.summary())
```
<img width="765" height="409" alt="image" src="https://github.com/user-attachments/assets/847f5d12-e464-46cf-a9fd-07eb46db7ea0" />

R-squared is 0.874, 87.4%, which means that the model explains 87.4%  variability in sales. This means TV is an effective predictor of sales. 
The y-intercept is 300.5296, which means without TV promotion, the baseline sales are that amount. Compared to a high TV promotional budget, a low TV budget has 208.8133 fewer sales, and a medium TV promotional budget has 101.5061 fewer sales. The P-value for all coefficients is 0.000 at p = 0.05. Confidence intervals should be reported. For instance, at 5% confidence level for a low TV budget, the true value of the mean sale is between [-215.353, -202.274].

## Step 5: Check Model Assumptions

1-**Normality of Residuals**

```
# First, we need to find residuals
residuals = model.resid
# Create a histogram with the residuals. 
fig, axes = plt.subplots(1,2, figsize = (8,4))

sns.histplot(residuals ,ax = axes[0], kde = True)
axes[0].set_title('Distribution of Residuals')
# Create a QQ plot of the residuals.
sm.qqplot(residuals, line = 's', ax = axes[1])
axes[1].set_title('Q-Q plot of Residuals')
plt.tight_layout()
plt.show()
```
Residuals deviate slightly from normality, but not severely enough to invalidate the model.
2-**Independance**
The independent observation assumption states that each observation in the dataset is independent. As each marketing promotion (row) is independent from one another, the independence assumption is not violated.
3- **Linearity Assumption**
Since we cannot use a scatterplot to check this assumption, we can see from the box plot that Sales increase from low to medium to high promotional budget categories in TV. Thus, we conclude that the linearity assumption is met.

4-**Homoscedasticity (Constant Variance)**

```
sns.scatterplot(x=model.fittedvalues, y=residuals)
plt.axhline(0, color='red')
plt.title('Homoscedasticity Check')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()
```
<img width="474" height="297" alt="image" src="https://github.com/user-attachments/assets/da39d83d-a42a-48e2-b738-e3abab5f38c8" />

We can see the variance is similarly distributed for all three categories, thus indicating the homoscedasticity assumption is met.

## Step 6: One-Way ANOVA Test

With the model fit, run a one-way ANOVA test to determine whether there is a statistically significant difference in Sales among groups.

```
sm.stats.anova_lm(model, typ = 2)
```
<img width="412" height="97" alt="image" src="https://github.com/user-attachments/assets/04a458a1-0099-4390-817d-5e7271252a1a" />

Null Hypothesis: the sales do not differ for high, medium and low categories. Alternative Hypothesis: sales differ for different categories of TV promotional budget.

The F-test statistic is 1971.46 and the p-value is 8.81∗10−256 (i.e., very small). Because the p-value is less than 0.05, you would reject the null hypothesis that there is no difference in Sales based on the TV promotion budget. Since the p-value is extremely small, we can reject the null hypothesis in favour of the alternative. We can say that there is a statistically significant difference in sales in the three categories of TV promotional budget.

## Step 7: Post Hoc Analysis (Tukey’s HSD)
We have significant results from the one-way ANOVA test. WE  apply ANOVA post hoc tests, such as Tukey’s HSD post hoc test. Let us run Tukey’s HSD post hoc test to determine if there is a significant difference between each pair of categories for TV.
```
tukey_oneway = pairwise_tukeyhsd(endog =data['Sales'], groups = data['TV'], alpha = 0.5)
tukey_oneway.summary()
```
<img width="424" height="148" alt="image" src="https://github.com/user-attachments/assets/8b0efae3-5db0-448e-961b-3cdbabf5fe80" />

Based on the Tukey HSD test, we can reject the null hypothesis that there is no statistically significant difference in sales between the three groups. A post hoc test was performed to identify which specific TV groups differ from each other and how many of these differences exist. Unlike the one-way ANOVA, which only indicates that at least one group is different, the post hoc test provides detailed pairwise comparisons. Using the Tukey HSD method also helps control for the increased risk of Type I errors that can occur when performing multiple comparisons.
The results showed that sales differ significantly between all pairs of TV groups.

## Conclusion

- TV promotion budget has a strong, statistically significant impact on sales.
- Higher budgets consistently yield higher sales.
- The regression model explains 87.4% of the variability in sales.
- ANOVA and Tukey’s HSD confirm meaningful differences between all budget categories.

## Recommendation:
Allocate higher TV promotion budgets to achieve better sales outcomes, as evidenced by both regression and ANOVA analyses.
