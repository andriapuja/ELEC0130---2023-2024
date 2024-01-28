#!/usr/bin/env python
# coding: utf-8

# # Coursework: Climate data analysis

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ## TASK I - Preliminary analysis

# **a. Import the `weather-denmark-resampled.pkl` dataset  provided  in  the  folderand explore  the dataset by answering the following questions.**

# In[4]:


df = pd.read_pickle('weather-denmark-resampled.pkl')
df.info()


# In[5]:


pd.set_option('display.max_rows', 8)
pd.set_option('display.max_columns', 10)
df


# In[6]:


# i.How many cities are there in the dataset?

cities = df.columns.levels[0]
num_cities = len(cities)
print("The number of cities in the dataset is:", num_cities)
print(cities)


# In[7]:


# ii.How many observations and features are there in this dataset?

num_observations, num_features = df.shape
print(f"The dataset has {num_observations} observations and {num_features} features")


# In[8]:


# iii.What are the names of the different features?

feature_names = df.columns
print("Feature names :")
for feature in feature_names:
    print(feature)


# **b. Now that you got confident with the dataset, evaluate if the dataset contains anymissing values? If so, then remove them using the pandas built-in function.**
# 
# 

# In[9]:


# Check for missing values
missing_values = df.isnull().sum()

# Print the number of missing values for each feature
print("Number of missing values for each feature:")
print(missing_values)


# In[10]:


# Remove rows with missing values
df_cleaned_rows = df.dropna()

print("DataFrame after removing missing values:")
df_cleaned_rows.info()


# **c. Extract the general statistical properties summarising the minimum, maximum, median, mean and standard deviation values for all the features in the dataset. Spot any anomalies in these properties and clearly explain why you classify them as anomalies.**

# In[24]:


statistical_summary = df_cleaned_rows.describe()
print(statistical_summary)


# In[23]:


# Calculate mean and standard deviation for each column
mean_values = df_cleaned_rows.mean()
std_deviation_values = df_cleaned_rows.std()

# Define threshold for anomaly detection (adjust as needed)
threshold_std_deviation = 8

# Identify anomalies in each column
anomalies = (df_cleaned_rows > mean_values + threshold_std_deviation * std_deviation_values) | (df_cleaned_rows < mean_values - threshold_std_deviation * std_deviation_values)

# Plot the data
fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(20, 15))
fig.suptitle('Anomaly Detection')

for i, ax in enumerate(axs.flatten()):
    column_name = df_cleaned_rows.columns[i]
    ax.plot(df_cleaned_rows.index, df_cleaned_rows[column_name], label=column_name, color='tab:blue')
    
    # Highlight anomalies
    anomaly_points = df_cleaned_rows.loc[anomalies[column_name], column_name]
    ax.scatter(anomaly_points.index, anomaly_points, color='tab:red', label='Anomalies')

    ax.legend()
    ax.set_title(column_name)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# ## TASK II - OUTLIERS
# 
# The second task is focused on spotting and overcoming outliers. Follow the instructions in the following:

# **d. Store the temperature measurements in May 2006 for the city of Odense. Then produce a simple plot of the temperature versus time.**
# 
# *HINT: In this dataset, the cities are vertically stacked. Therefore, we have a multi column dataset, which basically works as a nested dictionary.*
# 

# In[13]:


odense_may_2006 = df_cleaned_rows['Odense']['2006-05']
print(odense_may_2006)


# In[15]:


# Create a box plot
sn.boxplot(x=odense_may_2006["Temp"])

# Show the plot
plt.show()


# In[16]:


plt.figure(figsize=(15, 6))
odense_may_2006["Temp"].plot(label="Odense", marker="o", linestyle="-", color="tab:blue")
plt.title("Temperature Measurements in Odense - May 2006")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()


# **e. Find the outliers in this set of measurements (if any) and replace them using linear interpolation.**

# In[18]:


# Find outliers using the z-score method
z_scores = (odense_may_2006["Temp"] - odense_may_2006["Temp"].mean()) / odense_may_2006["Temp"].std()
outliers = odense_may_2006["Temp"][z_scores.abs() > 3]

# Replace outliers with NaN
odense_may_2006["Temp"][outliers.index] = pd.NA

# Interpolate NaN values using linear interpolation
odense_may_2006["Temp"] = odense_may_2006["Temp"].interpolate(method='linear')


# In[19]:


# Plot the modified temperature versus time
plt.figure(figsize=(15, 6))
odense_may_2006["Temp"].plot(label='Odense', marker='o', linestyle='-', color='tab:blue')
plt.title('Temperature Measurements in Odense (Interpolated) - May 2006')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()


# ## TASK III.1 - CORRELATION
# 
# In this last task, you will be seeking correlation between features of the data and inferring hidden patterns.  For  this  task,  you  will  be  working  with  a  smaller  dataset.  Follow  the  instructions  in  the following:

# **f. We  now  take  a new  dataset (`df_perth.pkl`),  which  collects  climate  data  of  a  city  in Australia. Here we have just one year of measurements, but more features.**

# In[27]:


df_perth = pd.read_pickle("df_perth.pkl")
df_perth


# In[28]:


missing_values_perth = df_perth.isnull().sum()
print(missing_values_perth)


# In[29]:


df_perth.info()


# **g. Find any significant correlations between features.**
# 
# *HINT: you might find useful looking fortrends and recurrent patterns within the data*

# In[30]:


correlation_matrix = df_perth.corr()

plt.figure(figsize=(7, 5))
sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features in Perth Climate Data')
plt.show()


# **h. We now focus on the correlation between precipitation and cloud cover. We want to infer the probability of having moderate to heavy rain (> 1 mm/h) as a function of the cloud cover index.**
# 
# *HINT: you might find useful to create a new column where you have 0 if precipitation < 1 mm/h and 1 otherwise*

# In[31]:


df_perth['precipitation_indicator'] = (df_perth['precipitation'] > 1).astype(int)
df_perth.head()


# In[37]:


correlation_cloud_cover_precipitation = df_perth['cloud cover'].corr(df_perth['precipitation_indicator'])
print(correlation_cloud_cover_precipitation)


# In[39]:


plt.figure(figsize=(5, 4))
sn.scatterplot(x='cloud cover', y='precipitation_indicator', data=df_perth)
plt.title('Cloud Cover vs. Precipitation Indicator')
plt.xlabel('Cloud Cover Index')
plt.ylabel('Precipitation Indicator (1 if > 1 mm/h, 0 otherwise)')
plt.show()

print(f"Correlation between cloud cover and precipitation indicator: {correlation_cloud_cover_precipitation:.2f}")


# ## TASK III.2 - INFERENCE

# **i. Let’s now assume that we want to predict the photovoltaic production (PV production) using  multiple  linear  regression.  Explain  which  features  are  statistically  significant  in modelling the target variable.**

# In[51]:


import statsmodels.api as sm

X = df_perth[["temp", "pressure", "relative humidity", "wind speed", "cloud cover", "precipitation", "diffuse radiation, tilt", "solar azimuth"]]  # Features
y = df_perth['PV production']  # Target variable

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())


# In[56]:


# to avoid multicollinearity, some features removed from the model (relative humidity, cloud cover, diffuse radiation, tilt)




# In[55]:


X = df_perth[["temp", "pressure", "wind speed", "precipitation", "solar azimuth"]]  # Features
y = df_perth['PV production']  # Target variable

# Add a constant term to the features matrix
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print summary statistics
print(model.summary())


# **j. Create a multivariate model using the predictors chosen in the previous question.**

# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# independent variables
X = df_perth[["temp", "pressure", "relative humidity", "wind speed", "cloud cover", "precipitation", "diffuse radiation, tilt", "solar azimuth"]]

# dependent variable
y = df_perth["PV production"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, y_pred)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, y_pred)) 

# Display coefficients and intercept
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)

