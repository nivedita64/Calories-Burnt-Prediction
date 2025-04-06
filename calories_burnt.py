#importing the dependencies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#data collection & processing
#loading the data from csv file to a Pandas DataFrame


# Getting current script's directory
base_dir = os.path.dirname(__file__)

calories_path = os.path.join(base_dir, "calories.csv")
calories = pd.read_csv(calories_path)
#printing the first five rows of the dataframe
calories.head()

exercise_path = os.path.join(base_dir, "exercise.csv")
exercise_data = pd.read_csv(exercise_path)
exercise_data.head()

#combining the two Dataframes
calories_data = exercise_data.merge(calories)
calories_data.head()

#checking the number of rows and columns
calories_data.shape

#getting some information about the data
calories_data.info()

#checking for missing values
calories_data.isnull().sum()

#DataAnalysis
#get some statistical measures about the data
calories_data.describe()

#Data Visualization
sns.set_theme()
#plotting the gender column in count plot
custom_palette = {'female':'pink','male':'purple'}
sns.countplot(x = 'Gender', data = calories_data, hue ='Gender', palette = custom_palette, legend = False)
plt.show()

#finding the distribution of "Age" column
sns.histplot(data = calories_data, x ='Age', kde = True)
plt.title("Histogram with KDE")
plt.show()

#finding the distribution of "Height" column
sns.histplot(data = calories_data, x ='Height', kde = True)
plt.title("Histogram with KDE")
plt.show()

#finding the distribution of "Weight" column
sns.histplot(data = calories_data, x ='Weight', kde = True)
plt.title("Histogram with KDE")
plt.show()

#finding the distribution of "Duration" column
sns.histplot(data = calories_data, x ='Duration', kde = True)
plt.title("Histogram with KDE")
plt.show()

#finding the distribution of "Heart_Rate" column
sns.histplot(data = calories_data, x ='Heart_Rate', kde = True)
plt.title("Histogram with KDE")
plt.show()

#finding the distribution of "Body_Temp" column
sns.histplot(data = calories_data, x ='Body_Temp', kde = True)
plt.title("Histogram with KDE")
plt.show()

#Finding the Correlation in the dataset

calories_data['Gender'] = calories_data['Gender'].map({'female': 0, 'male': 1})
cols_to_convert = calories_data.columns.drop('User_ID')
calories_data[cols_to_convert] = calories_data[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Dropping rows with NaNs
calories_data.dropna(inplace=True)

# Heatmap with User_ID
correlation = calories_data.corr()
print("\nCorrelation matrix:\n", correlation)

plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',
            annot=True, annot_kws={'size': 8}, cmap="Blues")
plt.title("Correlation Heatmap (with User_ID)")
plt.show()

print(calories_data.head())

# Separate features and target
x = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
y = calories_data['Calories']

print(x)
print(y)

#splitting the data into training data and test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 2)
print(x.shape, x_train.shape,x_test.shape)

#model training
#XGBoost Regressor
#loading the model
model = XGBRegressor()

#training the model with x_train
model.fit(x_train,y_train)

#evaluation
#prediction on test data
test_data_prediction = model.predict(x_test)
print(test_data_prediction)

#Mean Absolute Error
mae = metrics.mean_absolute_error(y_test,test_data_prediction)
print("Mean Absolute Error=", mae)