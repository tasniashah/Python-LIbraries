#!/usr/bin/env python
# coding: utf-8

# # Task 1: NumPy Basics

# Write a Python script that creates a NumPy array with random integers and performs basic mathematical operations like addition, subtraction, multiplication, and division on it.

# In[3]:


import numpy as np

array_size = (4, 4)  
min_value = 1
max_value = 17

# Create a random integer NumPy array
random_array = np.random.randint(min_value, max_value , size=array_size)

print("Random Array:")
print(random_array)

# Basic mathematical operations
# Addition
addition_result = random_array + 5  # Adding 5 to each element
print("\nAddition Result:")
print(addition_result)

# Subtraction
subtraction_result = random_array - 3  # Subtracting 3 from each element
print("\nSubtraction Result:")
print(subtraction_result)

# Multiplication
multiplication_result = random_array * 2  # Multiplying each element by 2
print("\nMultiplication Result:")
print(multiplication_result)

# Division
division_result = random_array / 2  # Dividing each element by 2
print("\nDivision Result:")
print(division_result)


# # Task 2: Pandas Data Analysis

# Using Pandas, load a dataset of your choice (you can find datasets online or use your own). Clean the data and perform basic data analysis tasks like calculating mean, median, and mode for specific columns.

# In[4]:


import pandas as pd
import numpy as np
df = pd.read_csv("D:/titanic_data.csv")


# In[5]:


df.head()


# In[6]:


# Fill missing age values with the median age
median_age = df["Age"].median()
df["Age"].fillna(median_age, inplace=True)

# Remove rows with missing embarked information
df.dropna(subset=["Embarked"], inplace=True)

# Remove duplicate rows (if any)
df.drop_duplicates(inplace=True)


# In[9]:


# Calculate the mean age of passengers
mean_age = df["Age"].mean()
print("Mean Age: ",mean_age)
# Calculate the survival rate
survival_rate = df["Survived"].mean() * 100  # Convert to percentage
print("Survival Rate: ",survival_rate)
print("Cleaned Dataset:")
print(df)


# In[12]:


# Calculate the mean, median, and mode for "Age" and "Fare" columns
mean_age = df["Age"].mean()
median_age = df["Age"].median()
mode_age = df["Age"].mode().values[0]  # Mode can have multiple values, so we take the first one

mean_fare = df["Fare"].mean()
median_fare = df["Fare"].median()
mode_fare = df["Fare"].mode().values[0]

print("Basic Data Analysis:")
print("Mean Age:", mean_age)
print("Median Age:", median_age)
print("Mode Age:" ,mode_age)

print(f"Mean Fare:", mean_fare)
print(f"Median Fare:", median_fare)
print("Mode Fare:", mode_fare)


# # Task 3: Data Visualization

# Using Matplotlib, create a bar chart or histogram to visualize the distribution of a dataset of your choice. Label the axes and add a title to the chart.

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt

# Create a bar chart for the age distribution
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)  # Create a subplot for the bar chart

# Calculate the age frequency
age_counts = df["Age"].value_counts().sort_index()

# Plot the bar chart
plt.bar(age_counts.index, age_counts.values, width=1.0, edgecolor="k")
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution (Bar Chart)")

# Create a histogram for the age distribution
plt.subplot(1, 2, 2)  # Create a subplot for the histogram

# Plot the histogram
plt.hist(df["Age"], bins=20, edgecolor="k")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution (Histogram)")

# Adjust the layout and display the plots
plt.tight_layout()
plt.show()


# # Task 4: NumPy Array Manipulation

# Create two NumPy arrays and perform operations to concatenate them vertically and horizontally.

# In[18]:


import numpy as np
array1 = np.array([[1, 2, 3],
                   [4, 5, 6]])
array2 = np.array([[7, 8, 9],
                   [10, 11, 12]])
# Vertical concatenation
vertical_concatenation = np.concatenate((array1, array2), axis=0)
# Horizontal concatenation
horizontal_concatenation = np.concatenate((array1, array2), axis=1)
# Print the original arrays and the concatenated arrays
print("Array 1:")
print(array1)
print("\nArray 2:")
print(array2)
print("\nVertically Concatenated:")
print(vertical_concatenation)
print("\nHorizontally Concatenated:")
print(horizontal_concatenation)


# # Task 5: Pandas Data Filtering

# Given a dataset (you can choose one or use your own), use Pandas to filter and extract rows that meet specific criteria (e.g., filtering data for a specific date range).

# In[21]:


import pandas as pd
older_passengers = df[df["Age"] > 30]
print("Passengers Older than 30:")
print(older_passengers)


# # Task 6: Matplotlib Customization

# Create a line plot or scatter plot using Matplotlib and customize it by adding a legend, different marker styles, and changing the line colors

# In[28]:


# Count the number of passengers in each passenger class (Pclass)
class_counts = df["Pclass"].value_counts().sort_index()
plt.figure(figsize=(8, 6))
plt.bar(class_counts.index, class_counts.values, tick_label=class_counts.index, color='lightblue')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Number of Passengers')
plt.title('Number of Passengers in Each Passenger Class')
plt.grid(axis='y')
plt.show()


# 
# # Task 7: NumPy Statistical Analysis

# Use NumPy to calculate descriptive statistics like variance, standard deviation, and correlation coefficients for a dataset of your choice (you can find datasets online or use your own).

# In[ ]:


# Extract the "Age" and "Fare" columns
age_data = df["Age"].to_numpy()
fare_data = df["Fare"].to_numpy()
age_variance = np.var(age_data)  # Variance of Age
age_stddev = np.std(age_data)    # Standard Deviation of Age

fare_variance = np.var(fare_data)  # Variance of Fare
fare_stddev = np.std(fare_data)    # Standard Deviation of Fare
# Calculate the correlation coefficient between Age and Fare
correlation_coefficient = np.corrcoef(age_data, fare_data)[0, 1]
# Display the results
print("Descriptive Statistics:")
print(f"Variance of Age: {age_variance:.2f}")
print(f"Standard Deviation of Age: {age_stddev:.2f}")
print(f"Variance of Fare: {fare_variance:.2f}")
print(f"Standard Deviation of Fare: {fare_stddev:.2f}")
print(f"Correlation Coefficient between Age and Fare: {correlation_coefficient:.2f}")


# # Task 8: Pandas Data Grouping
# 

# Given a dataset (you can choose one or use your own), use Pandas to group data by a specific column and calculate summary statistics (e.g., mean, median) for each group.

# In[30]:


# Group the data by "Pclass" and calculate mean and median age for each group
grouped_data = df.groupby("Pclass")["Age"].agg([("Mean Age", "mean"), ("Median Age", "median")])
# Display the summary statistics
print("Summary Statistics by Passenger Class (Pclass):")
print(grouped_data)


# # Task 9: Matplotlib Subplots

# Create a Matplotlib figure with multiple subplots to display different aspects of a dataset or related datasets (you can choose the dataset and the type of plots).

# In[31]:


# Create a Matplotlib figure with multiple subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create 1 row and 3 columns of subplots
# Subplot 1: Bar chart of passenger class (Pclass)
class_counts = df["Pclass"].value_counts().sort_index()
axes[0].bar(class_counts.index, class_counts.values, tick_label=class_counts.index, color='skyblue')
axes[0].set_xlabel('Passenger Class (Pclass)')
axes[0].set_ylabel('Number of Passengers')
axes[0].set_title('Passenger Class Distribution')
# Subplot 2: Histogram of passenger ages
axes[1].hist(df["Age"], bins=20, edgecolor='k', color='salmon')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Age Distribution')
# Subplot 3: Scatter plot of age vs. fare
axes[2].scatter(df["Age"], df["Fare"], color='purple', alpha=0.5)
axes[2].set_xlabel('Age')
axes[2].set_ylabel('Fare')
axes[2].set_title('Age vs. Fare')
# Adjust layout
plt.tight_layout()
# Show the figure with subplots
plt.show()


# # Task 10: Pandas Data Visualization

# Choose a dataset of your choice and create various visualizations (e.g., bar charts, pie charts, scatter plots) using Pandas to represent different aspects of the data, such as trends or distributions.
# 
# 

# In[33]:


# Visualization 1: Bar chart for count of passengers by passenger class (Pclass)
plt.figure(figsize=(8, 4))  # Create a new figure
class_counts = df["Pclass"].value_counts().sort_index()
class_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Number of Passengers')
plt.title('Count of Passengers by Passenger Class')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Ensure proper layout
# Visualization 2: Pie chart for distribution of passengers by gender (Sex)
plt.figure(figsize=(6, 6))  # Create another new figure
gender_distribution = df['Sex'].value_counts()
gender_distribution.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
plt.ylabel('')  # Remove the y-axis label
plt.title('Distribution of Passengers by Gender')
plt.tight_layout()
# Visualization 3: Scatter plot for age vs. fare
plt.figure(figsize=(8, 4))  # Create a third new figure
plt.scatter(df['Age'], df['Fare'], color='purple', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs. Fare')
plt.tight_layout()

# Show the plots
plt.show()


# In[ ]:




