# Feedback_Analysis
Analysing the feedback from students about the Intel Unnati sessions

# Exploratory Data Analysis with Python

In this Python script, we are leveraging popular data science and visualization libraries for exploratory data analysis (EDA). The key libraries utilized are:

- **NumPy:** A powerful library for numerical operations in Python, providing support for large, multi-dimensional arrays and matrices.

- **Pandas:** A data manipulation and analysis library, making it easy to work with structured data through its DataFrame object.

- **Seaborn:** Built on top of Matplotlib, Seaborn is a statistical data visualization library that enhances the aesthetics of plots and provides additional functionality.

- **Matplotlib:** A versatile plotting library for creating static, interactive, and animated visualizations in Python.

The `%matplotlib inline` magic command ensures that Matplotlib plots are displayed directly within the Jupyter Notebook or other compatible environments.

Additionally, the `warnings.filterwarnings('ignore')` line is included to suppress warning messages, enhancing the overall readability of the output.
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```
# Installing Seaborn Using Pip

Seaborn is a Python data visualization library based on Matplotlib. It provides a high-level interface for creating attractive and informative statistical graphics. 

```python
pip install seaborn
```

# Reading CSV Data Using `pd.read_csv()`
The following line of code loads data from a CSV file hosted on GitHub into a pandas DataFrame:

```python
df_class = pd.read_csv("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv")
```
The `pd.read_csv()` code segment plays a pivotal role in this script, making use of the `read_csv` function from the pandas library (`pd`). Below is a breakdown of its significance:

- **pd.read_csv():** This code employs the `read_csv` function, a fundamental tool in the pandas library. It specializes in reading data from CSV files, facilitating the creation of a DataFrame—a tabular data structure widely used in pandas.

- **File Path ("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv"):** The function receives a crucial argument, the URL pointing to the CSV file. This URL corresponds to a raw CSV file hosted on GitHub, specifically within the "Intel-Unnati-sessions" repository.

- **df_class = ...:** The outcome of the `pd.read_csv()` operation is assigned to a variable named `df_class`. This variable now holds a DataFrame, essentially representing the content extracted from the CSV file. The choice of the name `df_class` is arbitrary and can be personalized based on user preference.

# Displaying Initial Rows of the DataFrame

The `df_class.head()` line in Python is a DataFrame method used to display the first few rows of the DataFrame stored in the variable `df_class`. 
 `df_class.head()` allows you to view the initial rows of the DataFrame, providing a snapshot of the data and helping you understand its structure and content. This is a common practice in exploratory data analysis (EDA) to gain insights into the dataset before performing further analysis or visualization.
```python
# Displaying the first few rows of the DataFrame
print(df_class.head())
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/f61c1ad4-6128-48bf-a50c-8c3de39db071)

# Data Presentation Enhancement: Stylized DataFrame Subset

In our endeavor to improve the visual representation of our dataset, the following line of code has been implemented to stylize a sampled subset of our DataFrame, denoted as `df_class`. The primary goal is to enhance the visual appeal and accessibility of a randomly selected portion of our data for better analysis.

```python
df_class.sample(5).style.set_properties(**{'background-color': 'darkgreen', 'color': 'white', 'border-color': 'darkblack'})
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/570b1108-0247-4d80-af74-78c8ad700f64)


# Obtaining DataFrame Information Using `df_class.info()`

The line `df_class.info()` is utilized to retrieve essential information about the DataFrame `df_class`. 

```python
# Displaying DataFrame information
df_class.info()

```
# Removing Columns from DataFrame
The following code snippet aims to refine the structure of the DataFrame `df_class` by eliminating certain columns.
```python
df_class = df_class.drop(['Timestamp', 'Email ID', 'Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience.'], axis=1)
```
# Renaming DataFrame Columns
The following code snippet focuses on enhancing the readability and interpretability of the DataFrame `df_class` by renaming its columns. 

```python
df_class.columns = ["Name", "Branch", "Semester", "Resource Person", "Content Quality", "Effectiveness", "Expertise", "Relevance", "Overall Organization"]
```
Displaying Random Sample of DataFrame Rows
The code snippet `df_class.sample(5)` is designed to showcase a random sample of rows from the DataFrame `df_class`. 

```python
df_class.sample(5)
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/b12233ab-a29b-4c8f-a14f-66ab9c2e98dc)

# Counting Missing Values in DataFrame
The code snippet `df_class.isnull().sum().sum()` is employed to determine the total count of missing values present in the DataFrame `df_class`.
```python
df_class.isnull().sum().sum()
```
#  Determining DataFrame Dimensions
The code snippet `df_class.shape` is utilized to retrieve information about the dimensions of the DataFrame `df_class`. 
```python
df_class.shape
```
# Percentage Analysis of RP-wise Distribution of Data
The code snippet aims to analyze the percentage distribution of data based on the "Resourse Person" column in the DataFrame `df_class`.
```python
round(df_class["Resourse Person"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/3b7e59e7-c671-4b51-aa68-91347f00898e)

# Percentage Analysis of Participant-wise Distribution of Data
The code snippet aims to analyze the percentage distribution of data based on the "Name" column in the DataFrame `df_class`. 
```python
round(df_class["Name"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/89517ac7-4736-4a65-a400-20566355ce3d)

#  Visualizing Faculty-wise Distribution of Data
The following code aims to visually represent the faculty-wise distribution of data using both a countplot and a pie chart.
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Subplot 1: Countplot
ax = plt.subplot(1, 2, 1)
ax = sns.countplot(x='Resourse Person', data=df_class)
plt.title("Faculty-wise distribution of data", fontsize=20, color='Brown', pad=20)

# Subplot 2: Pie Chart
ax = plt.subplot(1, 2, 2)
ax = df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1, 0.1, 0.1, 0.1, 0.1], autopct='%1.2f%%', shadow=True)
ax.set_title(label="Resourse Person", fontsize=20, color='Brown', pad=20)
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/0271eb38-a7e1-40f8-bcc0-7c37ee48f7ba)

#  Visualizing Content Quality Ratings Across Resource Persons
The following code snippet aims to visually represent the distribution of "Content Quality" ratings across different resource persons using a boxplot.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Content Quality Ratings Across Resource Persons
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Content Quality'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/96da88f9-da7a-4bb1-962d-de72d718c7ee)

# Visualizing Effectiveness Ratings Across Resource Persons
The following code snippet aims to visually represent the distribution of "Effectiveness" ratings across different resource persons using a boxplot.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Effectiveness Ratings Across Resource Persons
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Effectiveness'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/0ef64773-b88f-479a-8e51-5edd5b734a75)
# Visualizing Expertise Ratings Across Resource Persons
The following code snippet aims to visually represent the distribution of "Expertise" ratings across different resource persons using a boxplot.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Expertise Ratings Across Resource Persons
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Expertise'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/57c69533-1a91-43e8-b021-0d00bee427a0)

# Visualizing Relevance Ratings Across Resource Persons
The following code snippet aims to visually represent the distribution of "Relevance" ratings across different resource persons using a boxplot.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Relevance Ratings Across Resource Persons
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Relevance'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/554c4f14-cc12-446f-98a3-fd2bc3e87cb8)

# Visualizing Overall Organization Ratings Across Resource Persons
The following code snippet aims to visually represent the distribution of "Overall Organization" ratings across different resource persons using a boxplot. 
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Overall Organization Ratings Across Resource Persons
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Overall Organization'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/24659a6d-387d-4275-bede-97b1b83931c1)
 Visualizing Feedback Distribution Across Branches and Resource Persons
The following code snippet aims to visually represent the distribution of feedback data across different branches for each resource person using a boxplot.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Feedback Distribution Across Branches and Resource Persons
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Branch'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/b8e78aed-0aaa-4876-84fc-24eba346e943)

 Visualizing Content Quality Ratings Across Branches
The following code snippet aims to visually represent the distribution of "Content Quality" ratings across different branches using a boxplot report.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Content Quality Ratings Across Branches
sns.boxplot(y=df_class['Branch'], x=df_class['Content Quality'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/eb487317-f052-4720-8c78-72eae479b161)
