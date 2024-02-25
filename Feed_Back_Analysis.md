# Machine Learning Report on Intel Certification Course Participant Segmentation

# 1. Problem Statement

A study of the segmentation of the Intel Certification course participants over satisfaction level.

# 2. Introduction

Feedback analysis is crucial in educational settings for improving instruction quality. Machine learning offers powerful tools to segment feedback into meaningful categories. This study combines exploratory data analysis (EDA) and K-means clustering to achieve participant segmentation.
### Dataset
- The dataset encompasses feedback from students across various academic branches and is targeted at different resource persons.
- There is a representation of 4 unique branches and feedback has been recorded for 4 distinct resource persons.
- The distribution of data across faculty members is diverse, reflecting a wide range of feedback for individual sessions. This variety in responses can be instrumental in understanding the multifaceted aspects of teaching and learning experiences.

# 3. Methodology

The methodology involves EDA to summarize data characteristics and identify patterns. K-means clustering, an unsupervised learning method, segments participants based on feedback similarity. The Elbow method aids in determining the optimal number of clusters.

# 4.Exploratory Data Analysis (EDA):

## Exploratory Data Analysis with Python

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
## Installing Seaborn Using Pip

Seaborn is a Python data visualization library based on Matplotlib. It provides a high-level interface for creating attractive and informative statistical graphics. 

```python
pip install seaborn
```

## Reading CSV Data Using `pd.read_csv()`
The following line of code loads data from a CSV file hosted on GitHub into a pandas DataFrame:

```python
df_class = pd.read_csv("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv")
```
The `pd.read_csv()` code segment plays a pivotal role in this script, making use of the `read_csv` function from the pandas library (`pd`). Below is a breakdown of its significance:

- **pd.read_csv():** This code employs the `read_csv` function, a fundamental tool in the pandas library. It specializes in reading data from CSV files, facilitating the creation of a DataFrame—a tabular data structure widely used in pandas.

- **File Path ("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv"):** The function receives a crucial argument, the URL pointing to the CSV file. This URL corresponds to a raw CSV file hosted on GitHub, specifically within the "Intel-Unnati-sessions" repository.

- **df_class = ...:** The outcome of the `pd.read_csv()` operation is assigned to a variable named `df_class`. This variable now holds a DataFrame, essentially representing the content extracted from the CSV file. The choice of the name `df_class` is arbitrary and can be personalized based on user preference.

## Displaying Initial Rows of the DataFrame

The `df_class.head()` line in Python is a DataFrame method used to display the first few rows of the DataFrame stored in the variable `df_class`. 
 `df_class.head()` allows you to view the initial rows of the DataFrame, providing a snapshot of the data and helping you understand its structure and content. This is a common practice in exploratory data analysis (EDA) to gain insights into the dataset before performing further analysis or visualization.
```python
# Displaying the first few rows of the DataFrame
print(df_class.head())
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/f61c1ad4-6128-48bf-a50c-8c3de39db071)

## Data Presentation Enhancement: Stylized DataFrame Subset

In our endeavor to improve the visual representation of our dataset, the following line of code has been implemented to stylize a sampled subset of our DataFrame, denoted as `df_class`. The primary goal is to enhance the visual appeal and accessibility of a randomly selected portion of our data for better analysis.

```python
df_class.sample(5).style.set_properties(**{'background-color': 'darkgreen', 'color': 'white', 'border-color': 'darkblack'})
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/570b1108-0247-4d80-af74-78c8ad700f64)


## Obtaining DataFrame Information Using `df_class.info()`

The line `df_class.info()` is utilized to retrieve essential information about the DataFrame `df_class`. 

```python
# Displaying DataFrame information
df_class.info()

```
## Removing Columns from DataFrame
The following code snippet aims to refine the structure of the DataFrame `df_class` by eliminating certain columns.
```python
df_class = df_class.drop(['Timestamp', 'Email ID', 'Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience.'], axis=1)
```
## Renaming DataFrame Columns
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

## Counting Missing Values in DataFrame
The code snippet `df_class.isnull().sum().sum()` is employed to determine the total count of missing values present in the DataFrame `df_class`.
```python
df_class.isnull().sum().sum()
```
##  Determining DataFrame Dimensions
The code snippet `df_class.shape` is utilized to retrieve information about the dimensions of the DataFrame `df_class`. 
```python
df_class.shape
```
## Percentage Analysis of RP-wise Distribution of Data
The code snippet aims to analyze the percentage distribution of data based on the "Resourse Person" column in the DataFrame `df_class`.
```python
round(df_class["Resourse Person"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/3b7e59e7-c671-4b51-aa68-91347f00898e)

## Percentage Analysis of Participant-wise Distribution of Data
The code snippet aims to analyze the percentage distribution of data based on the "Name" column in the DataFrame `df_class`. 
```python
round(df_class["Name"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/89517ac7-4736-4a65-a400-20566355ce3d)

##  Visualizing Faculty-wise Distribution of Data
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

##  Visualizing Content Quality Ratings Across Resource Persons
The following code snippet aims to visually represent the distribution of "Content Quality" ratings across different resource persons using a boxplot.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Content Quality Ratings Across Resource Persons
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Content Quality'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/96da88f9-da7a-4bb1-962d-de72d718c7ee)

## Visualizing Effectiveness Ratings Across Resource Persons
The following code snippet aims to visually represent the distribution of "Effectiveness" ratings across different resource persons using a boxplot.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Effectiveness Ratings Across Resource Persons
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Effectiveness'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/0ef64773-b88f-479a-8e51-5edd5b734a75)
## Visualizing Expertise Ratings Across Resource Persons
The following code snippet aims to visually represent the distribution of "Expertise" ratings across different resource persons using a boxplot.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Expertise Ratings Across Resource Persons
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Expertise'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/57c69533-1a91-43e8-b021-0d00bee427a0)

## Visualizing Relevance Ratings Across Resource Persons
The following code snippet aims to visually represent the distribution of "Relevance" ratings across different resource persons using a boxplot.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Relevance Ratings Across Resource Persons
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Relevance'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/554c4f14-cc12-446f-98a3-fd2bc3e87cb8)

## Visualizing Overall Organization Ratings Across Resource Persons
The following code snippet aims to visually represent the distribution of "Overall Organization" ratings across different resource persons using a boxplot. 
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Overall Organization Ratings Across Resource Persons
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Overall Organization'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/24659a6d-387d-4275-bede-97b1b83931c1)
##  Visualizing Feedback Distribution Across Branches and Resource Persons
The following code snippet aims to visually represent the distribution of feedback data across different branches for each resource person using a boxplot.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Feedback Distribution Across Branches and Resource Persons
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Branch'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/b8e78aed-0aaa-4876-84fc-24eba346e943)

## Visualizing Content Quality Ratings Across Branches
The following code snippet aims to visually represent the distribution of "Content Quality" ratings across different branches using a boxplot report.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: Content Quality Ratings Across Branches
sns.boxplot(y=df_class['Branch'], x=df_class['Content Quality'])
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/eb487317-f052-4720-8c78-72eae479b161)

## Creating a Feature Matrix
The following code snippet aims to create a feature matrix `X` by extracting values from specific columns in the DataFrame `df_class`.
```python
input_col = ["Content Quality", "Effeciveness", "Expertise", "Relevance", "Overall Organization"]
X = df_class[input_col].values
```
## Elbow Method for Optimal Number of Clusters

The following code snippet aims to utilize the Elbow Method with the KMeans algorithm to determine the optimal number of clusters for the given dataset.
```python
# Initialize an empty list to store the within-cluster sum of squares
from sklearn.cluster import KMeans
wcss = []

# Try different values of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```
## Elbow Method Plot for Optimal Number of Clusters
The following code snippet aims to visualize the within-cluster sum of squares (WCSS) for different values of k using the Elbow Method. 
```python
# Plot the within-cluster sum of squares for different values of k
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/a4c1bc31-6ca6-41a9-9deb-d79bfdc5f941)

##  Grid Search for Optimal Number of Clusters
The following code snippet aims to perform a grid search using GridSearchCV to find the optimal number of clusters (n_clusters) for the KMeans algorithm.
```python
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto', random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```
##  Displaying Best Parameters and Best Score
The following code snippet aims to display the best parameters and the corresponding best score obtained from the grid search for the KMeans algorithm. 
```python
# Print the best parameters and best score
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/b80a3205-0b63-4261-ba94-0d6c841ab189)
##  K-Means Clustering
The following code snippet aims to perform k-means clustering on the feature matrix `X` using the KMeans algorithm. 
```python
# Perform k-means clustering
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
kmeans.fit(X)
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/ae7a58f4-ca1b-44c8-b8b5-f66e55e40568)
## Retrieving Cluster Labels and Adding to DataFrame
The following code snippet aims to retrieve the cluster labels and centroids from the trained k-means model and add the cluster labels to the DataFrame `df_class`. 
```python
# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
df_class['Cluster'] = labels
df_class.head()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/7bb48793-da19-4133-b96c-f8d8b88717ba)
## Visualization of K-Means Clusters

The following code snippet aims to visualize the clusters formed by the k-means algorithm using a scatter plot. This report provides insights into the purpose of the visualization and its role in understanding the distribution of feedback entries in the feature space.

```python
# Visualize the clusters
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis')
plt.scatter(centroids[:, 1], centroids[:, 2], marker='X', s=200, c='red')
plt.xlabel(input_col[1])
plt.ylabel(input_col[2])
plt.title('K-means Clustering')
plt.show()
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/a23391ac-5b03-4c77-8698-66c8562dd969)

## Cross-Tabulation of Cluster vs Content Quality
The following code snippet aims to create a cross-tabulation table, using the Pandas `crosstab` function, to analyze the relationship between the assigned clusters and the 'Content Quality' ratings provided in the Intel Unnati session feedback.
```python
# Cross-Tabulation of Cluster vs Content Quality
pd.crosstab(columns=df_class['Cluster'], index=df_class['Content Quality'])
```
![image](https://github.com/AiswaryaArun19/Feedback_Analysis/assets/106422393/2a2addf2-0dab-4993-a7c6-945a795c0987)

Our initial exploration of the data (EDA) reveals valuable insights into the feedback provided by participants of the Intel Certification course. By employing various data visualization techniques, we can gain a comprehensive understanding of the participant experience across different instructors, content areas, and branches.

## Balanced Data Distribution and High Content Quality:

The visualizations depict a well-distributed dataset across instructors, with Mrs. Akshara Sasidharan contributing the largest share of feedback. This distribution allows for a thorough analysis of diverse teaching styles and content delivery methods.

Dominating the visualizations are high ratings for content quality across all resource persons. This positive response affirms that the course material generally meets or even surpasses participant expectations, providing a strong foundation for the learning experience.

## Variability in Session Effectiveness and Organization:

While content quality is consistently praised, variations and outliers are present in the ratings for effectiveness, expertise, and overall organization. These fluctuations suggest that specific sessions may require further investigation. Although certain sessions received high marks, others may warrant additional attention to improve participant satisfaction.

Further analysis reveals that the instructors' expertise is highly regarded, with minimal outliers present. This reinforces the exceptional quality of instruction delivered throughout the course.

## Focus on Relevance and Potential for Improvement:

Participants have assigned high relevance scores, highlighting the curriculum's effectiveness in preparing them to apply their acquired knowledge to real-world scenarios. This strong emphasis on practical application underscores the course's value proposition.

While trending positively, the overall organization of the sessions exhibits slightly more variability compared to other metrics. This variability indicates potential for improvement in time management and clarity of instructions, ensuring smoother and more focused learning experiences in future iterations.

## Branch-Wise Content Quality and Identifying Areas for Enhancement:

A granular analysis by branch reveals generally consistent content quality across all academic programs. However, the ECE branch demonstrates slightly more variability in its ratings. This finding suggests a potential area for focused improvements to ensure consistent learning outcomes across diverse academic disciplines.




# 5.Machine Learning Model to study segmentation: K-means clustering


The Elbow method, a visualization technique, is employed to guide the selection of the optimal number of clusters ('k') for K-means clustering, a machine learning algorithm used to segment data into distinct groups. The visualization displays a smooth decrease in the within-cluster sum of squares (WCSS) as 'k' increases.

While the initial trend indicates that a 4-cluster solution might be optimal, a detailed analysis of the feedback necessitates the selection of 3 clusters for a more nuanced exploration of participant segmentation.

## Visualizing Clusters for Tailored Interventions:

By employing the chosen 3-cluster K-means model, we can visualize distinct segments within the participant feedback. This segmentation is visualized using the effectiveness and expertise ratings as key features. This distinct grouping of feedback suggests potential variations in participant satisfaction levels or expectations.

The identification of these distinct segments holds significant value for course providers. This information can be leveraged to tailor resources and interventions to address the specific needs and preferences of different participant groups.

# 6.Results and Conclusion:

By effectively combining EDA and K-means clustering, we have gained valuable insights from the participant feedback on the Intel Certification course. The analysis reveals several key takeaways:

## Strengths:

- The course excels in content quality, instructor expertise, and course relevance, ensuring participants gain valuable knowledge applicable to real-world scenarios.

## Areas for Improvement:

- While content is generally strong, opportunities exist to enhance the effectiveness and overall organization of the sessions, fostering a more streamlined and efficient learning experience.

These insights, informed by the segmentation analysis, provide a crucial roadmap for future course iterations. By incorporating targeted improvements in identified areas and tailoring content delivery and organization strategies to address the needs of diverse participant groups, the course can continue to deliver exceptional learning experiences that empower participants to thrive in their respective fields.


