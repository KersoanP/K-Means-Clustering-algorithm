# Implementation of K-Means Clustering Algorithm
## Aim
To write a python program to implement K-Means Clustering Algorithm.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation

## Algorithm:

### Step1
<br>Import pandas.



### Step2
<br>Read the file using read_csv.



### Step3
<br>Plot the points using sns.scatterplot.



### Step4
<br>Display the number of rows.



### Step5
<br>Predict the class using .predict and print.



## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

x1 = pd.read_csv('clustering.csv')
print(x1.head(2))
x2 = x1.loc[:, ['ApplicantIncome','LoanAmount']]
print(x2.head(2))

x = x2.values
sns.scatterplot(x[:,0], x[:,1])
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()

kmean =KMeans(n_clusters=4)
kmean.fit(x)

print('Clusters Centers:', kmean.cluster_centers_)
print('Labels:', kmean.labels_)

predicted_class = kmean.predict([[9000,120]])
print('The cluster group for Applicant Income 10000 and Loanamount predicted_class')

```
## Output:

### Insert your output

<br>![image](https://user-images.githubusercontent.com/94525886/155125599-de2eb400-7427-49d2-99c1-a51a980fdd01.png)


## Result
Thus the K-means clustering algorithm is implemented and predicted the cluster class using python program.
