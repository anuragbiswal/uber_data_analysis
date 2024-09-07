import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

dataset = pd.read_csv("UberDataset.csv")
dataset.head()
dataset.shape
dataset.info()

dataset = dataset.copy()
dataset['PURPOSE'] = dataset['PURPOSE'].fillna("NOT")
dataset['START_DATE'] = pd.to_datetime(dataset['START_DATE'], errors='coerce')
dataset['END_DATE'] = pd.to_datetime(dataset['END_DATE'], errors='coerce')

dataset['date'] = pd.DatetimeIndex(dataset['START_DATE']).date
dataset['time'] = pd.DatetimeIndex(dataset['START_DATE']).hour

dataset['day-night'] = pd.cut(x=dataset['time'], bins=[0, 10, 15, 19, 24], labels=['Morning', 'Afternoon', 'Evening', 'Night'])

dataset.dropna(inplace=True)
dataset.drop_duplicates(inplace=True)

obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)

unique_vals = {}
for col in object_cols:
    unique_vals[col] = dataset[col].unique().size
print(unique_vals)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.countplot(dataset['CATEGORY'])
plt.xticks(rotation=90)

plt.subplot(1,2,2)
sns.countplot(dataset['PURPOSE'])
plt.xticks(rotation=90)

plt.show()

sns.countplot(dataset['day-night'])
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(15,5))
sns.countplot(data=dataset, x='PURPOSE', hue='CATEGORY')
plt.xticks(rotation=90)
plt.show()

object_cols = ['CATEGORY', 'PURPOSE']
OH_encoder = OneHotEncoder(sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))
OH_cols.index = dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = dataset.drop(object_cols, axis=1)
dataset = pd.concat([df_final, OH_cols], axis=1)


numeric_cols = dataset.select_dtypes(include=['number'])
correlation_matrix = numeric_cols.corr()

plt.figure(figsize=(12,6))
sns.heatmap(correlation_matrix, cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()

dataset['MONTH'] = pd.DatetimeIndex(dataset['START_DATE']).month
month_label = {1: 'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
dataset['MONTH'] = dataset['MONTH'].map(month_label)

mon = dataset['MONTH'].value_counts(sort=False)
df = pd.DataFrame({"MONTHS": mon.values, "VALUE COUNT": dataset.groupby('MONTH', sort=False)['MILES'].max()})

p = sns.lineplot(data=df)
p.set(xlabel="MONTHS", ylabel="VALUE COUNT")
plt.show()

dataset['DAY'] = dataset.START_DATE.dt.weekday
day_label = {0: 'Mon', 1:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat'}
dataset['DAY'] = dataset['DAY'].map(day_label)

day_label = dataset.DAY.value_counts()
sns.barplot(x=day_label.index, y=day_label)
plt.xlabel('DAY')
plt.ylabel('COUNT')
plt.show()

sns.boxplot(dataset['MILES'])
plt.show()

sns.boxplot(dataset[dataset['MILES'] < 100]['MILES'])
plt.show()

sns.histplot(dataset[dataset['MILES'] < 40]['MILES'], kde=True)
plt.xlabel('Miles')
plt.ylabel('Density')
plt.title('Distribution of Uber Ride Miles')
plt.show()