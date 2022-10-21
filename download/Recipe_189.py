### DataSource Card 
## Import Data 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

## Fetch Train Data 
df_train = pd.read_csv('train.csv') 

## Fetch Test Data 
df_test = pd.read_csv('test.csv') 

### DataCleanse Card 

from sklearn.model_selection import train_test_split

test_split = 0.19999999999999996

## Column Types 
from AIStudioModules.dataSummary.columnType import identifyColumnType
column_type_map = identifyColumnType(df_train)
column_type_list = list(column_type_map.values())
column_type_freq = {i:column_type_list.count(i) for i in set(column_type_map.values())}
plt.figure(figsize=(10,5))
plt.title('Column Types')
plt.pie(list(column_type_freq.values()), labels = list(column_type_freq.keys()))
plt.show()

## Column Distribution 
from AIStudioModules.dataSummary.columnSummary import columnValuesDistribution
columnValuesDistribution(<Enter a pandas.Series>)

## Drop Columns
drop_816 = ['Elevador'] 
df_train = df_train.drop(drop_816, axis=1)
df_test = df_test.drop(drop_816, axis=1)

## Drop Columns
drop_818 = ['Número de frentes'] 
df_train = df_train.drop(drop_818, axis=1)
df_test = df_test.drop(drop_818, axis=1)

## Drop Columns Based on their correlations 
corr_drop_822 = ['Latitud (Decimal)', 'Longitud (Decimal)'] 
df_train = df_train.drop(corr_drop_822, axis=1)
df_test = df_test.drop(corr_drop_822, axis=1)

## Drop Columns
drop_823 = ['Fecha entrega del Informe', 'Piso', 'Provincia', 'Distrito', 'Posición'] 
df_train = df_train.drop(drop_823, axis=1)
df_test = df_test.drop(drop_823, axis=1)

## Correlation HeatMap
from AIStudioModules.correlation import plot_correlation_matrix
corr_data = plot_correlation_matrix(df_train)
#plt.figure(figsize=(12,6))
sns.heatmap(pd.DataFrame(corr_data, index=list(corr_data.keys())))
plt.show()

## Select Target and split data
target = 'Valor comercial (USD)'
isnull_train_target = df_train[target].isnull().sum(axis=0)
isnull_test_target = df_test[target].isnull().sum(axis=0)
if isnull_train_target:
	raise Exception('Target column in  the train set, is empty')
if isnull_test_target:
	raise Exception('Target column in  the test set, is empty')

x_train = df_train.drop(target, axis=1)
y_train = df_train[target]
x_test = df_test.drop(target, axis=1)
y_test = df_test[target]

## Imputation if missing values still exist

dataset = pd.concat(objs=[x_train, x_test], axis=0)
for col in dataset.columns[dataset.isnull().any()]:
	dataset[col] = dataset[col].fillna(dataset[col].value_counts().index[0])
x_train, x_test = train_test_split(dataset, test_size=test_split,shuffle=False)

## Dummy
from AIStudioModules.dataSummary.columnType import identifyColumnType
dataset = pd.concat(objs=[x_train, x_test], axis=0)
dataset_column_type_map=identifyColumnType(dataset)
categorical_columns=[col_name for col_name, col_type in dataset_column_type_map.items() if col_type=='categorical']
dataset = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True)
X_train, X_test = train_test_split(dataset, test_size=test_split, shuffle=False)

## Train and Test Data with AutoML

# https://auto.gluon.ai/stable/install.html 
import autogluon as ag
from autogluon import TabularPrediction as task
from sklearn.model_selection import train_test_split

x_train_auto = task.Dataset(X_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
X_train[target] = y_train
X_val[target] = y_val
output_dir = <Enter path of the model to be saved>
predictor = task.fit(train_data=X_train, tuning_data=X_val, label=target, problem_type ='regression', output_directory=output_dir, excluded_model_types=['GBM', 'custom'], auto_stack=True)

X_test[target] = y_test
leader_board_data = predictor.leaderboard(X_test, silent=True)
leader_board_data.head()

best_model_index = (predictor.get_model_names().index(leader_board_data.iloc[0,0]))
best_model = predictor.get_model_names()[best_model_index]

y_pred = predictor.predict(X_test, model=best_model)

## Evaluate Model

import sklearn.metrics as metrics
import math
r2_score=metrics.r2_score(y_test, y_pred)
mae=metrics.mean_absolute_error(y_test, y_pred)
mse=metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print(f'Mean Absolute Error = {mae}')
print(f'Mean Squared Error = {mse}')
print(f'Root Mean Squared Error = {rmse}')

## Save Expainer

import lime
import lime.lime_tabular
import dill
categorical_features = np.argwhere(np.array([len(set(X_train.values[:,x])) for x in range(X_train.values.shape[1])]) <= 10).flatten()
explainer = lime.lime_tabular.LimeTabularExplainer(
	X_train.values,
	feature_names=X_train.columns.values.tolist(),
	class_names=[target],
	categorical_features=categorical_features,
	verbose=False,  mode='regression'
)
with open('<path to save the explainer>', 'wb') as f:
	dill.dump(explainer, f)

# Explainer Test

def predict_fn_reg(x):
	df = pd.DataFrame(x, columns=X_train.columns)
	return predictor.predict(df, model=best_model).astype(float)
exp = explainer.explain_instance(X_test.values[0].astype(float), predict_fn_reg)
exp.show_in_notebook(show_table=True)
