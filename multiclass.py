
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from os import path

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


data = pd.read_csv('UNSW_NB15.csv')

data.head(n=5)

data.info()

data[data['service']=='-']

data['service'].replace('-',np.nan,inplace=True)

data.isnull().sum()

data.shape

data.dropna(inplace=True)

data.shape

data['attack_cat'].value_counts()

data['state'].value_counts()

data


features = pd.read_csv('UNSW_NB15_features.csv')

features.head()

features['Type '] = features['Type '].str.lower()

# selecting column names of all data types
nominal_names = features['Name'][features['Type ']=='nominal']
integer_names = features['Name'][features['Type ']=='integer']
binary_names = features['Name'][features['Type ']=='binary']
float_names = features['Name'][features['Type ']=='float']

# selecting common column names from dataset and feature dataset
cols = data.columns
nominal_names = cols.intersection(nominal_names)
integer_names = cols.intersection(integer_names)
binary_names = cols.intersection(binary_names)
float_names = cols.intersection(float_names)

# Converting integer columns to numeric
for c in integer_names:
  pd.to_numeric(data[c])

# Converting binary columns to numeric
for c in binary_names:
  pd.to_numeric(data[c])

# Converting float columns to numeric
for c in float_names:
  pd.to_numeric(data[c])

data.info()

data

"""# **Data Visualization**


"""## **Multi-class Classification**"""

plt.figure(figsize=(8,8))
plt.pie(data.attack_cat.value_counts(),labels=data.attack_cat.unique(),autopct='%0.2f%%')
plt.title('Pie chart distribution of multi-class labels')
plt.legend(loc='best')
plt.savefig('plots/Pie_chart_multi.png')
plt.show()

"""# **One hot encoding**"""

num_col = data.select_dtypes(include='number').columns

# selecting categorical data attributes
cat_col = data.columns.difference(num_col)
cat_col = cat_col[1:]
cat_col

# creating a dataframe with only categorical attributes
data_cat = data[cat_col].copy()
data_cat.head()

# one-hot-encoding categorical attributes using pandas.get_dummies() function
data_cat = pd.get_dummies(data_cat,columns=cat_col)

data_cat.head()

data.shape

data = pd.concat([data, data_cat],axis=1)

data.shape

data.drop(columns=cat_col,inplace=True)

data.shape

"""# **Data Normalization**

"""

# selecting numeric attributes columns from data
num_col = list(data.select_dtypes(include='number').columns)
num_col.remove('id')
num_col.remove('label')
print(num_col)

# using minmax scaler for normalizing data
minmax_scale = MinMaxScaler(feature_range=(0, 1))
def normalization(df,col):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = minmax_scale.fit_transform(arr.reshape(len(arr),1))
  return df

# data before normalization
data.head()

# calling normalization() function
data = normalization(data.copy(),num_col)

# data after normalization
data.head()

"""# **Label Encoding**

## **Binary Labels**
"""

# changing attack labels into two categories 'normal' and 'abnormal'
bin_label = pd.DataFrame(data.label.map(lambda x:'normal' if x==0 else 'abnormal'))

# creating a dataframe with binary labels (normal,abnormal)
bin_data = data.copy()
bin_data['label'] = bin_label

# label encoding (0,1) binary labels
le1 = preprocessing.LabelEncoder()
enc_label = bin_label.apply(le1.fit_transform)
bin_data['label'] = enc_label

le1.classes_

np.save("le1_classes.npy",le1.classes_,allow_pickle=True)

"""## **Multi-class Labels**"""

# one-hot-encoding attack label
multi_data = data.copy()
multi_label = pd.DataFrame(multi_data.attack_cat)

multi_data = pd.get_dummies(multi_data,columns=['attack_cat'])

# label encoding (0,1,2,3,4,5,6,7,8) multi-class labels
le2 = preprocessing.LabelEncoder()
enc_label = multi_label.apply(le2.fit_transform)
multi_data['label'] = enc_label

le2.classes_

np.save("le2_classes.npy",le2.classes_,allow_pickle=True)

"""# **Correlation between features of dataset**"""

num_col.append('label')



"""## **Correlation Matrix for Multi-class Labels**"""

num_col = list(multi_data.select_dtypes(include='number').columns)

# Correlation Matrix for Multi-class Labels
plt.figure(figsize=(20,8))
corr_multi = multi_data[num_col].corr()
sns.heatmap(corr_multi,vmax=1.0,annot=False)
plt.title('Correlation Matrix for Multi Labels',fontsize=16)
plt.savefig('plots/correlation_matrix_multi.png')
plt.show()

"""# **Feature Selection**

## **Binary Labels**
"""







"""### **Saving Prepared Dataset to Disk**"""

bin_data.to_csv('datasets/bin_data.csv')

"""## **Multi-class Labels**"""

# finding the attributes which have more than 0.3 correlation with encoded attack label attribute
corr_ymulti = abs(corr_multi['label'])
highest_corr_multi = corr_ymulti[corr_ymulti >0.3]
highest_corr_multi.sort_values(ascending=True)

# selecting attributes found by using pearson correlation coefficient
multi_cols = highest_corr_multi.index
multi_cols

# Multi-class labelled Dataset
multi_data = multi_data[multi_cols].copy()

"""### **Saving Prepared Dataset to Disk**"""

multi_data.to_csv('./datasets/multi_data.csv')



"""# **MULTI-CLASS CLASSIFICATION**

## **Data Splitting**
"""

X = multi_data.drop(columns=['label'],axis=1)
Y = multi_data['label']

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.30, random_state=100)
# Assuming X_test is already defined
np.savetxt('X_test.txt', X_test)

logr_multi = LogisticRegression(random_state=123, max_iter=5000, solver='newton-cg', multi_class='multinomial')
logr_multi1 = LogisticRegression(random_state=123, max_iter=5000,solver='newton-cg',multi_class='multinomial')
logr_multi.fit(X_train,y_train)

y_pred = logr_multi.predict(X_test)
print("****************** **Logistic Regression***************************")
print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - " , metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - " , metrics.explained_variance_score(y_test, y_pred)*100)
print("Accuracy - ",accuracy_score(y_test,y_pred)*100)

print(classification_report(y_test, y_pred,target_names=le2.classes_))

"""### **Real and Predicted Data**"""

logr_multi_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
logr_multi_df.to_csv('./predictions/logr_real_pred_multi.csv')
logr_multi_df

"""### **Plot between Real and Predicted Data**"""

plt.figure(figsize=(20,8))
plt.plot(y_pred[:200], label="prediction", linewidth=2.0,color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0,color='lightcoral')
plt.legend(loc="best")
plt.title("Logistic Regression Multi-class Classification")
plt.savefig('plots/logr_real_pred_multi.png')
plt.show()

"""### **Saving Trained Model to Disk**"""

pkl_filename = "./models/logistic_regressor_multi.pkl"
if (not path.isfile(pkl_filename)):
  # saving the trained model to disk
  with open(pkl_filename, 'wb') as file:
    pickle.dump(logr_multi, file)
  print("Saved model to disk")
else:
  print("Model already saved")

"""## **Support Vector Machine**"""

lsvm_multi = SVC()
lsvm_multi1 = SVC(kernel='linear',gamma='auto')
lsvm_multi.fit(X_train,y_train)

y_pred = lsvm_multi.predict(X_test)
print("********************Support Vector Machine***************************")
print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - " , metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - " , metrics.explained_variance_score(y_test, y_pred)*100)
print("Accuracy - ",accuracy_score(y_test,y_pred)*100)

print(classification_report(y_test, y_pred,target_names=le2.classes_))

"""### **Real and Predicted Data**"""

lsvm_multi_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
lsvm_multi_df.to_csv('./predictions/lsvm_real_pred_multi.csv')
lsvm_multi_df

"""### **Plot between Real and Predicted Data**"""

plt.figure(figsize=(20,8))
plt.plot(y_pred[:200], label="prediction", linewidth=2.0,color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0,color='lightcoral')
plt.legend(loc="best")
plt.title("Linear SVM Multi-class Classification")
plt.savefig('plots/lsvm_real_pred_multi.png')
plt.show()

"""### **Saving Trained Model to Disk**"""


pkl_filename = "./models/lsvm_multi.pkl"
if (not path.isfile(pkl_filename)):
  # saving the trained model to disk
  with open(pkl_filename, 'wb') as file:
    pickle.dump(lsvm_multi, file)
  print("Saved model to disk")
else:
  print("Model already saved")

"""## **K Nearest Neighbor Classifier**"""

knn_multi = KNeighborsClassifier()
knn_multi1 = KNeighborsClassifier(n_neighbors=5)
knn_multi.fit(X_train,y_train)

y_pred = knn_multi.predict(X_test)
print("********************K Nearest Neighbor Classifier***************************")
print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - " , metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - " , metrics.explained_variance_score(y_test, y_pred)*100)
print("Accuracy - ",accuracy_score(y_test,y_pred)*100)

print(classification_report(y_test, y_pred,target_names=le2.classes_))

"""### **Real and Predicted Data**"""

knn_multi_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
knn_multi_df.to_csv('./predictions/knn_real_pred_multi.csv')
knn_multi_df

"""### **Plot between Real and Predicted Data**"""

plt.figure(figsize=(20,8))
plt.plot(y_pred[400:500], label="prediction", linewidth=2.0,color='blue')
plt.plot(y_test[400:500].values, label="real_values", linewidth=2.0,color='lightcoral')
plt.legend(loc="best")
plt.title("KNN Multi-class Classification")
plt.savefig('plots/knn_real_pred_multi.png')
plt.show()

"""### **Saving Trained Model to Disk**"""

pkl_filename = "./models/knn_multi.pkl"
if (not path.isfile(pkl_filename)):
  # saving the trained model to disk
  with open(pkl_filename, 'wb') as file:
    pickle.dump(knn_multi, file)
  print("Saved model to disk")
else:
  print("Model already saved")


"""## **Multi Layer Perceptron**"""
mlp_multi = MLPClassifier()
mlp_multi1 = MLPClassifier(random_state=123, solver='adam', max_iter=8000)
mlp_multi.fit(X_train,y_train)

y_pred = mlp_multi.predict(X_test)
print("********************Multi Layer Perceptron***************************")
print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - " , metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - " , metrics.explained_variance_score(y_test, y_pred)*100)
print("Accuracy - ",accuracy_score(y_test,y_pred)*100)

print(classification_report(y_test, y_pred,target_names=le2.classes_))

"""### **Real and Predicted Data**"""

mlp_multi_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
mlp_multi_df.to_csv('./predictions/mlp_real_pred_multi.csv')
mlp_multi_df

"""### **Plot between Real and Predicted Data**"""

plt.figure(figsize=(20,8))
plt.plot(y_pred[100:300], label="prediction", linewidth=2.0,color='blue')
plt.plot(y_test[100:300].values, label="real_values", linewidth=2.0,color='lightcoral')
plt.legend(loc="best")
plt.title("MLP Multi-class Classification")
plt.savefig('plots/mlp_real_pred_multi.png')
plt.show()

"""### **Saving Trained Model to Disk**"""

pkl_filename = "./models/mlp_multi.pkl"
if (not path.isfile(pkl_filename)):
  # saving the trained model to disk
  with open(pkl_filename, 'wb') as file:
    pickle.dump(mlp_multi, file)
  print("Saved model to disk")
else:
  print("Model already saved")


"""## **Gradient Boosting Machine (GBM)**"""

gbc_multi = GradientBoostingClassifier()
gbc_multi1 = GradientBoostingClassifier(n_estimators=300, random_state=123)
gbc_multi.fit(X_train, y_train)

y_pred = gbc_multi.predict(X_test)
print("****************** **Gradient Boosting Machine (GBM)***************************")
print("Mean Absolute Error - ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - ", metrics.explained_variance_score(y_test, y_pred)*100)
print("Accuracy - ", accuracy_score(y_test, y_pred)*100)

print(classification_report(y_test, y_pred, target_names=le2.classes_))

"""### **Real and Predicted Data**"""

gbc_multi_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
gbc_multi_df.to_csv('./predictions/gbc_real_pred_multi.csv')
gbc_multi_df

"""### **Plot between Real and Predicted Data**"""

plt.figure(figsize=(20,8))
plt.plot(y_pred[100:200], label="prediction", linewidth=2.0, color='blue')
plt.plot(y_test[100:200].values, label="real_values", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("Gradient Boosting Machine (GBM) Multi-class Classification")
plt.savefig('plots/gbc_real_pred_multi.png')
plt.show()

"""### **Saving Trained Model to Disk**"""

pkl_filename = "./models/gbc_multi.pkl"
if (not path.isfile(pkl_filename)):
    # saving the trained model to disk
    with open(pkl_filename, 'wb') as file:
        pickle.dump(gbc_multi, file)
    print("Saved model to disk")
else:
    print("Model already saved")


from sklearn.ensemble import StackingClassifier

# Define the base classifiers
estimators = [
    ('logr', logr_multi1),
    ('knn' , knn_multi1),
    ('mlp', mlp_multi1),
    ('svc', lsvm_multi1),
    ('gbc', gbc_multi1)
]

# Define the meta-classifier
meta_classifier = LogisticRegression(random_state=123, max_iter=5000, solver='newton-cg', multi_class='multinomial')

# Create the Stacking Classifier
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_classifier)

# Train the Stacking Classifier
stacking_clf.fit(X_train, y_train)

# Make predictions
y_pred = stacking_clf.predict(X_test)

# Print metrics
print("****************** **Stacking Classifier***************************")
print("Mean Absolute Error - ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("Accuracy - ", accuracy_score(y_test, y_pred) * 100)

print(classification_report(y_test, y_pred, target_names=le2.classes_))

import pandas as pd

# Assuming y_test and y_pred are already defined from your model evaluation

# Create a DataFrame to store actual and predicted values
sc_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Specify the file path where you want to save the CSV
csv_filename = 'predictions/sc_real_pred_multi.csv'

# Save the DataFrame to CSV
sc_df.to_csv(csv_filename, index=False)  # Set index=False to omit row indices in the CSV

# Optionally, you can print or display the DataFrame
print("Saved actual and predicted values to:", csv_filename)
print(sc_df.head())  # Display the first few rows of the DataFrame
pkl_filename = "./models/stacking_classifier.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(stacking_clf, file)
    print("Saved model to disk")
else:
    print("Model already saved")