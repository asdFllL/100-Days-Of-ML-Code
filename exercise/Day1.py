import numpy as np
import pandas as pd

dataset = pd.read_csv(
    r'/Users/ky_c/Desktop/github/100-Days-Of-ML-Code/datasets/Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Handling the missing data Imputer为缺失值处理器
# 老方法不管用了：from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer as Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean")
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])

# Creating a dummy variable
# categorical_feature失效
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer([('Country', OneHotEncoder(), [0])], 
                                  remainder= 'passthrough') # 引用 ColumnTransformer来代替 之前版本
X = onehotencoder.fit_transform(X) # .toarray() 国家分类不用再 toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)

# Splitting the datasets into training sets and Test sets
# cross_validation包不使用，用model_selection代替
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y,
                                                    test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)