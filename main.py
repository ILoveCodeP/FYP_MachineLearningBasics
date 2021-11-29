from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.svm import SVC

from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier


def fill_ndarray(t1):
    for i in range(t1.shape[1]):  # 遍历每一列（每一列中的nan替换成该列的均值）
        temp_col = t1[:, i]  # 当前的一列
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:  # 不为0，说明当前这一列中有nan
            temp_not_nan_col = temp_col[temp_col == temp_col]  # 去掉nan的ndarray

            # 选中当前为nan的位置，把值赋值为不为nan的均值
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()  # mean()表示求均值。
    return t1




dataset = pd.read_csv('dataset7_magAndPhase.txt',dtype={'code':str},sep='\t',index_col=0,encoding='latin1')
col=dataset.columns.values.tolist()
col1=col[0:8000]
data_x=np.array(dataset[col1])
data_y=dataset['activity']
#data_x = fill_ndarray(data_x)
data_x = data_x[0:289,:]
data_y = data_y[0:288]

data_x3=preprocessing.scale(data_x)
#规模化特征值到一定范围内
min_max_scaler=preprocessing.MinMaxScaler()
x_minmax=min_max_scaler.fit_transform(data_x3)
#正则化
x_normalized=preprocessing.normalize(x_minmax,norm='l2')
data_x4=x_normalized

# empty=[]
# file_name = 'noselection.mat'
# savemat(file_name, {'empty':empty,'all_data':data_x4})

############训练#############
train_x, test_x, train_y, test_y = train_test_split(data_x4, data_y, test_size=0.1, random_state=5)


clf=SVC(kernel='rbf',probability=True)
clf.fit(train_x, train_y)
score = clf.score(test_x, test_y)
y_predict = clf.predict(test_x)
accuracy = metrics.accuracy_score(test_y, y_predict)
#target_names = ['1', '2', '3', '4', '5', '6']
target_names = ['1', '2', '3', '4', '5']
mat1=metrics.classification_report(test_y, y_predict, target_names=target_names)
#mat2=clf.predict_proba(test_x)
print('the result of SVM-RBF')
print(mat1)
print(y_predict)

clf=SVC(kernel='poly',degree=3)
clf.fit(train_x, train_y)
score = clf.score(test_x, test_y)
y_predict = clf.predict(test_x)
accuracy = metrics.accuracy_score(test_y, y_predict)
#target_names = ['1', '2', '3', '4', '5', '6']
target_names = ['1', '2', '3', '4', '5']
mat2=metrics.classification_report(test_y, y_predict, target_names=target_names)
print('the result of SVM-cubic')
print(mat2)

clf=SVC(kernel='poly',degree=2)
clf.fit(train_x, train_y)
score = clf.score(test_x, test_y)
y_predict = clf.predict(test_x)
accuracy = metrics.accuracy_score(test_y, y_predict)
#target_names = ['1', '2', '3', '4', '5', '6']
target_names = ['1', '2', '3', '4', '5']
mat3=metrics.classification_report(test_y, y_predict, target_names=target_names)
print('the result of SVM-quadratic')
print(mat3)

clf=neighbors.KNeighborsClassifier()
clf.fit(train_x, train_y)
score = clf.score(test_x, test_y)
y_predict = clf.predict(test_x)
accuracy = metrics.accuracy_score(test_y, y_predict)
#target_names = ['1', '2', '3', '4', '5', '6']
target_names = ['1', '2', '3', '4', '5']
mat4=metrics.classification_report(test_y, y_predict, target_names=target_names)
print('the result of KNN')
print(mat4)

clf=RandomForestClassifier()
clf.fit(train_x, train_y)
score = clf.score(test_x, test_y)
y_predict = clf.predict(test_x)
accuracy = metrics.accuracy_score(test_y, y_predict)
#target_names = ['1', '2', '3', '4', '5', '6']
target_names = ['1', '2', '3', '4', '5']
mat5=metrics.classification_report(test_y, y_predict, target_names=target_names)
print('the result of RF')
print(mat5)