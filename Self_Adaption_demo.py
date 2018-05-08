# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/1/31 10:30
# file: Self_Adaption_demo.py
# description:

from Common.Evaluation_tool import Evaluation
from read import read_Microarray_Dataset
from ECOC.Classifier import Self_Adaption_ECOC

if __name__ == '__main__':
    filepath = 'E:/workspace/pycharm/microarray/treated_data/'

    each = 'Breast'
    train_path = filepath + each + '_train.csv'
    test_path =  filepath + each + '_test.csv'
    train_data, train_label = read_Microarray_Dataset(train_path)
    test_data, test_label = read_Microarray_Dataset(test_path)

    E = eval('Self_Adaption_ECOC()')
    E.fit(train_data, train_label,dc_option=['F1','N2','F3'])

    predicted_label = E.predict(test_data)
    res = Evaluation(test_label, predicted_label).evaluation(simple=True)


    print('res:%d' %round(res))
