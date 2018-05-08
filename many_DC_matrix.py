# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/3/1 15:03
# file: many_DC_matrix.py
# description: save many DC matrix of many datasets

import numpy as np
import copy
import types
import time
import os
import logging

from ECOCDemo.Common.Evaluation_tool import Evaluation
from ECOCDemo.ECOC.Classifier import ECOC_ONE
from ECOCDemo.ECOC.Classifier import OVO_ECOC
from ECOCDemo.ECOC.Classifier import OVA_ECOC
from ECOCDemo.ECOC.Classifier import DC_ECOC
from ECOCDemo.ECOC.Classifier import D_ECOC
from ECOCDemo.ECOC.Classifier import Dense_random_ECOC
from ECOCDemo.ECOC.Classifier import Sparse_random_ECOC
from ECOCDemo.ECOC.Classifier import Self_Adaption_ECOC
from ECOCDemo.FS.DC_Feature_selection import DC_FS, select_data_by_feature_index
from ECOCDemo.Common.Read_Write_tool import read_Microarray_Dataset
from ECOCDemo.Common.Read_Write_tool import write_matrix

def ECOC_Process(train_data,train_label,test_data,test_label,ECOC_name):

    dc_option = ['F1','F2','F3','N2','N3','N4','L3','Cluster']
    for i,each in enumerate(dc_option):
        if each in ECOC_name:
            E = eval('DC_ECOC()')
            M,index = E.create_matrix(train_data,train_label,dc_option=each)
            break

    return M

<<<<<<< HEAD
def other_ECOC_Process(train_data,train_label,ECOC_name):
    E = eval(ECOC_name + '()')
    M,index = E.create_matrix(train_data,train_label)
    return M
=======
>>>>>>> faa23ca778d809c7318e4b150e1428603a5aabba

if __name__ == '__main__':

    LOG_FORMAT = "%(message)s"
    # set log filepath, log level and info format
<<<<<<< HEAD
    logging.basicConfig(filename='E:/workspace1/ECOCDemo/Microarray_res/SVC/other_ECOC_matrix.txt', level=logging.DEBUG, format=LOG_FORMAT)
=======
    logging.basicConfig(filename='E:/workspace/pycharm/UCI/DC_matrix/DC_matrix.txt', level=logging.DEBUG, format=LOG_FORMAT)
>>>>>>> faa23ca778d809c7318e4b150e1428603a5aabba

    microarray_dataname = ['Breast','Cancers','DLBCL','GCM','Leukemia1','Leukemia2'\
                ,'Lung1','SRBCT']

    UCI_dataname = ['car', 'cleveland', 'dermatology', 'ecoli', 'isolet', 'led7digit' \
        , 'led24digit', 'letter', 'nursery', 'penbased', 'satimage', 'segment' \
        , 'shuttle', 'vehicle', 'vowel', 'yeast', 'zoo']

    fs_name = ['variance_threshold','linear_svc','tree','fclassif','RandForReg']
    ecoc_name = ['DC_ECOC F1','DC_ECOC F2','DC_ECOC F3','DC_ECOC N2','DC_ECOC N3','DC_ECOC Cluster']
<<<<<<< HEAD
    other_ecoc_name = ['OVA_ECOC','OVO_ECOC','Dense_random_ECOC','Sparse_random_ECOC'\
                ,'D_ECOC']

    folder_path = 'E:/workspace1/ECOCDemo/Microarray_data/FS_data/'
    save_folder_path = 'E:/workspace1/ECOCDemo/Microarray_res/SVC/other_ECOC_matrix/'
    selected_dataname = microarray_dataname
    selected_ecoc_name = other_ecoc_name
=======

    folder_path = 'E:/workspace/pycharm/UCI/FS_data/'
    save_folder_path = 'E:/workspace/pycharm/UCI/DC_matrix/'
    selected_dataname = UCI_dataname
    selected_ecoc_name = ecoc_name
>>>>>>> faa23ca778d809c7318e4b150e1428603a5aabba
    selected_fs_name = fs_name

    for k in range(len(selected_fs_name)):
        fin_folder_path = folder_path + selected_fs_name[k]
        fin_save_folder_path = save_folder_path + selected_fs_name[k]
        if not os.path.exists(fin_save_folder_path):
            os.mkdir(fin_save_folder_path)

        for i in range(len(selected_dataname)):

            logging.info('Time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
            logging.info('Dataset： ' + selected_dataname[i])
            train_path = fin_folder_path + '/' + selected_dataname[i] + '_train.csv'
            test_path = fin_folder_path + '/' + selected_dataname[i] + '_test.csv'
            train_data, train_label = read_Microarray_Dataset(train_path)
            test_data, test_label = read_Microarray_Dataset(test_path)

            for j in range(len(selected_ecoc_name)):

<<<<<<< HEAD
                # logging.info('DC: ' + selected_ecoc_name[j])
                # Matrix = ECOC_Process(train_data,train_label,test_data,test_label,selected_ecoc_name[j])#DC coding matrix
                # dc = selected_ecoc_name[j].split(' ')[1]
                # save_filepath = fin_save_folder_path + '/' + dc + '_' + selected_dataname[i] + '.xls'
                # write_matrix(save_filepath,Matrix)

                Matrix = other_ECOC_Process(train_data,train_label,selected_ecoc_name[j])# other ECOC
                save_file_path = fin_save_folder_path + '/' + selected_ecoc_name[j] + '_' + selected_dataname[i] + '.xls'
                write_matrix(save_file_path,Matrix)
=======
                logging.info('DC: ' + selected_ecoc_name[j])
                Matrix = ECOC_Process(train_data,train_label,test_data,test_label,selected_ecoc_name[j])
                dc = selected_ecoc_name[j].split(' ')[1]
                save_filepath = fin_save_folder_path + '/' + dc + '_' + selected_dataname[i] + '.xls'
                write_matrix(save_filepath,Matrix)

>>>>>>> faa23ca778d809c7318e4b150e1428603a5aabba



