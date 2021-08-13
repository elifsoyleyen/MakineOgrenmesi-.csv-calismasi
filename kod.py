# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:02:05 2020

@author: elif
"""

import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Dense 
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from PyQt5.QtWidgets import QApplication, QTableView
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,  cross_val_predict
from sklearn import metrics

from tasarim import Ui_Dialog
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,StackingClassifier
from sklearn.utils import check_array
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC 
from scipy import stats
from scipy.stats import uniform
from scipy.stats import randint

class MainWindow(QWidget,Ui_Dialog):
    dataset_file_path = ""
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)  
        self.setupUi(self)
        self.pushButton.clicked.connect(self.islemler)  
        self.pushButton_2.clicked.connect(self.veribilgi) 
        self.pushButton_13.clicked.connect(self.yapaysinir)
        self.pushButton_3.clicked.connect(self.importet)
        self.pushButton_7.clicked.connect(self.gridsearch)
        self.pushButton_10.clicked.connect(self.algoritmalar)
        self.pushButton_11.clicked.connect(self.randomsearch)
        self.pushButton_12.clicked.connect(self.topluogrenme)
        self.comboBox.addItem("0.2")
        self.comboBox.addItem("0.3")
        self.comboBox.addItem("0.4")
        self.comboBox.addItem("0.5")
        self.comboBox_2.addItem("Hold Out")
        self.comboBox_2.addItem("K Fold")                
        self.comboBox_3.addItem("Bagging")
        self.comboBox_3.addItem("Voting")
        self.comboBox_3.addItem("Boosting")
        self.comboBox_4.addItem("SVM")
        self.comboBox_4.addItem("Logistic Regrasyon")
        self.comboBox_4.addItem("DecisionTreeClassifier")
        self.comboBox_5.addItem("5")
        self.comboBox_5.addItem("10")
        self.comboBox_6.addItem("1")
        self.comboBox_6.addItem("2")
        self.comboBox_6.addItem("3")
        self.comboBox_6.addItem("4")
        self.comboBox_6.addItem("5")
        self.comboBox_6.addItem("6")
        self.comboBox_6.addItem("7")
        self.comboBox_6.addItem("8")
        self.comboBox_6.addItem("9")
        self.comboBox_6.addItem("10")
        self.comboBox_7.addItem("Logistic Regrasyon")
        self.comboBox_7.addItem("Decision Tree Classifier")
        self.comboBox_7.addItem("SVM")
        self.comboBox_7.addItem("KNN")
        self.comboBox_7.addItem("Navie Bayes")
        self.comboBox_8.addItem("SVM")
        self.comboBox_8.addItem("Logistic Regrasyon")
        self.comboBox_8.addItem("DecisionTreeClassifier")
        
    dataset = [] 
    def importet(self):
        self.label_33.setText("")
        self.textEdit.setText("")
        self.textEdit_2.setText("")
        self.textEdit_3.setText("")
        self.textEdit_4.setText("")
        
        filename = QFileDialog.getOpenFileName()
        path = filename[0]  
        print("Seçilen veriseti:",path)
        self.dataset=pd.read_excel(path, comment='#')  
        for i in self.dataset.columns:
             deger=self.dataset[i].isnull().any()
             if(deger==True):
                 self.dataset[i].fillna(self.dataset[i].mean(), inplace=True)  
                 self.label_23.setText(str('Null Olan Değerler Dolduruldu..'))
        self.dataset.info() 
        self.veriler =self.dataset.values
        veri = self.veriler[:, 0:self.veriler.shape[0]]
        self.tableWidget.setRowCount(veri.shape[0])
        self.tableWidget.setColumnCount(veri.shape[1])
        for i in range(0, self.veriler.shape[0]):
            for j in range(0, self.veriler.shape[1]):
                bakalim=self.veriler[i][j]
                if bakalim=="Male":
                    self.veriler[i][j]=1
                elif bakalim=="Female":
                    self.veriler[i][j]=0
                self.tableWidget.setItem(i,j ,QtWidgets.QTableWidgetItem(str(self.veriler[i][j]).replace(".0","")))
        print('DENEME',self.veriler)
        

    def islemler(self):
        testayirma=self.comboBox_2.currentText()
        #self.input=len(labels)-1
        self.ndegeri = self.comboBox_5.currentText()
        self.test_size = self.comboBox.currentText()
        labels=self.dataset.columns
        x_data=self.veriler[:,0:len(labels)-1]
        self.y=self.veriler[:,len(labels)-1] #son sutün
        self.y = self.y.astype ('int') 
        self.target=self.dataset.columns[-1]

         #normalizasyon işlemi
        from sklearn.preprocessing import MinMaxScaler 
        mms = MinMaxScaler() 
        self.X = mms.fit_transform(x_data)
        self.dengelidengesiz()
       
        if testayirma=='Hold Out':
           
            if self.radioButton.isChecked():
                self.label_25.setText(str(self.X.shape))
                pca_model = PCA(n_components=2).fit(self.X)
                self.X = pca_model.transform(self.X)
                self.label_26.setText(str(self.X.shape))
                self.hold_out()
                self.label_30.setText(str("HOLD-OUT VE PCA İŞLEMİ İLE HAZIRLANDI.."))
                self.label_41.setText(str("HOLD-OUT VE PCA İŞLEMİ İLE HAZIRLANDI.."))
                self.label_42.setText(str("HOLD-OUT VE PCA İŞLEMİ İLE HAZIRLANDI.."))


            if self.radioButton_2.isChecked():
                self.label_27.setText(str(self.X.shape))
                selector = SelectKBest(chi2, k=5)
                selector.fit(self.X, self.y)
                self.X = selector.transform(self.X)
                self.label_28.setText(str(self.X.shape))
                self.hold_out()
                self.label_30.setText(str("HOLD-OUT Kİ-KARE İŞLEMİ İLE HAZIRLANDI.."))
                self.label_41.setText(str("HOLD-OUT Kİ-KARE İŞLEMİ İLE HAZIRLANDI.."))
                self.label_42.setText(str("HOLD-OUT Kİ-KARE İŞLEMİ İLE HAZIRLANDI.."))
                print("HoldOutChiKare") 
            if self.radioButton.isChecked()==False and self.radioButton_2.isChecked()==False:
                 print("HoldOutHicbiri") 
                 self.hold_out()  
                 self.label_30.setText(str("YALNIZCA HOLD OUT İLE HAZIRLANDI.."))
                 self.label_41.setText(str("YALNIZCA HOLD OUT İLE HAZIRLANDI.."))
                 self.label_42.setText(str("YALNIZCA HOLD OUT İLE HAZIRLANDI.."))
                 

        
        if testayirma=='K Fold': 
            if self.radioButton.isChecked():
                self.label_25.setText(str(self.X.shape))
                self.secenek = self.comboBox_6.currentText()
                pca_model = PCA(n_components=2).fit(self.X)
                self.X = pca_model.transform(self.X)
                self.label_26.setText(str(self.X.shape))
              
                self.caprazdogrulama()
                self.label_30.setText(str(" K-FOLD VE PCA İŞLEMİ İLE HAZIRLANDI.."))
                self.label_41.setText(str(" K-FOLD VE PCA İŞLEMİ İLE HAZIRLANDI.."))
                self.label_42.setText(str(" K-FOLD VE PCA İŞLEMİ İLE HAZIRLANDI.."))
                
            if self.radioButton_2.isChecked():
                self.label_27.setText(str(self.X.shape))
                self.secenek = self.comboBox_6.currentText()
                selector = SelectKBest(chi2, k=5)
                selector.fit(self.X, self.y)
                self.X = selector.transform(self.X)
                self.label_28.setText(str(self.X.shape))
                self.caprazdogrulama()
                self.label_30.setText(str(" K-FOLD VE Kİ-KARE İŞLEMİ İLE HAZIRLANDI.."))
                self.label_41.setText(str(" K-FOLD VE Kİ-KARE İŞLEMİ İLE HAZIRLANDI.."))
                self.label_42.setText(str(" K-FOLD VE Kİ-KARE İŞLEMİ İLE HAZIRLANDI.."))
                
            if self.radioButton.isChecked()==False and self.radioButton_2.isChecked()==False:
                self.caprazdogrulama()
                self.label_30.setText(str(" YALNIZCA K-FOLD İŞLEMİ İLE HAZIRLANDI.."))
                self.label_41.setText(str(" YALNIZCA K-FOLD İŞLEMİ İLE HAZIRLANDI.."))
                self.label_42.setText(str(" YALNIZCA K-FOLD İŞLEMİ İLE HAZIRLANDI.."))

 
    def hold_out(self):
        self.X_train, self.X_test, self.y_train,self.y_test=train_test_split(self.X,self.y,  test_size=float(self.test_size),random_state=0)
        hold_out = {"x_train"  : self.X_train,
                  "x_test"  : self.X_test
                 ,"y_train" : self.y_train
                 ,"y_test"  : self.y_test}
        for i in hold_out:
            print(f"{i}: satır sayısı {hold_out.get(i).shape[0]}")
            self.textEdit.setText(str(self.X_train))
            self.textEdit_2.setText(str(self.X_test))
            self.textEdit_3.setText(str(self.y_train))
            self.textEdit_4.setText(str(self.y_test))
        
    def caprazdogrulama(self):
        self.ndegeri = self.comboBox_5.currentText()
        self.secenek = self.comboBox_6.currentText()
        say=0
        kf=KFold(n_splits=int(self.ndegeri), random_state=1, shuffle=True)
        for train_x, test_x in kf.split(self.X):
            say+=1
            print("TRAIN:", train_x, "TEST:", test_x)
            if(say==int(self.secenek)):
                self.X_train, self.X_test = self.X[train_x], self.X[test_x]
                self.y_train, self.y_test = self.y[train_x], self.y[test_x]
                self.textEdit.setText(str(self.X_train))
                self.textEdit_2.setText(str(self.X_test))
                self.textEdit_3.setText(str(self.y_train))
                self.textEdit_4.setText(str(self.y_test))


    def algoritmalar(self):
        self.secimyap = self.comboBox_7.currentText()
        if self.secimyap=='Decision Tree Classifier':
            self.model=DecisionTreeClassifier()
            self.model.fit(self.X_train,self.y_train)
            self.y_pred=self.model.predict(self.X_test)
            self.textEdit_8.setText(str("Başarı oranı:{:0.2f}".format(accuracy_score(self.y_test, self.y_pred))))
            self.textEdit_19.setText(str("Karışıklık Matrisi:\n{}".format(confusion_matrix(self.y_test, self.y_pred))))
            confMat = confusion_matrix(self.y_test, self.y_pred)
            TN = confMat[0,0]
            TP = confMat[1,1]
            FN = confMat[1,0]
            FP = confMat[0,1]
            sensivity = float(TP)/(TP+FN)
            specifity = float(TN)/(TN+FP)
            self.textEdit_15.setText(str('Sensivity:%.3f' %sensivity))
            self.textEdit_16.setText(str('Specifity:%.3f' %specifity))
            tahmin_test = self.model.predict(self.X_test)
            self.textEdit_9.setText(str('R2: %.4f' % r2_score(self.y_test, tahmin_test)))
            self.textEdit_10.setText(str('MSE: %.4f' % mean_squared_error(self.y_test, tahmin_test)))
        
            self.roc()
            plt.savefig('./rocdt.png')
            self.pixmap = QPixmap("./rocdt.png") 
            self.label_39.setPixmap(self.pixmap)
            self.confmatrix()
            plt.savefig('./confdt.png')
            self.pixmap = QPixmap("./confdt.png") 
            self.label_15.setPixmap(self.pixmap)
           
            
            
        if self.secimyap=='SVM':
             self.model=SVC(probability=True)
             self.model.fit(self.X_train,self.y_train)
             self.y_pred=self.model.predict(self.X_test)
             self.textEdit_8.setText(str("Başarı oranı:{:0.2f}".format(accuracy_score(self.y_test, self.y_pred))))
             self.textEdit_19.setText(str("Karışıklık Matrisi:\n{}".format(confusion_matrix(self.y_test, self.y_pred))))
             confMat = confusion_matrix(self.y_test, self.y_pred)
             TN = confMat[0,0]
             TP = confMat[1,1]
             FN = confMat[1,0]
             FP = confMat[0,1]
             sensivity = float(TP)/(TP+FN)
             specifity = float(TN)/(TN+FP)
             self.textEdit_15.setText(str('Sensivity:%.3f' %sensivity))
             self.textEdit_16.setText(str('Specifity:%.3f' %specifity))
             tahmin_test = self.model.predict(self.X_test)
             self.textEdit_9.setText(str('R2: %.4f' % r2_score(self.y_test, tahmin_test)))
             self.textEdit_10.setText(str('MSE: %.4f' % mean_squared_error(self.y_test, tahmin_test)))
             self.textEdit_11.setText(str('MAE:%.4f' % metrics.mean_absolute_error(self.y_test, tahmin_test)))
             self.roc()
             plt.savefig('./rocsvm.png')
             self.pixmap = QPixmap("./rocsvm.png") 
             self.label_39.setPixmap(self.pixmap)
             self.confmatrix()
             plt.savefig('./confsvm.png')
             self.pixmap = QPixmap("./confsvm.png") 
             self.label_15.setPixmap(self.pixmap)
             
        if self.secimyap=='Logistic Regrasyon':
              self.model=LogisticRegression()
              self.model.fit(self.X_train,self.y_train)
              self.y_pred=self.model.predict(self.X_test)
              self.textEdit_8.setText(str("Başarı oranı:{:0.2f}".format(accuracy_score(self.y_test, self.y_pred))))
              self.textEdit_19.setText(str("Karışıklık Matrisi:\n{}".format(confusion_matrix(self.y_test, self.y_pred))))
              confMat = confusion_matrix(self.y_test, self.y_pred)
              TN = confMat[0,0]
              TP = confMat[1,1]
              FN = confMat[1,0]
              FP = confMat[0,1]
              sensivity = float(TP)/(TP+FN)
              specifity = float(TN)/(TN+FP)
              self.textEdit_15.setText(str('Sensivity:%.3f' %sensivity))
              self.textEdit_16.setText(str('Specifity:%.3f' %specifity))
              tahmin_test = self.model.predict(self.X_test)
              self.textEdit_9.setText(str('R2: %.4f' % r2_score(self.y_test, tahmin_test)))
              self.textEdit_10.setText(str('MSE: %.4f' % mean_squared_error(self.y_test, tahmin_test)))
              self.textEdit_11.setText(str('MAE:%.4f' % metrics.mean_absolute_error(self.y_test, tahmin_test)))
            
              self.roc()
              plt.savefig('./roclog.png')
              self.pixmap = QPixmap("./roclog.png") 
              self.label_39.setPixmap(self.pixmap)
              
              self.confmatrix()
              plt.savefig('./conflog.png')
              self.pixmap = QPixmap("./conflog.png") 
              self.label_15.setPixmap(self.pixmap)
              
        if self.secimyap=='KNN':
            self.model = KNeighborsClassifier(n_neighbors=3)
            self.model.fit(self.X_train, self.y_train)
            self.y_pred=self.model.predict(self.X_test)
            self.textEdit_8.setText(str("Başarı oranı:{:0.2f}".format(accuracy_score(self.y_test, self.y_pred))))
            self.textEdit_19.setText(str("Karışıklık Matrisi:\n{}".format(confusion_matrix(self.y_test, self.y_pred))))
            confMat = confusion_matrix(self.y_test, self.y_pred)
            TN = confMat[0,0]
            TP = confMat[1,1]
            FN = confMat[1,0]
            FP = confMat[0,1]
            sensivity = float(TP)/(TP+FN)
            specifity = float(TN)/(TN+FP)
            self.textEdit_15.setText(str('Sensivity:%.3f' %sensivity))
            self.textEdit_16.setText(str('Specifity:%.3f' %specifity))
            tahmin_test = self.model.predict(self.X_test)
            self.textEdit_9.setText(str('R2: %.4f' % r2_score(self.y_test, tahmin_test)))
            self.textEdit_10.setText(str('MSE: %.4f' % mean_squared_error(self.y_test, tahmin_test)))
            self.textEdit_11.setText(str('MAE:%.4f' % metrics.mean_absolute_error(self.y_test, tahmin_test)))
            self.roc()
            plt.savefig('./rocknn.png')
            self.pixmap = QPixmap("./rocknn.png") 
            self.label_39.setPixmap(self.pixmap)
            self.confmatrix()
            plt.savefig('./confknn.png')
            self.pixmap = QPixmap("./confknn.png") 
            self.label_15.setPixmap(self.pixmap)
            
        if self.secimyap=='Navie Bayes':
            self.model=GaussianNB()
            self.model.fit(self.X_train,self.y_train)
            self.y_pred=self.model.predict(self.X_test)
            self.textEdit_8.setText(str("Başarı oranı:{:0.2f}".format(accuracy_score(self.y_test, self.y_pred))))
            self.textEdit_19.setText(str("Karışıklık Matrisi:\n{}".format(confusion_matrix(self.y_test, self.y_pred))))
            confMat = confusion_matrix(self.y_test, self.y_pred)
            TN = confMat[0,0]
            TP = confMat[1,1]
            FN = confMat[1,0]
            FP = confMat[0,1]
            sensivity = float(TP)/(TP+FN)
            specifity = float(TN)/(TN+FP)
            self.textEdit_15.setText(str('Sensivity:%.3f' %sensivity))
            self.textEdit_16.setText(str('Specifity:%.3f' %specifity))
            tahmin_test = self.model.predict(self.X_test)
            self.textEdit_9.setText(str('R2: %.4f' % r2_score(self.y_test, tahmin_test)))
            self.textEdit_10.setText(str('MSE: %.4f' % mean_squared_error(self.y_test, tahmin_test)))
            self.textEdit_11.setText(str('MAE:%.4f' % metrics.mean_absolute_error(self.y_test, tahmin_test)))
            
            plt.figure(figsize=(14,4))
            plt.subplot(1, 2, 1)
            plt.plot(self.y_test)
            plt.plot(self.y_pred)
            plt.title('Tahmin')
            plt.legend(['Test', 'Tahmin '], loc='upper left')
            
            self.roc()
            plt.savefig('./rocnavie.png')
            self.pixmap = QPixmap("./rocnavie.png") 
            self.label_39.setPixmap(self.pixmap)
            self.confmatrix()
            plt.savefig('./confnb.png')
            self.pixmap = QPixmap("./confnb.png") 
            self.label_15.setPixmap(self.pixmap)
            
       
            
            

        
    def roc(self):
        pred_prob1 = self.model.predict_proba(self.X_test)
        fpr, tpr, thresh = roc_curve(self.y_test, pred_prob1[:,1], pos_label=1)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr,color='red')
        plt.plot([0,1], [0,1], linestyle='--', color='green')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        
        
        
        
    def confmatrix(self):
        plot_confusion_matrix(self.model,self.X_test, self.y_test,cmap=plt.cm.Blues)
      

    def veribilgi(self):       
        liver_patients, not_liver_patinets=self.dataset[self.target].value_counts()
        self.textEdit_5.setText(str(" hasta sayısı:{}, \n hasta olmayanların sayısı:{}".\
                                    format(liver_patients,not_liver_patinets)))


    def gridsearch(self):
        gridayir=self.comboBox_4.currentText()
        if gridayir=='SVM':
             params_clfs=list()
             svm_params=[
                {'kernel':['rbf'], 'gamma':[1e-3, 1e-4]},
                {'kernel':['linear'], 'C':[1, 10, 100, 1000]}       
                ]
             params_clfs.append((SVC(),svm_params))                  
             for clf, param in params_clfs:
                  grid_search=GridSearchCV(clf, param, cv=5)
                  grid_search.fit(self.X_train, self.y_train)
                  test_means=grid_search.cv_results_['mean_test_score']  
                  y_pred=grid_search.predict(self.X_test)
                  confMat = confusion_matrix(self.y_test, y_pred)
                  TN = confMat[0,0]
                  TP = confMat[1,1]
                  FN = confMat[1,0]
                  FP = confMat[0,1]
                  sensivity = float(TP)/(TP+FN)
                  specifity = float(TN)/(TN+FP)
                  self.textEdit_6.setText(str("{} İçin sklearn GridSearchCV Sonuçları".format(clf.__class__.__name__)) 
                    +str("best params:{}".format(grid_search.best_params_))
                           +   "\n"    +str("ortalama test sonucu:{:.2f}".format(np.mean(test_means))) 
                           +   "\n"    +str("en iyi parametre sonucu:{:.2f}".format(accuracy_score(self.y_test, y_pred)))
                           +   "\n"        +str("Karışıklık matrisi:\n{}".format(confusion_matrix(self.y_test, y_pred)))
                           +   "\n"        +str('Sensivity:%.3f' %sensivity)
                           +   "\n"        +str('Specifity:%.3f' %specifity))
            
        
        if gridayir=='Logistic Regrasyon':
            params_clfs=list()
            lr_params= {'penalty':['l1', 'l2'], 'C':np.logspace(0, 4, 10)}
            params_clfs.append((LogisticRegression(),lr_params))
            for clf, param in params_clfs:
                  grid_search=GridSearchCV(clf, param, cv=5)
                  grid_search.fit(self.X_train, self.y_train)
                  test_means=grid_search.cv_results_['mean_test_score']  
                  y_pred=grid_search.predict(self.X_test)
                  confMat = confusion_matrix(self.y_test, y_pred)
                  TN = confMat[0,0]
                  TP = confMat[1,1]
                  FN = confMat[1,0]
                  FP = confMat[0,1]
                  sensivity = float(TP)/(TP+FN)
                  specifity = float(TN)/(TN+FP)
                  self.textEdit_6.setText(str("{} İçin sklearn GridSearchCV Sonuçları".format(clf.__class__.__name__)) 
                    +str("best params:{}".format(grid_search.best_params_))
                           +   "\n"    +str("ortalama test sonucu:{:.2f}".format(np.mean(test_means))) 
                           +   "\n"    +str("en iyi parametre sonucu:{:.2f}".format(accuracy_score(self.y_test, y_pred)))
                           +   "\n"        +str("Karışıklık matrisi:\n{}".format(confusion_matrix(self.y_test, y_pred)))
                           +   "\n"        +str('Sensivity:%.3f' %sensivity)
                           +   "\n"        +str('Specifity:%.3f' %specifity))
   


        if gridayir=='DecisionTreeClassifier':
            params_clfs=list()
            dt_params={'max_features': ['auto', 'sqrt', 'log2'],
                   'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                   'min_samples_leaf':[1],
                   'random_state':[123]}
            params_clfs.append((DecisionTreeClassifier(),dt_params))
            for clf, param in params_clfs:
                  grid_search=GridSearchCV(clf, param, cv=5)
                  grid_search.fit(self.X_train, self.y_train)
                  test_means=grid_search.cv_results_['mean_test_score']  
                  y_pred=grid_search.predict(self.X_test)
                  confMat = confusion_matrix(self.y_test, y_pred)
                  TN = confMat[0,0]
                  TP = confMat[1,1]
                  FN = confMat[1,0]
                  FP = confMat[0,1]
                  sensivity = float(TP)/(TP+FN)
                  specifity = float(TN)/(TN+FP)
                  self.textEdit_6.setText(str("{} İçin sklearn GridSearchCV Sonuçları".format(clf.__class__.__name__)) 
                    +str("best params:{}".format(grid_search.best_params_))
                           +   "\n"    +str("ortalama test sonucu:{:.2f}".format(np.mean(test_means))) 
                           +   "\n"    +str("en iyi parametre sonucu:{:.2f}".format(accuracy_score(self.y_test, y_pred)))
                           +   "\n"        +str("Karışıklık matrisi:\n{}".format(confusion_matrix(self.y_test, y_pred)))
                           +   "\n"        +str('Sensivity:%.3f' %sensivity)
                           +   "\n"        +str('Specifity:%.3f' %specifity))
                  
   

         
                                 
    def randomsearch(self):
        random=self.comboBox_8.currentText()
        
        if random=='DecisionTreeClassifier':
             param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}
             tree = DecisionTreeClassifier()
             tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
             tree_cv.fit(self.X_train,self.y_train)
             y_pred=tree_cv.predict(self.X_test)
             confMat = confusion_matrix(self.y_test, y_pred)
             TN = confMat[0,0]
             TP = confMat[1,1]
             FN = confMat[1,0]
             FP = confMat[0,1]
             sensivity = float(TP)/(TP+FN)
             specifity = float(TN)/(TN+FP)
             print('Sensivity:%.3f' %sensivity)
             print('Specifity:%.3f' %specifity)
             self.textEdit_6.setText('Best Score: %s' % tree_cv.best_params_ +"/n"
                                     + 'Best Score: %s' % tree_cv.best_score_ 
                                     + "/n" +str('Sensivity:%.3f' %sensivity)
                                     + "/n" +str('Specifity:%.3f' %specifity))
          
            
             
        if random=='Logistic Regrasyon':
            logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,
                                  random_state=0)
            distributions = dict(C=uniform(loc=0, scale=4),
                         penalty=['l2', 'l1'])
            clf = RandomizedSearchCV(logistic, distributions, random_state=0)
            search = clf.fit(self.X_train, self.y_train)     
            y_pred=search.predict(self.X_test)
            confMat = confusion_matrix(self.y_test, y_pred)
            TN = confMat[0,0]
            TP = confMat[1,1]
            FN = confMat[1,0]
            FP = confMat[0,1]
            sensivity = float(TP)/(TP+FN)
            specifity = float(TN)/(TN+FP)
            print('Sensivity:%.3f' %sensivity)
            print('Specifity:%.3f' %specifity)
            
            self.textEdit_6.setText('Best Score: %s' % search.best_score_  +"/n"
                                    + 'Best Score: %s' % search.best_params_ 
                                    + "/n" +str('Sensivity:%.3f' %sensivity)
                                    + "/n" +str('Specifity:%.3f' %specifity))
        
        
        if random=='SVM':
            g_range = np.random.uniform(0.0, 0.3, 5).astype(float)
            C_range = np.random.normal(1, 0.1, 5).astype(float)
             
            # Check that gamma>0 and C>0 
            C_range[C_range < 0] = 0.0001
            hyperparameters = {'gamma': list(g_range), 
                    'C': list(C_range)}
            randomCV = RandomizedSearchCV(SVC(kernel='rbf', ), param_distributions=hyperparameters, n_iter=20)
            svmfit=randomCV.fit(self.X_train, self.y_train)
            # Identify optimal hyperparameter values
            best_gamma  = svmfit.best_params_['gamma']
            best_C      = svmfit.best_params_['C']
           
            y_pred=svmfit.predict(self.X_test)
            confMat = confusion_matrix(self.y_test, y_pred)
            TN = confMat[0,0]
            TP = confMat[1,1]
            FN = confMat[1,0]
            FP = confMat[0,1]
            sensivity = float(TP)/(TP+FN)
            specifity = float(TN)/(TN+FP)
            print('Sensivity:%.3f' %sensivity)
            print('Specifity:%.3f' %specifity)
            self.textEdit_6.setText('Best Score: %s' %best_gamma  +"/n"+ 'Best Score: %s' % best_C
                                    + "/n" +str('Sensivity:%.3f' %sensivity)
                                    + "/n" +str('Specifity:%.3f' %specifity))
         

    def topluogrenme(self):
        topluogrenme=self.comboBox_3.currentText()
        if topluogrenme=='Bagging':
            from sklearn.ensemble import BaggingClassifier
            self.model = BaggingClassifier(SVC(), n_estimators=20, 
                                 max_features=1.0, 
                                 max_samples=0.5,
                                )
            
            self.model.fit(self.X_train, self.y_train)
            y_pred=self.model.predict(self.X_test)
            self.textEdit_7.setText(str('R2: %.4f' % r2_score(self.y_test, y_pred)))
            self.textEdit_12.setText(str('MSE: %.4f' % mean_squared_error(self.y_test, y_pred)))
            self.textEdit_17.setText(str('MAE:%.4f' % metrics.mean_absolute_error(self.y_test, y_pred)))
            
            self.textEdit_13.setText(str("Başarı oranı:{:0.2f}".format(accuracy_score(self.y_test, y_pred)))) 
            
            self.roc()
            plt.savefig('./baggin.png')
            self.pixmap = QPixmap("./baggin.png") 
            self.label_17.setPixmap(self.pixmap)
            self.confmatrix()
            plt.savefig('./bagginconf.png')
            self.pixmap = QPixmap("./bagginconf.png") 
            self.label_18.setPixmap(self.pixmap)
            
        if topluogrenme=='Voting':
            log=LogisticRegression()
            dtc=DecisionTreeClassifier()
            svc=SVC(probability=True)
            votting_clf=VotingClassifier(
            estimators=[('lr',log),('dt',dtc),('svc',svc)]
            ,voting='soft')
            for self.model in (log,dtc,svc,votting_clf):
                self.model.fit(self.X_train,self.y_train)
                y_pred=self.model.predict(self.X_test)   
                self.textEdit_7.setText(str('R2: %.4f' % r2_score(self.y_test, y_pred)))
                self.textEdit_12.setText(str('MSE: %.4f' % mean_squared_error(self.y_test, y_pred)))
                self.textEdit_17.setText(str('MAE:%.4f' % metrics.mean_absolute_error(self.y_test, y_pred)))
                self.textEdit_13.setText(str("Başarı oranı:{:0.2f}".format(accuracy_score(self.y_test, y_pred))))
                self.roc()
                plt.savefig('./voting.png')
                self.pixmap = QPixmap("./voting.png") 
                self.label_17.setPixmap(self.pixmap)
                self.confmatrix()
                plt.savefig('./votingconf.png')
                self.pixmap = QPixmap("./votingconf.png") 
                self.label_18.setPixmap(self.pixmap)
                
        if topluogrenme=='Boosting':
            self.model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
            self.model.fit(self.X_train, self.y_train)
            self.model.score(self.X_test,self.y_test)
            y_pred=self.model.predict(self.X_test)   
            self.textEdit_13.setText(str("Başarı oranı:{:0.2f}".format(accuracy_score(self.y_test, y_pred))))
            self.textEdit_7.setText(str('R2: %.4f' % r2_score(self.y_test, y_pred)))
            self.textEdit_12.setText(str('MSE: %.4f' % mean_squared_error(self.y_test, y_pred)))
            self.textEdit_17.setText(str('MAE:%.4f' % metrics.mean_absolute_error(self.y_test, y_pred)))
            self.roc()
            plt.savefig('./bossting.png')
            self.pixmap = QPixmap("./bossting.png") 
            self.label_17.setPixmap(self.pixmap)
            self.confmatrix()
            plt.savefig('./confboot.png')
            self.pixmap = QPixmap("./confboot.png") 
            self.label_18.setPixmap(self.pixmap)



    def dengelidengesiz(self):
        if self.label_33.text()=="":
            target=self.dataset.columns[-1]
            print(target)
            target_count = self.dataset[target].value_counts()
            print('Target Count',target_count)
            deneme=self.dataset[target]
            print(target_count)
            count_class_0, count_class_1 = deneme.value_counts()
            print(count_class_0, count_class_1)
            df_class_0 = self.dataset[self.dataset[target] == 0]
            df_class_1 = self.dataset[self.dataset[target] == 1]
            bir=len(self.dataset[self.dataset[target] == 0])
            iki=len(self.dataset[self.dataset[target] == 1])
            tB=0
            tK=0
            if(bir>iki):
                tB=bir
                tK=iki
                
            else:
                tB=iki
                tK=bir
            if(tK-tB)/2>(tK/2):               
                df_class_1_over = df_class_1.sample(count_class_0, replace=True)
                df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
                print('Random over-sampling:')
                print(df_test_over[target].value_counts())
                ros = RandomOverSampler()
                self.X,self.y=ros.fit_resample(self.X, self.y)  
                self.label_33.setText(str('Veriler arasındaki oranlara bakılarak oversampling uygulandı.! '))
            else:
                self.y = self.y.astype ('int')                
                print(bir,iki)
                df_class_0_under = df_class_0.sample(count_class_1)
                df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
                print('Random under-sampling:')
                print(df_test_under[target].value_counts())
                undersample = RandomUnderSampler(sampling_strategy='majority')
                self.X, self.y = undersample.fit_resample(self.X, self.y)
                self.label_33.setText(str('Veriler arasındaki oranlara bakılarak undersampling uygulandı.! '))
                

            
    def yapaysinir(self):
        model=Sequential()
        model.add(Dense(12, kernel_initializer="uniform", activation='relu'))
        model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        print("Ağ oluşturuldu...")
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
        print("Ağ derlendi...")
        epochsdeger = int(self.lineEdit.text())
        modelfit=model.fit(self.X_train, self.y_train,
          epochs=epochsdeger,batch_size=1,validation_data=(self.X_test,self.y_test)
          )
        print("Ağ eğitildi...")
        self.y_pred=model.predict(self.X_test)
       
        #self.y_pred = np.argmax(self.y_pred, axis=1)
        self.textEdit_14.setText(str("Başarı oranı:{:0.2f}".format(accuracy_score(self.y_test, self.y_pred.round(), normalize=True))))
       
        plt.subplot(2,1,1)
        plt.figure(figsize=(5,3))
        plt.plot(modelfit.history['acc']) #cizdirme
        plt.plot(modelfit.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        plt.savefig('./loss.png')
        self.pixmap = QPixmap("./loss.png") 
        self.label_8.setPixmap(self.pixmap)
   
        
        plt.subplot(2,1,2)
        plt.figure(figsize=(5,3))
        plt.plot(modelfit.history['loss']) #cizdirme
        plt.plot(modelfit.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower https://cdn.discordapp.com/attachments/808275646413013012/808394344687468600/untitled0_5.pyright')
        plt.savefig('./loss.png')
        self.pixmap = QPixmap("./loss.png") 
        self.label_31.setPixmap(self.pixmap)
        plt.show()
        
        
        pred_prob1 = model.predict_proba(self.X_test)
        fpr, tpr, thresh = roc_curve(self.y_test, pred_prob1)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr,color='red')
        plt.plot([0,1], [0,1], linestyle='--', color='green')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig('./rocdo.png')
        self.pixmap = QPixmap("./rocdo.png") 
        self.label_32.setPixmap(self.pixmap)
        plt.show()
        
        import scikitplot.metrics as splt
        plt.figure(figsize=(5,4))
        splt.plot_confusion_matrix(self.y_test, self.y_pred.round(), normalize=False)
        plt.savefig("./confm.png")
        self.pixmap = QPixmap("./confm.png")
        self.label_24.setPixmap(self.pixmap)
     
      
       

       
            
