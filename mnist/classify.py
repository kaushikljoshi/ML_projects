# Decision Tree Classification  

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys

class Classifier():
    
    #-----------------------------------------------------------------------------------------
    def __init__(self, data_name = None, X_data = None, Y_data = None, training_size = None,
                 validation_set = None, feature_scaling = None):
        
        if X_data is None:
            print ('Cannot perform classification without  data of independent variables')
            sys.exit()
        else:
            self.X = X_data
        
        if Y_data is None:
            print ('Cannot perform classification without vector of dependent variables')
            sys.exit()
        else:
            self.y = Y_data
            self.y_labels = []
            self.y_labels_pop =[]
        
        self.training_size = []
        if training_size is None:
            self.training_size.append(0.2)
        else:
            if (type(training_size) is not list):
                #currently it assumes that it's just a number
                self.training_size.append(training_size)
            else:
                self.training_size = training_size
        self.error = []
        self.best_training_size = 0
        
        if validation_set is None:
            self.validation_set = 0
        else:
            self.validation_set = validation_set
        
        if feature_scaling:
            self.feature_scaling = True
        else:
            self.feature_scaling = False
        
        if data_name is None:
            self.data_name = 'input_data'
        else:
            self.data_name = data_name
    #____________________________________________________________________________________________
        
    #--------------------------------------------------------------------------------------------
    def pre_process(self, size):
        from sklearn.model_selection import train_test_split
        #first split the data into training set and test set
        if (self.validation_set != 0):
            self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, 
                                                                   train_size = size, random_state = 0)
            self.X_validation, self.X_test, self.y_validation, self.y_test = train_test_split(self.X_temp, self.y_temp, train_size = self.validation_set, random_state = 0)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = size, random_state = 0)
        
        #Apply feature scale if requested by user
        if (self.feature_scaling):
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            
            if (self.validation_set != 0):
                self.X_validation = sc.fit_transform(self.X_validation)

            self.X_train = sc.fit_transform(self.X_train)
            self.X_test = sc.transform(self.X_test)
    #_____________________________________________________________________________________________
            
    #---------------------------------------------------------------------------------------------
    def find_optimum_training_size(self,method_object):
        from sklearn.metrics import accuracy_score
        max_accuracy = -1.0
        for i1 in range(0,len(self.training_size)):    
            #prepare training and test sets
            self.pre_process(self.training_size[i1])
            
            #Fit the classifier and then calculate error for that training set
            method_object.fit(self.X_train, self.y_train)
            y_pred_training = method_object.predict(self.X_train)
            accuracy_training = accuracy_score(self.y_train,y_pred_training)
            if (self.validation_set != 0):
                y_pred_validation = method_object.predict(self.X_validation)
                accuracy_validation = accuracy_score(self.y_validation,y_pred_validation)
            
            if (self.validation_set != 0):
                if (accuracy_validation > max_accuracy):
                    max_accuracy = accuracy_validation
                    self.best_training_size = self.training_size[i1]
                    self.error.append([self.training_size[i1],accuracy_training,
                                       accuracy_validation])
            else:
                if (accuracy_training > max_accuracy):
                    max_accuracy = accuracy_training
                    self.best_training_size = self.training_size[i1]
                    self.error.append([self.training_size[i1],accuracy_training])
            
        return max_accuracy
    #____________________________________________________________________________________________
    
   #----------------------------------------------------------------------------------------------------------- 
    def linear_svm(self):
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        
        #Initialize svm object
        classifier = SVC(kernel = 'linear', random_state = 0)
        max_accuracy = self.find_optimum_training_size(classifier)
        
        if (self.validation_set != 0):
            self.line_plot(self.error, ylabel = 'Accuracy',xlabel = 'Size of training_set',
                           legends = ['Training Set','Validation Set'])
        else:
            self.line_plot(self.error, ylabel = 'Accuracy',xlabel = 'Size of training_set')
        print ('Best accuracy of %f obtained for training size of %d' %(max_accuracy,self.best_training_size))
        
        #identify best regularization value, C using validation set
        self.pre_process(self.best_training_size)
        #c_values = np.linspace(1e-4,1,num = 5)        
        c_values = [1e-4,1e-3,1e-1,1.0,3.0,5.0,10.0]
        c_accuracy = []
        c_best_accuracy = -1.0
        c_best_value = 1.0
        for value in c_values:
            c_classifier = SVC(C = value, kernel = 'linear',random_state = 0)
            c_classifier.fit(self.X_train, self.y_train)
            y_pred_validation = c_classifier.predict(self.X_validation)
            accuracy = accuracy_score(self.y_validation,y_pred_validation)
            c_accuracy.append([math.log(value,10),accuracy])
            if (accuracy > c_best_accuracy):
                c_best_accuracy = accuracy
                c_best_value = value
        
        print ('Best accuracy of %f obtained for C-value of %f' %(c_best_accuracy,c_best_value))
        
        #self.line_plot(c_accuracy, ylabel ='% Accuracy', xlabel = 'log(C-value)')
        
        for i1 in range(0,len(c_values)):
            print ('%f %f\n'%(c_accuracy[i1][0],c_accuracy[i1][1]))
            
        #Train the classifier with best training size and best C-value
        #Then predict its performance on test data
        final_classifier = SVC(C = c_best_value, kernel = 'linear',random_state = 0)
        final_classifier.fit(self.X_train, self.y_train)
        self.y_pred_test = final_classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, self.y_pred_test)
        print ('Overall accuracy of classifier on test set is %f'%(accuracy))
   #___________________________________________________________________________________________________________
   
   #---------------------------------------------------------------------------------------------------
    def get_class_error(self):
        from sklearn.metrics import confusion_matrix
        
        if (self.y_labels == []):
            self.get_labels()
        
        self.class_error = {}
        
        self.cm = confusion_matrix(self.y_test,self.y_pred_test,labels = self.y_labels)
        
        total_class = np.sum(self.cm,axis = 1)
        for i in range(0,len(self.y_labels)):
            self.class_error[self.y_labels[i]] = (total_class[i]-self.cm[i,i])/total_class[i]
        
        bar_x = []
        bar_y = []
        for key in sorted(self.class_error.keys()):
            bar_x.append(key)
            bar_y.append(self.class_error[key]*100)
        
        self.simple_bar_plot(bar_x,bar_y,xlabel = 'Class Label', ylabel = '% Error', 
                             title = 'Class Errors')
        
   #___________________________________________________________________________________________________
   
   #---------------------------------------------------------------------------------------------------
    def get_labels(self):
        
        for i in range(0,self.y.shape[0]):
            for j in range(0,self.y.shape[1]):
                if self.y[i,j] in self.y_labels:
                    index = self.y_labels.index(self.y[i,j])
                    self.y_labels_pop[index] = self.y_labels_pop[index] + 1
                else:
                    self.y_labels.append(self.y[i,j])
                    self.y_labels_pop.append(1)
        
        #self.y_labels.sort()
   #___________________________________________________________________________________________________
   
   #___________________________________________________________________________________________________
    def simple_bar_plot(self,xdata = None, ydata = None ,xlabel = None, 
                        ylabel = None, title = None):
        plt.figure()
        
        if xdata is None:
            print ('Cannot plot bar plot without x-values')
            sys.exit()
        
        if ydata is None:
            print ('Cannot plot bar plot without y-values')
            sys.exit()
        
        if xlabel is None:
            plt.xlabel("x",fontsize = 15)
        else:
            plt.xlabel(xlabel, fontsize = 15)
        
        if ylabel is None:
            plt.xlabel("y",fontsize = 15)
        else:
            plt.ylabel(ylabel,fontsize = 15)
        
        if title is None:
            plt.title('Bar Diagram')
        else:
            plt.title(title)
        
            
        plt.bar(xdata,ydata)
        plt.xticks(xdata)
        
        plt.savefig(title +'.png')
        
   #---------------------------------------------------------------------------------------------------
   
   
   #----------------------------------------------------------------------------------------------------
    def line_plot(self,data,xlabel = None, ylabel = None,legends = None):
        plt.figure()
        if legends is None: 
            legends = []
            for i1 in range(0,len(data[0])-1):
                legends.append('Data_' + str(i1+1))
        
        for i1 in range(1,len(data[0])):
            plt.plot([row[0] for row in data], [row[i1] for row in data], label = legends[i1-1])
        
        if xlabel is None:
            plt.xlabel("x",fontsize = 15)
        else:
            plt.xlabel(xlabel, fontsize = 15)
        
        if ylabel is None:
            plt.xlabel("y",fontsize = 15)
        else:
            plt.ylabel(ylabel,fontsize = 15)
        
        plt.legend(fontsize = 15)
        
        plt.savefig(self.data_name +'.png')
   #___________________________________________________________________________________________________

#-----------------------------------------------------------------------------------------------
def import_mat_data_set():
    from scipy import io
    data_name = "mnist"
    data = io.loadmat("%s_data.mat" % data_name)
    print("\nloaded %s data!" % data_name)
    fields = "test_data", "training_data", "training_labels"
    for field in fields:
        print(field, data[field].shape)
    
    X = data['training_data']
    y = data['training_labels']
        
    return X,y
#_________________________________________________________________________________________________

# Importing the dataset
#dataset = pd.read_csv('diabetes.csv')
#X = dataset.iloc[:, 0:8].values
#y = dataset.iloc[:, 8].values
X, y = import_mat_data_set()
#decision_tree_classification(X,y,'Diabetes')
#logistic_regression_classification(X,y)
cl = Classifier(X_data = X, Y_data = y, training_size = [100,200,500,1000,2000,8000], 
                feature_scaling = False, validation_set = 10000)
#cl.pre_process()
cl.linear_svm()
cl.get_class_error()