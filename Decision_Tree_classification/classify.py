# Decision Tree Classification  

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Classifier():
    
    def __init__(self, data_name = None, X_data = None, Y_data = None, training_size = None,
                 feature_scaling = None):
        
        if X_data is None:
            print ('Cannot perform classification without vector of independent variables')
        else:
            self.X = X_data
        
        if Y_data is None:
            print ('Cannot perform classification without vector of dependent variables')
        else:
            self.Y = Y_data
        
        if training_size is None:
            self.training_size = 0.2
        else:
            self.training_size = training_size
        
        if feature_scaling:
            self.feature_scaling = True
        else:
            self.feature_scaling = False
        
        if data_name is None:
            self.data_name = 'input_data'
        else:
            self.data_name = data_name
        
            
    def pre_process(self):
        #first split the data into training set and test set
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = self.training_size, random_state = 0)
        
        #Apply feature scale if requested by user
        if (self.feature_scaling):
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            self.X_train = sc.fit_transform(self.X_train)
            self.X_test = sc.transform(self.X_test)
            
    
    def logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0,solver = 'newton-cg')
        classifier.fit(self.X_train, self.y_train)

        # Predicting the Test set results
        self.y_pred_test = classifier.predict(self.X_test)
        self.y_pred_train = classifier.predict(self.X_train)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        self.cm_test  = confusion_matrix(self.y_test, self.y_pred_test)
        self.cm_train = confusion_matrix(self.y_train, self.y_pred_train)
        
        print (self.cm_test)
        print (self.cm_train)
    
    def decision_tree(self):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import confusion_matrix
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
    
        #Now get the default depth
        depth = classifier.get_depth()
        accuracy = []
    
        for i1 in range(3,depth+1):
            classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,max_depth = i1)
            classifier.fit(self.X_train,self.y_train)
            y_train_pred = classifier.predict(self.X_train)
            y_test_pred = classifier.predict(self.X_test)
            cm_train = confusion_matrix(self.y_train,y_train_pred)
            cm_test = confusion_matrix(self.y_test,y_test_pred)
            accuracy.append([i1,np.trace(cm_train)*100/len(self.y_train),np.trace(cm_test)*100/len(self.y_test)])
        
        self.line_plot(accuracy, ylabel ='% Accuracy', xlabel = 'No of branches', 
                       legends = ['Training_set','Test_Set'])
    
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
        
# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

#decision_tree_classification(X,y,'Diabetes')
#logistic_regression_classification(X,y)
cl = Classifier(X_data = X, Y_data = y, training_size = 0.25, feature_scaling = True)
cl.pre_process()
cl.decision_tree()