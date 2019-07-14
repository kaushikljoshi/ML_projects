# Decision Tree Classification  

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def line_plot(data,fig_name = None ,xlabel = None, ylabel = None,legends = None):
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
    
    if fig_name is None:
        plt.savefig(xlabel + '_'+ ylabel +'.png')
    else:
        plt.savefig(fig_name+'.png')

def decision_tree_classification(X,y,dataname):
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    #
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    
    #Now get the default depth
    depth = classifier.get_depth()
    accuracy = []
    
    for i1 in range(3,depth+1):
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,max_depth = i1)
        classifier.fit(X_train,y_train)
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        cm_train = confusion_matrix(y_train,y_train_pred)
        cm_test = confusion_matrix(y_test,y_test_pred)
        accuracy.append([i1,np.trace(cm_train)*100/len(y_train),np.trace(cm_test)*100/len(y_test)])
    
    line_plot(accuracy, ylabel ='% Accuracy', xlabel = 'No of branches', legends = ['Training_set','Test_Set'],fig_name = dataname)
    #line_plot([row[0] for row in accuracy], [row[1] for row in accuracy], [row[2] for row in accuracy] ylabel ='accuracy', xlabel = 'No of branches')


# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

decision_tree_classification(X,y,'Diabetes')
