import matplotlib as mpl
mpl.use('Agg')
import seaborn as sn; sn.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import set_option

import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import linear_model
from sklearn.svm import SVC


"""
This experiment uses all ML algorithms to test classification using different coding schemes and 
different pre-process techniques, but using h-method and tuning one parameter of each algorithm
using as metric Accuracy
"""


def classification(filename, prepro, threads):
    set_option("display.max_rows", 10)
    pd.options.mode.chained_assignment = None

    schemes = ["complementary", "DAX", "EIIP", "enthalpy", "Galois4", "kmers", "pc"]
    # evaluate each model in turn
    for scheme in schemes:

        training_data = pd.read_csv(filename+'.'+scheme)
        print(training_data)

        # basic statistics
        training_data.describe()
        label_vectors = training_data['Label'].values
        feature_vectors = training_data.drop(['Label'], axis=1).values
        print(label_vectors)
        print(feature_vectors)

        x_data = []
        y_data = []

        if prepro == 1:
            x_data = feature_vectors
            y_data = label_vectors
            print("### Any")
        elif prepro == 2:
            # information scaling
            scaler = preprocessing.StandardScaler().fit(feature_vectors)
            feature_vectors_scaler = scaler.transform(feature_vectors)
            x_data = feature_vectors_scaler
            y_data = label_vectors
            print("### Scaling")
        elif prepro == 3:
            # PCA without scaling
            pca = decomposition.PCA(n_components=0.96,svd_solver='full',tol=1e-4)
            pca.fit(feature_vectors)
            X_trainPCA = pca.transform(feature_vectors)
            x_data = X_trainPCA
            y_data = label_vectors
            print("### PCA")
            print('X_PCA:', X_trainPCA.shape)
        elif prepro == 4:
            # information scaling
            scaler = preprocessing.StandardScaler().fit(feature_vectors)
            feature_vectors_scaler = scaler.transform(feature_vectors)

            # PCA with scaling
            pca = decomposition.PCA(n_components=0.9,svd_solver='full',tol=1e-4)
            pca.fit(feature_vectors_scaler)
            X_trainPCAScaler = pca.transform(feature_vectors_scaler)
            x_data = X_trainPCAScaler
            y_data = label_vectors
            print("### PCA + Scaling")
            print('X_PCA:',X_trainPCAScaler.shape)

        # information split with scaling
        validation_size = 0.2
        seed = 7
        X_train, X_validation, Y_train, Y_validation = train_test_split(x_data, y_data, test_size=validation_size, random_state=seed)

        # Logistic Regression
        # Testing different values of C
        limit = 1
        step = 0.1
        x = [0 for x in range(0,int(limit/step))]
        yValidation = [0 for x in range(0,int(limit/step))]
        ytrain = [0 for x in range(0,int(limit/step))]
        i = step
        index = 0
        while i < limit:
            lr = LogisticRegression(C=i, n_jobs=threads)

            lr.fit(X_train, Y_train)
            trainScore=lr.score(X_train,Y_train)
            validationScore=lr.score(X_validation,Y_validation)

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('ite:',i)

            x[index]=i
            i+=step
            index+=1

        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'o-',label='Train')
        plt.plot(x,yValidation,'v-',label='Validation')
        
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.title('C vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('LR-Algorithm_'+scheme+'.png', dpi=100)
        print('### '+scheme)
        print('### The best score with data validation: ', max(yValidation),'with C: ',x[yValidation.index(max(yValidation))])
        
        lr = LogisticRegression(C=x[yValidation.index(max(yValidation))])
        lr.fit(X_train, Y_train)
        predictions = lr.predict(X_validation)
        metrics(Y_validation,predictions)

        # LDA
        # Testing different values of tolerance
        limit=0.001
        step=0.0001
        x=[0 for x in range(0,int(limit/step))]
        yValidation=[0 for x in range(0,int(limit/step))]
        ytrain=[0 for x in range(0,int(limit/step))]
        i=step
        index=0
        while i<limit:
            LDA = LinearDiscriminantAnalysis(tol=i)

            LDA.fit(X_train, Y_train)

            trainScore=LDA.score(X_train,Y_train)
            validationScore=LDA.score(X_validation,Y_validation)

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('ite:',i)

            x[index]=i
            i+=step
            index+=1

        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'o-',label='Train')
        plt.plot(x,yValidation,'v-',label='Validation')
        
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.title('Tolerance vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('LDA-Algorithm_'+scheme+'.png', dpi=100)
        print('### '+scheme)
        print('### The best score with data validation: ', max(yValidation),'with tol: ',x[yValidation.index(max(yValidation))])
        
        LDA = LinearDiscriminantAnalysis(tol=x[yValidation.index(max(yValidation))])
        LDA.fit(X_train, Y_train)
        predictions = LDA.predict(X_validation)
        metrics(Y_validation,predictions)

        # KNN algorithm
        # Testing different quantities of neighbors
        limit=100
        x=[x for x in range(1,limit)]
        yValidation=[0 for x in range(1,limit)]
        ytrain=[0 for x in range(1,limit)]
        
        for i in range(1,limit):
            KNN = KNeighborsClassifier(n_neighbors=i, n_jobs=threads)

            KNN.fit(X_train, Y_train)
            trainScore=KNN.score(X_train,Y_train)
            validationScore=KNN.score(X_validation,Y_validation)

            print('KNN Score:',i)
            ytrain[i-1]=trainScore
            yValidation[i-1]=validationScore

        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,label='Train')
        plt.plot(x,yValidation,label='Validation')
        plt.xlabel('n-Neighbors')
        plt.ylabel('Accuracy')
        plt.title('Neighbors vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('KNN-Algorithm_'+scheme+'.png', dpi=100)

        print('### '+scheme)
        print('### The best score with data validation: ', max(yValidation),'with Neighbors: ',x[yValidation.index(max(yValidation))])
        
        KNN = KNeighborsClassifier(n_neighbors=x[yValidation.index(max(yValidation))])
        KNN.fit(X_train, Y_train)
        predictions = KNN.predict(X_validation)
        metrics(Y_validation,predictions)

        # SVC
        #Testing different values of C
        limit=100
        step=10
        x=[x for x in range(0,int(limit/step)-1)]
        yValidation=[0 for x in range(0,int(limit/step)-1)]
        ytrain=[0 for x in range(0,int(limit/step)-1)]
        
        i=step
        index=0
        while i<limit:
            svc = SVC(C=i, gamma=1e-6)

            svc.fit(X_train, Y_train)
            trainScore=svc.score(X_train,Y_train)
            validationScore=svc.score(X_validation,Y_validation)

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('ite:',i)

            x[index]=i
            i+=step
            index+=1

        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'o-',label='Train')
        plt.plot(x,yValidation,'v-',label='Validation')
        
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.title('C vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('SVC-Algorithm_'+scheme+'.png', dpi=100)
        plt.show()
        print('### '+scheme)
        print('### The best score with data validation: ', max(yValidation),'with C: ',x[yValidation.index(max(yValidation))])
        
        svc = SVC(C=x[yValidationScaler.index(max(yValidationScaler))],gamma=1e-6)
        svc.fit(X_trainScaler, Y_trainScaler)
        predictions = svc.predict(X_validationScaler)
        metrics(Y_validationScaler,predictions)

        # MLP
        limit=1050
        step=50
        x=[x for x in range(0,int(limit/step)-1)]
        yValidation=[0 for x in range(0,int(limit/step)-1)]
        ytrain=[0 for x in range(0,int(limit/step)-1)]
        
        i=step
        index=0
        while i<limit:
            MLP = MLPClassifier(solver='lbfgs', alpha=.5, hidden_layer_sizes=(i))
            MLP.fit(X_train, Y_train)
            trainScore=MLP.score(X_train,Y_train)
            validationScore=MLP.score(X_validation,Y_validation)

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('it:',i)
            x[index]=i
            i+=step
            index+=1

        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'o-',label='Train')
        plt.plot(x,yValidation,'v-',label='Validation')
        
        plt.xlabel('Neurons')
        plt.ylabel('Accuracy')
        plt.title('Neurons vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('MLP-Algorithm_'+scheme+'.png', dpi=100)
        plt.show()
        print('### '+scheme)
        print('### The best score with data validation: ', max(yValidation),'with Neurons: ',x[yValidation.index(max(yValidation))])
        MLP = MLPClassifier(solver='lbfgs', alpha=.5, hidden_layer_sizes=x[yValidation.index(max(yValidation))])
        MLP.fit(X_train, Y_train)
        predictions = MLP.predict(X_validation)
        metrics(Y_validation,predictions)

        # RF
        limit=100
        step=10
        x=[x for x in range(0,int(limit/step)-1)]
        yValidation=[0 for x in range(0,int(limit/step)-1)]
        ytrain=[0 for x in range(0,int(limit/step)-1)]
        
        i=step
        index=0
        while i<limit:
            RF = RandomForestClassifier(n_estimators=i, n_jobs=threads)
            RF.fit(X_train, Y_train)
            trainScore=RF.score(X_train,Y_train)
            validationScore=RF.score(X_validation,Y_validation)

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('n_estimators:',i)
            x[index]=i
            i+=step
            index+=1

        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'o-',label='Train')
        plt.plot(x,yValidation,'v-',label='Validation')
        
        plt.xlabel('Trees')
        plt.ylabel('Accuracy')
        plt.title('Trees vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('RF-Algorithm_'+scheme+'.png', dpi=100)
        plt.show()
        print('### '+scheme)
        print('### The best score with data validation: ', max(yValidation),'with n_estimators: ',x[yValidation.index(max(yValidation))])
        RF = RandomForestClassifier(n_estimators=x[yValidation.index(max(yValidation))])
        RF.fit(X_train, Y_train)
        predictions = RF.predict(X_validation)
        metrics(Y_validation,predictions)

        # DT
        limit=10
        step=1
        x=[x for x in range(0,int(limit/step)-1)]
        yValidation=[0 for x in range(0,int(limit/step)-1)]
        ytrain=[0 for x in range(0,int(limit/step)-1)]
        
        i=step
        index=0
        while i<limit:
            DT = DecisionTreeClassifier(max_depth=i)
            DT.fit(X_train, Y_train)
            trainScore=DT.score(X_train,Y_train)
            validationScore=DT.score(X_validation,Y_validation)

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('max_depth:',i)
            x[index]=i
            i+=step
            index+=1

        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'o-',label='Train')
        plt.plot(x,yValidation,'v-',label='Validation')
        
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.title('Max Depth vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('DT-Algorithm_'+scheme+'.png', dpi=100)
        plt.show()
        print('### '+scheme)
        print('### The best score with data validation: ', max(yValidation),'with max_depth: ',x[yValidation.index(max(yValidation))])
        DT = DecisionTreeClassifier(max_depth=x[yValidation.index(max(yValidation))])
        DT.fit(X_train, Y_train)
        predictions = DT.predict(X_validation)
        metrics(Y_validation,predictions)

        # Bayesian Classifier
        values=[1e-1,1e-3,1e-5,1e-7,1e-9,1e-11,1e-13,1e-15,1e-17,1e-19]
        x=[0 for x in range(0,len(values))]
        yValidation=[0 for x in range(0,len(values))]
        ytrain=[0 for x in range(0,len(values))]
        for i,index in zip(values,range(len(values))):
            NB = GaussianNB(var_smoothing=i)

            NB.fit(X_train,Y_train)
            trainScore=NB.score(X_train,Y_train)
            validationScore=NB.score(X_validation, Y_validation)

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('ite:',i)

            x[index]=i
    
        plt.close('all')
        fig=plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'-',label='Train')
        plt.plot(x,yValidation,'-',label='Validation')
        
        plt.xlabel('var_smoothing')
        plt.ylabel('Accuracy')
        plt.ylim((0,1.1))
        plt.title('C vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('NB-Algorithm_'+scheme+'.png', dpi=100)
        print('### '+scheme)
        print('### NB ### The best score with data validation: ', max(yValidation),'with var_smoothing: ',x[yValidation.index(max(yValidation))])
        
        NB = GaussianNB(var_smoothing=x[yValidation.index(max(yValidation))])
        NB.fit(X_train, Y_train)
        predictions = NB.predict(X_validation)
        metrics(Y_validation,predictions)
        #free memory
        NB = None


def metrics(Y_validation,predictions):
    print('Accuracy:', accuracy_score(Y_validation, predictions))
    print('F1 score:', f1_score(Y_validation, predictions,average='weighted'))
    print('Recall:', recall_score(Y_validation, predictions,average='weighted'))
    print('Precision:', precision_score(Y_validation, predictions, average='weighted'))
    print('\n clasification report:\n', classification_report(Y_validation, predictions))
    print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))
    # To create confusion matrix
    snn_cm = confusion_matrix(Y_validation, predictions)

    # Displaying the confusion matrix
    snn_df_cm = pd.DataFrame(snn_cm, range(12), range(12))
    plt.figure(figsize = (20,14))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 12}) # font size
    plt.show()

if __name__ == '__main__':

    fileData = sys.argv[1]
    threads = int(sys.argv[2])
    classification(fileData, 1, threads) # for no pre-procesing
    classification(fileData, 2, threads) # for Scaling
    classification(fileData, 3, threads) # for PCA
    classification(fileData, 4, threads) # for Scaling + PCA

