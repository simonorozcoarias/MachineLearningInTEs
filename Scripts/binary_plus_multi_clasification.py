#!/usr/bin/env python3

import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sn

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
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn import decomposition
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from mlxtend.classifier import StackingClassifier
import lightgbm as lgb

import random
seed = random.seed(7)
import time
Fstart_t = time.time()


def detection(filename, prepro, threads):
	set_option("display.max_rows", 10)
	pd.options.mode.chained_assignment = None

	# Cross Validation
	seed = 42
	scoring = 'accuracy'

	models = []
	schemes = ["complementary", "DAX", "EIIP", "enthalpy", "Galois4", "pc","kmers"]
	# evaluate each model in turn
	results = []
	names = []
	for scheme in schemes:

		training_data = pd.read_csv(filename+'.'+scheme, index_col=False)
		print(training_data)

		label_vectors =training_data['Label'].values
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
			x_data = scaler.transform(feature_vectors)
			y_data = label_vectors
			print("### Scaling")
		elif prepro == 3:
			# PCA without scaling
			pca = decomposition.PCA(n_components=0.96,svd_solver='full',tol=1e-4)
			pca.fit(feature_vectors)
			x_data = pca.transform(feature_vectors)
			y_data = label_vectors

			#releasing memory
			pca = None
			label_vectors = None
			print("### PCA")
			print('X_PCA:', x_data.shape)
		elif prepro == 4:
			# information scaling
			scaler = preprocessing.StandardScaler().fit(feature_vectors)
			feature_vectors_scaler = scaler.transform(feature_vectors)
            
			# PCA with scaling
			pca = decomposition.PCA(n_components=0.9,svd_solver='full',tol=1e-4)
			pca.fit(feature_vectors_scaler)
			x_data = pca.transform(feature_vectors_scaler)
			y_data = label_vectors

			# realeasing memory
			feature_vectors_scaler = None
			pca = None
			scaler = None
			print("### PCA + Scaling")
			print('X_PCA:',x_data.shape)

		# information split with scaling
		validation_size = 0.2
		seed = 7
		X_train, X_val_test, Y_train, Y_val_test = train_test_split(x_data, list(y_data), test_size=validation_size, random_state=seed)
		X_validation, X_test, Y_validation, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5, random_state=seed)
		x_data = None
		y_data = None
		training_data = None
		label_vectors = None
		feature_vectors = None

		######################### Logistic Regression
		# Testing different values of C
		limit=1
		step=0.1
		x=[0 for x in range(0,int(limit/step))]
		yValidation=[0 for x in range(0,int(limit/step))]
		ytrain=[0 for x in range(0,int(limit/step))]
		i=step
		index=0
		while i<limit:
			lr = LogisticRegression(C=i, n_jobs=threads)
			lr.fit(X_train,Y_train)
			trainScore=f1_score(Y_train, lr.predict(X_train), average='macro')
			validationScore=f1_score(Y_validation, lr.predict(X_validation), average='macro')
			ytrain[index]=trainScore
			yValidation[index]=validationScore
			print('ite:',i)
			x[index]=i
			i+=step
			index+=1
	
		plt.close('all')
		fig=plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
		plt.plot(x,ytrain,'-',label='Train')
		plt.plot(x,yValidation,'-',label='Validation')
        
		plt.xlabel('C')
		plt.ylabel('F1-Score')
		plt.ylim((0,1.1))
		plt.legend()
		plt.savefig('LR-Algorithm_'+scheme+'.png', dpi=100)
		print('### '+scheme)
		print('### LG ### The best score with data validation: ', max(yValidation),'with C: ',x[yValidation.index(max(yValidation))])
        
		lr = LogisticRegression(C=x[yValidation.index(max(yValidation))])
		lr.fit(X_train, Y_train)
		predictions = lr.predict(X_test)
		metrics(Y_test,predictions)
		#free memory
		lr = None

		######################### LDA
		# Testing different values of tolerance
		limit=0.001
		step=0.0001
		x=[0 for x in range(0,int(limit/step))]
		yValidation=[0 for x in range(0,int(limit/step))]
		ytrain=[0 for x in range(0,int(limit/step))]
		i=step
		index=0
		while i<=limit:
			LDA = LinearDiscriminantAnalysis(tol=i)
			LDA.fit(X_train, Y_train)
			trainScore=f1_score(Y_train, LDA.predict(X_train), average='macro')
			validationScore=f1_score(Y_validation, LDA.predict(X_validation), average='macro')
			ytrain[index]=trainScore
			yValidation[index]=validationScore
			print('ite:',i)
			x[index]=i
			i=round(i+step,4)
			index+=1
		plt.close('all')
		fig=plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
		plt.plot(x,ytrain,'-',label='Train')
		plt.plot(x,yValidation,'-',label='Validation')
		plt.ylim((0,1.1))
		plt.xlabel('C')
		plt.ylabel('F1-Score')
		plt.legend()
		plt.savefig('LDA-Algorithm_'+scheme+'.png', dpi=100)
		print('### '+scheme)
		print('### LDA ### The best score with data validation: ', max(yValidation),'with tol: ',x[yValidation.index(max(yValidation))])
        
		LDA = LinearDiscriminantAnalysis(tol=x[yValidation.index(max(yValidation))])
		LDA.fit(X_train, Y_train)
		predictions = LDA.predict(X_test)
		metrics(Y_test,predictions)
		#free memory
		LDA = None

		######################### KNN algorithm
		# Testing different quantities of neighbors
		limit=100
		x=[x for x in range(1,limit, 10)]
		yValidation=[0 for x in range(1,limit, 10)]
		ytrain=[0 for x in range(1,limit, 10)]
		index = 0
		for i in range(1,limit, 10):
			KNN = KNeighborsClassifier(n_neighbors=i, n_jobs=threads)
			KNN.fit(X_train, Y_train)
			trainScore=f1_score(Y_train, KNN.predict(X_train), average='macro')
			validationScore=f1_score(Y_validation, KNN.predict(X_validation), average='macro')
			print('KNN Score:',i)
			ytrain[index]=trainScore
			yValidation[index]=validationScore
			index += 1
		plt.close('all')
		fig=plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
		plt.plot(x,ytrain,label='Train')
		plt.ylim((0,1.1))
		plt.plot(x,yValidation,label='Validation')
		plt.xlabel('n-Neighbors')
		plt.ylabel('F1-Score')
		plt.legend()
		plt.savefig('KNN-Algorithm_'+scheme+'.png', dpi=100)
		print('### '+scheme)
		print('### KNN ### The best score with data validation: ', max(yValidation),'with Neighbors: ',x[yValidation.index(max(yValidation))])

		KNN = KNeighborsClassifier(n_neighbors=x[yValidation.index(max(yValidation))])
		KNN.fit(X_train, Y_train)
		predictions = KNN.predict(X_test)
		metrics(Y_test,predictions)
		#free memory
		KNN = None
              
		######################### MLP
		limit=500
		step=50
		x=[x for x in range(0,int(limit/step)-1)]
		yValidation=[0 for x in range(0,int(limit/step)-1)]
		ytrain=[0 for x in range(0,int(limit/step)-1)]
        
		i=step
		index=0
		while i<limit:
			MLP = MLPClassifier(solver='lbfgs', alpha=.5, hidden_layer_sizes=(i))
			MLP.fit(X_train, Y_train)
			trainScore=f1_score(Y_train, MLP.predict(X_train), average='macro')
			validationScore=f1_score(Y_validation, MLP.predict(X_validation), average='macro')
			ytrain[index]=trainScore
			yValidation[index]=validationScore
			print('it:',i)
			x[index]=i
			i+=step
			index+=1
	
		plt.close('all')
		fig=plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
		plt.plot(x,ytrain,'-',label='Train')
		plt.plot(x,yValidation,'-',label='Validation')
		plt.ylim((0,1.1))
		plt.xlabel('Neurons')
		plt.ylabel('F1-Score')
		plt.legend()
		plt.savefig('MLP-Algorithm_'+scheme+'.png', dpi=100)
		print('### '+scheme)
		print('### MLP ### The best score with data validation: ', max(yValidation),'with Neurons: ',x[yValidation.index(max(yValidation))])
		MLP = MLPClassifier(solver='lbfgs', alpha=.5, hidden_layer_sizes=x[yValidation.index(max(yValidation))])
		MLP.fit(X_train, Y_train)
		predictions = MLP.predict(X_test)
		metrics(Y_test,predictions)
		#free memory
		MLP = None
        
		######################### RF
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
			trainScore=f1_score(Y_train, RF.predict(X_train), average='macro')
			validationScore=f1_score(Y_validation, RF.predict(X_validation), average='macro')
			ytrain[index]=trainScore
			yValidation[index]=validationScore
			print('n_estimators:',i)
			x[index]=i
			i+=step
			index+=1
            
		plt.close('all')
		plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
		plt.plot(x,ytrain,'-',label='Train')
		plt.plot(x,yValidation,'-',label='Validation')
		plt.ylim((0,1.1))
		plt.xlabel('Trees')
		plt.ylabel('F1-Score')
		plt.legend()
		plt.savefig('RF-Algorithm_'+scheme+'.png', dpi=100)
		print('### '+scheme)
		print('### RF ### The best score with data validation: ', max(yValidation),'with n_estimators: ',x[yValidation.index(max(yValidation))])
		RF = RandomForestClassifier(n_estimators=x[yValidation.index(max(yValidation))])
		RF.fit(X_train, Y_train)
		predictions = RF.predict(X_test)
		metrics(Y_test,predictions)
		#free memory
		RF = None

		######################### DT
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
			trainScore=f1_score(Y_train, DT.predict(X_train), average='macro')
			validationScore=f1_score(Y_validation, DT.predict(X_validation), average='macro')
			ytrain[index]=trainScore
			yValidation[index]=validationScore
			print('max_depth:',i)
			x[index]=i
			i+=step
			index+=1
		plt.close('all')
		plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
		plt.plot(x,ytrain,'-',label='Train')
		plt.plot(x,yValidation,'-',label='Validation')
		plt.ylim((0,1.1))
		plt.xlabel('Max Depth')
		plt.ylabel('F1-Score')
		#plt.title('Max Depth vs Accuracy ('+scheme+')')
		plt.legend()
		plt.savefig('DT-Algorithm_'+scheme+'.png', dpi=100)
		#plt.show()
		print('### '+scheme)
		print('### DT ### The best score with data validation: ', max(yValidation),'with max_depth: ',x[yValidation.index(max(yValidation))])
		DT = DecisionTreeClassifier(max_depth=x[yValidation.index(max(yValidation))])
		DT.fit(X_train, Y_train)
		predictions = DT.predict(X_test)
		metrics(Y_test,predictions)
		#free memory
		DT = None

		######################### Bayesian Classifier
		limit=1e-15
		step=1e-2
		values=[1e-1,1e-3,1e-5,1e-7,1e-9,1e-11,1e-13,1e-15,1e-17,1e-19]
		x=[0 for x in range(0,len(values))]
		yValidation=[0 for x in range(0,len(values))]
		ytrain=[0 for x in range(0,len(values))]
		for i,index in zip(values,range(len(values))):
			NB = GaussianNB(var_smoothing=i)

			NB.fit(X_train,Y_train)
			trainScore=f1_score(Y_train, NB.predict(X_train), average='macro')
			validationScore=f1_score(Y_validation, NB.predict(X_validation), average='macro')

			ytrain[index]=trainScore
			yValidation[index]=validationScore

			print('ite:',i)

			x[index]=i
	
		plt.close('all')
		fig=plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
		plt.plot(x,ytrain,'-',label='Train')
		plt.plot(x,yValidation,'-',label='Validation')
        
		plt.xlabel('var_smoothing')
		plt.ylabel('F1-Score')
		plt.ylim((0,1.1))
		plt.legend()
		plt.savefig('NB-Algorithm_'+scheme+'.png', dpi=100)
		print('### '+scheme)
		print('### NB ### The best score with data validation: ', max(yValidation),'with var_smoothing: ',x[yValidation.index(max(yValidation))])
        
		NB = GaussianNB(var_smoothing=x[yValidation.index(max(yValidation))])
		NB.fit(X_train, Y_train)
		predictions = NB.predict(X_test)
		metrics(Y_test,predictions)
		#free memory
		NB = None

		######################### SVC
		# Testing different values of C
		limit=100
		step=10
		x=[x for x in range(0,int(limit/step)-1)]
		yValidation=[0 for x in range(0,int(limit/step)-1)]
		ytrain=[0 for x in range(0,int(limit/step)-1)]
        
		i=step
		index=0
		while i<limit:
			svc = OneVsRestClassifier(SVC(C=i, gamma=1e-6), n_jobs = threads)

			svc.fit(X_train, Y_train)
			trainScore=f1_score(Y_train, svc.predict(X_train), average='macro')
			validationScore=f1_score(Y_validation, svc.predict(X_validation), average='macro')

			ytrain[index]=trainScore
			yValidation[index]=validationScore

			print('ite:',i)

			x[index]=i
			i+=step
			index+=1
	
		plt.close('all')
		plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
		plt.plot(x,ytrain,'-',label='Train')
		plt.plot(x,yValidation,'-',label='Validation')
		plt.ylim((0,1.1))
		plt.xlabel('C')
		plt.ylabel('F1-Score')
		plt.legend()
		plt.savefig('SVC-Algorithm_'+scheme+'.png', dpi=100)
		print('### '+scheme)
		print('### SVM ### The best score with data validation: ', max(yValidation),'with C: ',x[yValidation.index(max(yValidation))])
        
		svc = OneVsRestClassifier(SVC(C=x[yValidation.index(max(yValidation))],gamma=1e-6), n_jobs = threads) 
		svc.fit(X_train, Y_train)
		predictions = svc.predict(X_test)
		metrics(Y_test,predictions)
		#releasing memory
		svc = None


def metrics(Y_validation,predictions):
	print('Accuracy:', accuracy_score(Y_validation, predictions))
	print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
	print('Recall:', recall_score(Y_validation, predictions, average='macro'))
	print('Precision:', precision_score(Y_validation, predictions, average='macro'))
	print('\n clasification report:\n', classification_report(Y_validation, predictions))
	print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))
	snn_cm = confusion_matrix(Y_validation, predictions)
	return snn_cm


"""
Used to plot the training curves
"""
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt    


def classification_tuning(filename, n_jobs, opc):

	set_option("display.max_rows", 10)
	pd.options.mode.chained_assignment = None
	scoring = 'f1_weighted'
	models = []
	results = []
	names = []

	# ### to load teh dataset
	training_data = pd.read_csv(filename)
	label_vectors = training_data['Label'].values
	feature_vectors = training_data.drop(['Label'],axis=1)
	x_data = []
	y_data = []


	#### To apply the defined pre-processing Scaling + PCA

	# information scaling
	scaler = StandardScaler().fit(feature_vectors)
	feature_vectors_scaler = scaler.transform(feature_vectors)

	# PCA with scaling
	pca = PCA(n_components=0.9,svd_solver='full',tol=1e-4)
	pca.fit(feature_vectors_scaler)
	X_trainPCAScaler = pca.transform(feature_vectors_scaler)
	x_data = X_trainPCAScaler
	y_data = label_vectors


	#### Split the dataset

	training_size = .90
	X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, train_size=training_size, random_state=seed)

	print("Y_train length = ", Y_train.shape[0])
	print("Y_test length = ", Y_test.shape[0])

	new_perc = (Y_test.shape[0]*100)/Y_train.shape[0]
	print("New percentage is= {:.2f}%".format(new_perc))
	# 1/CV is the percentage if data to test the model with cross validation, 
	CV = round(1/(new_perc/100))
	print("K-fold is equal to ",CV)

	Xtraint, Xval, Ytraint, Yval = train_test_split(X_train,Y_train, test_size=new_perc/100, random_state=seed)
	# ### To select which ML model will be tuned

	if opc == 1:
	    estimator = LogisticRegression(C=0.01, tol=10, max_iter=1000, penalty='l2',solver='sag')
	    text_model = "LR"
	if opc == 2:
	    estimator = LinearDiscriminantAnalysis(shrinkage=0.0001,solver='lsqr',tol=0.1)
	    text_model = "LDA"
	if opc == 3:
	    estimator = KNeighborsClassifier(algorithm='auto', n_neighbors=2, weights='distance', metric = 'euclidean')
	    text_model = "KNN"
	if opc == 4:
		estimator = LinearSVC(C= 0.001, loss= 'squared_hinge', penalty= 'l2', tol= 0.1)
		text_model = "LinearSVC"

	tunnedModel = estimator

	#### To plot the curves

	Cros_val = cross_val_score(tunnedModel,X_train,Y_train,cv=CV,scoring=scoring, n_jobs=n_jobs)
	print("Cross validation score: ",np.asarray(Cros_val).mean())

	tunnedModel.fit(Xtraint,Ytraint)
	prediction = tunnedModel.predict(X_test)

	metrics(Y_test,prediction)

	fig, axes = plt.subplots(3, 1, figsize=(10, 15))
	title = "Learning Curves (tunned "+str(text_model)+")"

	plot_learning_curve(tunnedModel, title, X_train, Y_train, cv=CV, n_jobs=n_jobs)
	plt.savefig(str(text_model)+"_Tunned_Algorithm.png", dpi=100)
	Fend_t = time.time()
	print("Full Script Tunning time",Fend_t-Fstart_t)


def ensemble_method(filename, n_jobs, ):
	set_option("display.max_rows", 10)
	pd.options.mode.chained_assignment = None
	scoring = 'f1_weighted'
	models = []
	results = []
	names = []

	#### To load the dataset

	training_data = pd.read_csv(filename)
	label_vectors = training_data['Label'].values
	feature_vectors = training_data.drop(['Label'],axis=1)
	x_data = []
	y_data = []


	#### To apply the defined pre-processing Scaling + PCA

	# information scaling
	scaler = StandardScaler().fit(feature_vectors)
	feature_vectors_scaler = scaler.transform(feature_vectors)

	# PCA with scaling
	pca = PCA(n_components=0.9,svd_solver='full',tol=1e-4)
	pca.fit(feature_vectors_scaler)
	X_trainPCAScaler = pca.transform(feature_vectors_scaler)
	x_data = X_trainPCAScaler
	y_data = label_vectors


	#### Split the dataset

	training_size = .90
	X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, train_size=training_size, random_state=seed)

	print("Y_train length = ", Y_train.shape[0])
	print("Y_test length = ", Y_test.shape[0])

	new_perc = (Y_test.shape[0]*100)/Y_train.shape[0]
	print("New percentage is= {:.2f}%".format(new_perc))
	# 1/CV is the percentage if data to test the model with cross validation, 
	CV = round(1/(new_perc/100))
	print("K-fold is equal to ",CV)

	Xtraint, Xval, Ytraint, Yval = train_test_split(X_train,Y_train, test_size=new_perc/100, random_state=seed)


	#### Ensamble method

	clf1 = KNeighborsClassifier(algorithm='auto', n_neighbors=2, weights='distance', metric = 'euclidean')
	clf2 = LogisticRegression(C=0.01, tol=10, max_iter=1000, penalty='l2',solver='sag')
	clf3 = LinearDiscriminantAnalysis(shrinkage=0.0001,solver='lsqr',tol=0.1)

	Rf = RandomForestClassifier()
	sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=Rf)
	text_model = "Stacking"

	label = ['KNN','LR', 'LDA','Stacking Classifier']
	clf_list = [clf1, clf2, clf3, sclf]

	clf_cv_mean = []
	clf_cv_std = []
	for clf, label in zip(clf_list, label):
	        
	    scores = cross_val_score(clf, X_train,Y_train, cv=CV, scoring=scoring, n_jobs=n_jobs)
	    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
	    clf_cv_mean.append(scores.mean())
	    clf_cv_std.append(scores.std()) 


	##### Final Model evaluation 

	Cros_val = cross_val_score(sclf,X_train,Y_train,cv=CV,scoring=scoring, n_jobs=n_jobs)
	print("Cross validation score: ",np.asarray(Cros_val).mean())

	sclf.fit(Xtraint, Ytraint)
	prediction = sclf.predict(X_test)

	snn_cm = metrics(Y_test,prediction)
	print("#####################################")

	fig, axes = plt.subplots(3, 1, figsize=(10, 15))
	title = "Learning Curves (tunned "+str(text_model)+")"

	plot_learning_curve(sclf, title, X_train, Y_train,
	                    cv=CV, n_jobs=n_jobs)
	#plt.show()
	plt.savefig(str(text_model)+"_Tunned_Algorithm.png", dpi=300)
	plt.savefig(str(text_model)+"_Tunned_Algorithm.svg", format = "svg", dpi=1200)
	plt.savefig(str(text_model)+"_Tunned_Algorithm.pdf", dpi=1200)


	#To plot confusion matrix
	snn_df_cm = pd.DataFrame(snn_cm)  
	plt.figure(figsize = (20,14))  
	sn.set(font_scale=1.4) #for label size  
	sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 25},cmap="YlOrRd") # font size
	plt.savefig(str(text_model)+"_Confussion_matrix.png", dpi=300)
	plt.savefig(str(text_model)+"_Confussion_matrix.svg",format = "svg", dpi=1200)
	plt.savefig(str(text_model)+"_Confussion_matrix.pdf", dpi=1200)

	Fend_t = time.time()
	print("Full Script Tunning time",Fend_t-Fstart_t)


def feature_selection(filename, n_jobs):
	set_option("display.max_rows", 10)
	pd.options.mode.chained_assignment = None
	scoring = 'f1_weighted'
	models = []
	results = []
	names = []

	# ### to load teh dataset
	training_data = pd.read_csv(filename)
	label_vectors = training_data['Label'].values
	feature_vectors = training_data.drop(['Label'],axis=1)
	x_data = []
	y_data = []


	#### To apply the defined pre-processing Scaling + PCA

	# information scaling
	scaler = StandardScaler().fit(feature_vectors)
	x_data = scaler.transform(feature_vectors)
	y_data = label_vectors


	#### Split the dataset

	training_size = .90
	X_train, X_test_dev, Y_train, Y_test_dev = train_test_split(x_data, y_data, train_size=training_size, random_state=seed)
	X_dev, X_test, Y_dev, Y_test = train_test_split(x_data, y_data, train_size=training_size, random_state=seed)

	print("Y_train length = ", Y_train.shape[0])
	print("Y_test length = ", Y_test.shape[0])

	# Initialize an empty array to hold feature importances
	feature_importances = np.zeros(X_train.shape[1])

	# Create the model with several hyperparameters
	model = lgb.LGBMClassifier(objective='multiclass', boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')

	# Fit the model    
	# Train using early stopping
	model.fit(X_train, Y_train, early_stopping_rounds=100, eval_set = [(X_dev, Y_dev)], eval_metric = 'multi_logloss', verbose = 1)

	# Record the feature importances
	feature_importances += model.feature_importances_

	print("model evaluation: "+str(model.score(X_test, Y_test)))

	# Make sure to average feature importances! 
	feature_importances = feature_importances / 2
	feature_importances = pd.DataFrame({'feature': list(training_data.drop(['Label'], axis=1).columns), 'importance': feature_importances}).sort_values('importance', ascending = False)

	print(feature_importances[0:10])

	i = 0
	importances = []
	features = []
	for imp in feature_importances['importance']:
	    if imp > 0:
	        print(list(feature_importances['feature'])[i]+" "+str(imp))
	        importances.append(imp)
	        features.append(list(feature_importances['feature'])[i])
	    i += 1
	    if len(importances) == 508:
	        break
	    
	with plt.style.context('seaborn-white'):
	    plt.figure(figsize=(10, 10))
	    #Plot training & validation loss values
	    plt.plot([x for x in range(len(importances))], importances)
	    plt.title('Importance Vs features')
	    plt.ylabel('Importance')
	    plt.xlabel('Feature')
	    plt.grid('on')
	    plt.savefig('importances.png', format='png')
	    plt.savefig('importances.svg', format='svg')
	    plt.savefig('importances.pdf', format='pdf') 

	    # to show selected features:
	    print("##################### selected features #####################")
	    print(','.join(features))

	num_fea_k = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
	for f in features:
		l = len(f)
		num_fea_k[l] += 1

	print(num_fea_k)



if __name__ == '__main__':

	###### JUST MODIFY THESE VARIABLES.  ####################################################
	# Before run this step, you must convert the sequences to the coding schemes implemented in: https://github.com/simonorozcoarias/MachineLearningInTEs/blob/master/Scripts/formatDB_final.py
	fileData = "/home/bioml/Projects/PhD/InpactorDB/InpactorDB_non-redundant+negative.fasta"
	threads=20
	pre_proc=1
	#preprocessing 
	#1-None
	#2-Scaling
	#3-PCA
	#4-PCA+Scaling
	detection(fileData, pre_proc, threads)
	##########################################################################################

	###### JUST MODIFY THESE VARIABLES.  #####################################################
	filename = "/home/bioml/Projects/PhD/InpactorDB/InpactorDB_non-redundant+negative.fasta.kmers"
	n_jobs = 64
	algorithm = 1
	#algorithms
	# 1: LR
	# 2: LDA
	# 3: KNN
	# 4: Linear SVC
	classification_tuning(filename, n_jobs, algorithm)
	##########################################################################################

	####### JUST MODIFY THESE VARIABLES ######################################################
	filename = "/home/bioml/Projects/PhD/InpactorDB/InpactorDB_non-redundant+negative.fasta.kmers"
	n_jobs = 64
	ensemble_method(filename, n_jobs)
	########################################################################################## 

	####### JUST MODIFY THESE VARIABLES ######################################################
	filename = "/home/bioml/Projects/PhD/InpactorDB/InpactorDB_non-redundant+negative.fasta.kmers"
	n_jobs = 64
	feature_selection(filename, n_jobs)
	########################################################################################## 