
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import model_selection
from toolbox_02450 import rlr_validate

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)



#############################################
####             Functions               ####
#############################################


# Season categorizing (winter=1, spring=2, summer=3, fall=4)
def get_season(day):
        if day >= 355 or day < 81:
            return 1
        elif day < 173:
            return 2
        elif day < 266:
            return 3
        else:
            return 4

# Loading data in and creating X, Y and class_labels 
def dataprep ():
    # Loading file into str. array
    df = df = pd.read_csv("LA_Ozone_Data.txt", header=None)

    # Converting the panda dataframe to a numpy array: 
    raw_data = df.values
    data_values = raw_data[1:,:].astype(float)
    labels_raw = raw_data[0,:]

    #Subtrackting doy and ozone from the dataset, wich will be added later:
    X_raw = np.delete(data_values, np.s_[0,9], 1)
    class_labels = np.delete(labels_raw, np.s_[0,9])

    #Standardizing the X dataset:
    standard_X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)

    # Categorizing the datapoints with season and adding it to the array
    Seasons = np.array([get_season(day) for day in data_values[0:, 9]], copy=False, subok=True, ndmin=2).T
    X = np.append(standard_X, Seasons, axis=1)

    #Extending class_labels with seasons:
    label = list(class_labels)
    label.append('seasons')
    class_labels = np.asarray(label)

    # Creating Y column
    Y_raw = data_values[:, 0]
    Y = (Y_raw - np.mean(Y_raw))/np.std(Y_raw)

    # .to_numpy()

    print("Data loaded succesfully. \n")
    return X, Y, class_labels

# Linear regression model
def linear_regression(X, y, class_labels, K, K_inner, lambdas):
    
    # Applying method from script 8.1.1
    # Adding offset attribute to X
    X_off = np.concatenate((np.ones((X.shape[0],1)),X),1)
    attributeNames = ['offset'] + list(class_labels)
    M_off = M+1

    # Create crossvalidation partition for evaluation
    CV = model_selection.KFold(K, shuffle=True)

    # Initialize variables
    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))
    Error_train_rlr = np.empty((K,1))
    Error_test_rlr = np.empty((K,1))
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))
    w_rlr = np.empty((M_off,K))
    mu = np.empty((K, M))
    sigma = np.empty((K, M))
    w_noreg = np.empty((M_off,K))

    i=0
    for train_index, test_index in CV.split(X,y):
    
        # extract training and test set for current CV fold
        X_train = X_off[train_index]
        y_train = y[train_index]
        X_test = X_off[test_index]
        y_test = y[test_index]


        ###### Inner fold, validation
        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, K_inner)
        ######

        mu[i, :] = np.mean(X_train[:, 1:], 0)
        sigma[i, :] = np.std(X_train[:, 1:], 0)
        
        X_train[:, 1:] = (X_train[:, 1:] - mu[i, :] ) / sigma[i, :] 
        X_test[:, 1:] = (X_test[:, 1:] - mu[i, :] ) / sigma[i, :] 
        
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        
        # Compute mean squared error without using the input data at all
        Error_train_nofeatures[i] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
        Error_test_nofeatures[i] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M_off)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w_rlr[:,i] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[i] = np.square(y_train-X_train @ w_rlr[:,i]).sum(axis=0)/y_train.shape[0]
        Error_test_rlr[i] = np.square(y_test-X_test @ w_rlr[:,i]).sum(axis=0)/y_test.shape[0]

        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:,i] = np.linalg.solve(XtX,Xty).squeeze()
        # Compute mean squared error without regularization
        Error_train[i] = np.square(y_train-X_train @ w_noreg[:,i]).sum(axis=0)/y_train.shape[0]
        Error_test[i] = np.square(y_test-X_test @ w_noreg[:,i]).sum(axis=0)/y_test.shape[0]

        # Display the results for the last cross-validation fold
        if i == K-1:
            figure(figsize=(12,8))
            subplot(1,2,1)
            semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-')
            xlabel('Regularization factor')
            ylabel('Mean Coefficient Values')
            grid()
            legend(attributeNames[1:], loc='best')

            
            subplot(1,2,2)
            title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
            loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
            xlabel('Regularization factor')
            ylabel('Squared error (crossvalidation)')
            legend(['Train error','Validation error'])
            grid()
        
        i+=1
    
    show()

    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    # Display results
    # print('Linear regression without feature selection:')
    # print('- Training error: {0}'.format(Error_train.mean()))
    # print('- Test error:     {0}'.format(Error_test.mean()))
    # print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    # print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    # print('Regularized linear regression:')
    # print('- Training error: {0}'.format(Error_train_rlr.mean()))
    # print('- Test error:     {0}'.format(Error_test_rlr.mean()))
    # print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
    # print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

    # print('Weights in last fold:')
    # for m in range(M):
    #     print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))



#############################################
####             Main code               ####
#############################################

#extracting base data
X, y, class_labels = dataprep()
N, M = X.shape   

print(class_labels)

# Linear regression model, part A:
K = 10
K_inner = K
lambdas = np.power(10.,range(-2,9))
linear_regression(X, y, class_labels, K, K_inner, lambdas)

    

# %%
