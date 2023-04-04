
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from toolbox_02450 import rlr_validate
from matplotlib import pyplot as plt
from scipy.linalg import svd


from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)



#############################################
####             Functions               ####
#############################################


# Season categorizing (winter=1, spring=2, summer=3, fall=4)
def get_season(day):

    if day <= 59 or day > 334:
        return 'winter'
    elif day <= 151:
        return 'spring'
    elif day <= 243:
        return 'summer'
    else:
        return 'fall'
 
# Loading data in and creating X, Y and class_labels 
def dataprep ():
    # Loading file into str. array
    df = df = pd.read_csv("LA_Ozone_Data.txt", header=None)

    # Converting the panda dataframe to a numpy array: 
    raw_data = df.values
    data_values = raw_data[1:,:].astype(float)
    labels_raw = raw_data[0,:]

    # Subtrackting unsignificant attributes '[2,3,6,8]' for ozone prediction,
    # and further subtracting doy'9' and ozone'0' from the dataset, wich will be added later:
    X_raw = np.delete(data_values, np.s_[0,2,3,6,8,9], 1)
    attributes = np.delete(labels_raw, np.s_[0,2,3,6,8,9])

    #Standardizing the X dataset:
    standard_X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)

    # One-of-k encoding with season and adding it to the X array
    seasons = np.array([get_season(day) for day in data_values[0:, 9]])
    encoder = OneHotEncoder(sparse=False)
    one_k = encoder.fit_transform(seasons.reshape(-1, 1))
    X = np.append(standard_X, one_k, axis=1)
    # exporting the season category array
    season_categories = encoder.categories_[0]

    # Seasons = np.array([get_season(day) for day in data_values[0:, 9]], copy=False, subok=True, ndmin=2).T
    

    #Extending attributes with seasons:
    label = list(attributes)
    label = np.concatenate((label, season_categories))
    attributes = np.asarray(label)

    # Creating Y column
    Y_raw = data_values[:, 0]
    Y = (Y_raw - np.mean(Y_raw))/np.std(Y_raw)

    # .to_numpy()

    print("Data loaded succesfully. \n")
    return X, Y, attributes, data_values, one_k

# Re-computing PCA for 'Season' influence:
def PCA(data_values, one_k):
    
    # Subtracting doy fromthe dataset, to reduce noise:
    X_pca = np.delete(data_values, np.s_[9], 1)

    # Normalizing the data:
    N = len(X_pca)
    Y = (X_pca - X_pca.mean(axis=0)*np.ones((N,1))) / X_pca.std(axis=0)*np.ones((N,1))

    # Running the Single Value Decompositioning (SVD)
    U, S, V = svd(Y,full_matrices=False)

    fig1 = plt.figure(figsize=(10,8), facecolor='w')
    ax = fig1.add_subplot(projection='3d')

    # Computing the dot product 
    for i in range(Y.shape[0]):
        x = V[0,:] @ Y[i,:].T
        y = V[1,:] @ Y[i,:].T
        z = V[2,:] @ Y[i,:].T

        if one_k[i, 3] == 1:
            ax.scatter(x,y,z, marker='o', color='gold', s=20)
        elif one_k[i, 1] == 1:
            ax.scatter(x,y,z, marker='o', color='dodgerblue', s=20)
        elif one_k[i, 2] == 1:
            ax.scatter(x,y,z, marker='o', color='limegreen', s=20)
        else:
            ax.scatter(x,y,z, marker='o', color='darkorange', s=20)

    ax.set_xlabel('\nPC1', fontsize = 15, linespacing=1)
    ax.set_ylabel('\nPC2', fontsize = 15, linespacing=2)
    ax.set_zlabel('\nPC3 ', fontsize = 15, linespacing=2)
    ax.view_init(25,-50)
    # creating dummy plot for legend applyance: 
    proxy1 = plt.Line2D([0],[0], linestyle="none", color='gold', marker = 'o')
    proxy2 = plt.Line2D([0],[0], linestyle="none", color='dodgerblue', marker = 'o')
    proxy3 = plt.Line2D([0],[0], linestyle="none", color='limegreen', marker = 'o')
    proxy4 = plt.Line2D([0],[0], linestyle="none", color='darkorange', marker = 'o')
    ax.legend([proxy1, proxy2, proxy3, proxy4], ['Winter', 'Spring', 'Summer', 'Fall'], numpoints = 1)

    plt.show()

# Linear regression model
def linear_regression(X, y, attributes, K, K_inner, lambdas):
    
    # Applying method from script 8.1.1
    # Adding offset attribute to X
    X_off = np.concatenate((np.ones((X.shape[0],1)),X),1)
    attributeNames = ['offset'] + list(attributes)
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
            # Adjust figure size
            figure(figsize=(12,8))
            
            # Adjust horizontal spacing between subplots
            plt.subplots_adjust(wspace=0.5)

            subplot(1,2,1)
            semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-')
            xlabel(r"$\lambda$", fontsize=16)
            ylabel(r"$w_{i}$", fontsize=16)
            grid()
            legend(attributeNames[1:], loc='best')

            
            subplot(1,2,2)
            loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
            xlabel(r"$\lambda$", fontsize=16)
            ylabel('SE', fontsize=16)
            legend(['Train error','Validation error'])
            grid()

            # Adding line as marker for optimal lambda value:
            subplot(1,2,1).axvline(x=opt_lambda, color='red', linestyle='--', linewidth=1.2)
            subplot(1,2,2).axvline(x=opt_lambda, color='red', linestyle='--', linewidth=1.2)
        
        i+=1
    
    show()

    print("RESULTS: \n")
    print("Lamdas vs Generalization error:")
    print([f"{lamb:.0e}" for lamb in lambdas])
    print([round(test_err, 3) for test_err in test_err_vs_lambda])
    print(" \n")
    print('Weights in last fold:')
    for m in range(M):
        print('{:>15} {:>15}'.format(attributeNames[m+1], np.round(w_rlr[m,-1],2)))




#############################################
####             Main code               ####
#############################################


#### Extracting base data:
X, y, attributes, data_values, one_k = dataprep()
N, M = X.shape   


#### Analysing 'Seasons' influence with PCA as for P.1:
# PCA(data_values, one_k)


#### Linear regression model, part A:
K = 10
K_inner = K
lambdas = np.power(10.,range(-2,8))
linear_regression(X, y, attributes, K, K_inner, lambdas)



#### Linear regression model, part A:
K = 10
K_inner = K
lambdas = np.power(10.,range(-1,8))
linear_regression(X, y, attributes, K, K_inner, lambdas)
