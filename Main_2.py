
import numpy as np
import pandas as pd
import time
import torch
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from toolbox_02450 import rlr_validate, train_neural_net
from matplotlib import pyplot as plt
from scipy.linalg import svd
from matplotlib.pylab import figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid
from tqdm import tqdm


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
def linear_regression(X, y, attributes, K, lambdas):
    
    # Applying method from script 8.1.1
    # Adding offset attribute to X
    X_off = np.concatenate((np.ones((X.shape[0],1)),X),1)
    attributeNames = ['offset'] + list(attributes)
    N, M = X.shape
    M_off = M+1

    # Create crossvalidation partition for evaluation
    CV = model_selection.KFold(K, shuffle=True)

    # Creating empty matrices for the results
    w_rlr = np.empty((M_off,K))
    mu = np.empty((K, M))
    sigma = np.empty((K, M))

    i=0
    for train_index, test_index in CV.split(X,y):
    
        # extract training and test set for current CV fold
        X_train = X_off[train_index]
        y_train = y[train_index]

        # Calculating the optimal lambda value
        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, K)
        

        mu[i, :] = np.mean(X_train[:, 1:], 0)
        sigma[i, :] = np.std(X_train[:, 1:], 0)
        
        X_train[:, 1:] = (X_train[:, 1:] - mu[i, :] ) / sigma[i, :] 
        
        # Pre-calculate matrices
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M_off)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w_rlr[:,i] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()

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
    
    print('Weights of regularized coefficients:')
    for m in range(M):
        print('{:>15} {:>15}'.format(attributeNames[m+1], np.round(w_rlr[m+1,-1],3)))


# Defining ANN model for comparisment
def ANN_model(M, hidden_units):
    model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, hidden_units), #M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(),                            #torch.nn.ReLU(),
                    torch.nn.Linear(hidden_units, 1), # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function
                    )
    return model


# Comparing models (Linear reg, ANN, baseline):
def compare_models(X, y, K1, K2, lambdas, h_units):

    N, M = X.shape

    # Create crossvalidation partition for the loops
    # rand.state = subtracting the same 'random' set every run
    CV_out = model_selection.KFold(K1, shuffle=True, random_state=42)
    CV_inn = model_selection.KFold(K2, shuffle=True, random_state=42)
    
    # Parameters for neural network classifier
    
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000         # stop criterion 2 (max epochs in training)
    
    # Defining loss function
    loss_fn = torch.nn.BCELoss()

    # Creating empty result array [0]K1-Fold, [1-2]RLR MSE, [3-4]ANN MSE, [5]Baseline MSE:
    results = np.zeros((K1, 6))


    # print('Initiating training of model of type:\n{}\n'.format(str(model())))
    start_time = time.time()

    ###################################
    # Starting up the training:
    ###################################

    #### Initiating outer loop ####
    for i, (train_index, test_index) in enumerate(CV_out.split(X,y)): 
        print('\n Outer fold: {0}/{1}'.format(i+1,K1))

        # Fetch appropriate training and test data
        X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
        

        # Preallocate memory to hold error terms for baseline "w", linear reg "lr"
        w = np.empty((M,K2,len(lambdas)))
        rlr_test_error = np.empty((K2,len(lambdas)))
        h_test_error = np.empty((K2,len(h_units)))
        errors = []
        
        opt_lambda = 0
        best_lambda_mse = float('inf')

        #### Initiating inner loop ####
        for i2, (train_index_inn, test_index_inn) in enumerate(CV_inn.split(X_train,y_train)): 
            print('\n Inner fold: {0}/{1}'.format(i2+1,K2))

            # Fetch appropriate training and test data
            X_train_inn, y_train_inn, X_test_inn, y_test_inn = X_train[train_index_inn], y_train[train_index_inn], X_train[test_index_inn], y_train[test_index_inn]

            # Converting testdata to PyTorch tensors
            X_train_tensor = torch.Tensor(X_train_inn)
            y_train_tensor = torch.Tensor(np.reshape(y_train_inn, (len(y_train_inn), -1)))
            X_test_tensor = torch.Tensor(X_test_inn)

            # Pre-calculate matrices
            Xty_inn = X_train_inn.T @ y_train_inn
            XtX_inn = X_train_inn.T @ X_train_inn
            
            # Calculating optimal lambda for regularized linear regression:
            # for i3, lambda_n in enumerate(lambdas):
            #     # Estimating the optimal lamba value
            #     lambdaI = lambda_n * np.eye(M)
                
            #     # Calculate weights
            #     w[:,i2,i3] = np.linalg.solve(XtX_inn+lambdaI, Xty_inn).squeeze()

            #     # Insert Mean Squared Error into the array
            #     rlr_test_error[i2,i3] = np.square(y_test_inn-X_test_inn @ w[:,i2,i3].T).mean(axis=0)

            #### Training RLR model(s) ####
            mse_l = []
            for i3, lambda_n in enumerate(lambdas):
                rlr = Ridge(alpha=lambda_n)
                rlr.fit(X_train_inn, y_train_inn)
                y_pred_rlr = rlr.predict(X_test_inn)
                mse_l.append(((y_test_inn - y_pred_rlr)**2).mean())
            
            # Determining the optimal lambda value
            mean_mse_l = np.mean(mse_l)

            if mean_mse_l < best_lambda_mse:
                opt_lambda = lambda_n
                opt_lambda_mse = mean_mse_l


            #### Training ANN model(s) ####
            for i4, h in enumerate(h_units):
                # Choosing ANN by h_unit
                model = ANN_model(M, h)

                # Training the ANN
                net, final_loss, learning_curve = train_neural_net(model, loss_fn, X=X_train_tensor, y=y_train_tensor, n_replicates=n_replicates, max_iter=max_iter)

                # Calculate y prediction and convert it to np array
                y_test_est = net(X_test_tensor)
                y_test_est_np = y_test_est.cpu().detach().numpy() 

                # Insert MSE into the array
                h_test_error[i2, i4] =  np.square(y_test_est_np - y_test_inn.T).sum(axis=0)[0] / y_test_inn.size


        # Convert np arrays to tensors
        X_train_tensor = torch.Tensor(X_train)
        y_train_tensor = torch.Tensor(np.reshape(y_train, (len(y_train), -1)))
        X_test_tensor = torch.Tensor(X_test)


        #### Computing the ANN with optimal h-units ####
        #(ANN) Calculate the optimal amount of hidden units
        opt_h = h_units[np.argmin(np.mean(h_test_error,axis=0))]
        
        #(ANN) creating ANN model with optimal 'opt_h'
        model = ANN_model(M, opt_h)

        #(ANN) Training the opt_ANN:
        net, final_loss, learning_curve = train_neural_net(model, loss_fn, X=X_train_tensor, y=y_train_tensor, n_replicates=n_replicates)

        #(ANN) Calculate y prediction and convert it to np array
        y_test_est = net(X_test_tensor)
        y_test_est_np = y_test_est.cpu().detach().numpy() 

        #(ANN) Calculating the opt_h MSE
        opt_h_mse =  np.square(y_test_est_np.flatten() - y_test.T).mean(axis=0)
        ################################################


        #### Computing the RLR with optimal lambda-value ####
        # #(RLR) Collecting the optimal lambda choice
        # opt_lambda = lambdas[np.argmin(np.mean(rlr_test_error,axis=0))]

        # #(RLR) Pre-calculate matrices
        # Xty = X_train.T @ y_train
        # XtX = X_train.T @ X_train

        # #(RLR) calculating the weights
        # lambdaI = opt_lambda * np.eye(M)
        # w_rlr = np.linalg.solve(XtX+lambdaI, Xty).squeeze()

        # #(RLR) Calculating the MSE for the optimal lambda value
        # opt_lambda_mse = np.square(y_test-X_test @ w_rlr).sum(axis=0)/y_test.shape[0]
        ################################################


        #### Computing the LR (Baseline) ####
        # Training a LR without regularization i.e alpha=0
        baseline = Ridge(alpha=0)
        baseline.fit(X_train, y_train)
        y_pred_baseline = baseline.predict(X_test)

        #(LR) Calculate MSE for baseline model
        lr_mse = ((y_test - y_pred_baseline)**2).mean()
        ################################################


        # Record the results
        results[i, 0] = i+1
        results[i, 1] = opt_h
        results[i, 2] = opt_h_mse
        results[i, 3] = opt_lambda
        results[i, 4] = opt_lambda_mse
        results[i, 5] = lr_mse
    

    end_time = time.time()

    runtime = end_time - start_time
    print("Runtime:", runtime, "seconds")
    print("\n Results:")
    print(results)


'''
def compare_models(X, y, K1, K2, lambdas, h_values):

    kf_outer = KFold(n_splits=K1, shuffle=True, random_state=42)
    kf_inner = KFold(n_splits=K2, shuffle=True, random_state=42)
    
    results = np.zeros((K1, 6))


    for i, (train_index, test_index) in enumerate(kf_outer.split(X,y)):
        X_train_outer, X_test_outer = X[train_index], X[test_index]
        y_train_outer, y_test_outer = y[train_index], y[test_index]

        best_h = 0
        best_h_mse = float('inf')
        best_lambda = 0
        best_lambda_mse = float('inf')

        for h in h_values:
            for l in lambdas:
                mse_h = []
                mse_l = []
                
                for train_inner_index, test_inner_index in kf_inner.split(X_train_outer):
                    X_train_inner, X_test_inner = X_train_outer[train_inner_index], X_train_outer[test_inner_index]
                    y_train_inner, y_test_inner = y_train_outer[train_inner_index], y_train_outer[test_inner_index]

                    # Train ANN
                    ann = MLPRegressor(hidden_layer_sizes=(h,), max_iter=1000, alpha=l, solver='adam',
                                        activation='relu', random_state=42)
                    ann.fit(X_train_inner, y_train_inner)
                    y_pred_ann = ann.predict(X_test_inner)
                    mse_h.append(((y_test_inner - y_pred_ann)**2).mean())

                    # Train RLR
                    rlr = Ridge(alpha=l)
                    rlr.fit(X_train_inner, y_train_inner)
                    y_pred_rlr = rlr.predict(X_test_inner)
                    mse_l.append(((y_test_inner - y_pred_rlr)**2).mean())

                mean_mse_h = np.mean(mse_h)
                mean_mse_l = np.mean(mse_l)

                # Update best h and lambda
                if mean_mse_h < best_h_mse:
                    best_h = h
                    best_h_mse = mean_mse_h
                if mean_mse_l < best_lambda_mse:
                    best_lambda = l
                    best_lambda_mse = mean_mse_l

        # Train baseline model
        baseline = Ridge(alpha=0)
        baseline.fit(X_train_outer, y_train_outer)
        y_pred_baseline = baseline.predict(X_test_outer)

        # Record results
        results[i, 0] = i+1
        results[i, 1] = best_h
        results[i, 2] = best_h_mse
        results[i, 3] = best_lambda
        results[i, 4] = best_lambda_mse
        results[i, 5] = ((y_test_outer - y_pred_baseline)**2).mean()

    return results
'''


#############################################
####             Main code               ####
#############################################


#### Extracting base data:
X, y, attributes, data_values, one_k = dataprep()   


#### Analysing 'Seasons' influence with PCA as for P.1:
# PCA(data_values, one_k)


#### Linear regression model, part A:
K = 10
K_inner = K
lambdas = np.power(10.,range(-2,8))
# linear_regression(X, y, attributes, K, lambdas)

#### Comparing models, part B:
h_units = range(1,11)     # number of hidden units
K1 = 10
K2 = K1
lambdas = range(1,50)
compare_models(X, y, K1, K2, lambdas, h_units)
