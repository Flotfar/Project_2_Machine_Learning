
import numpy as np
import pandas as pd
import time
import torch
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn import model_selection, linear_model, tree
from toolbox_02450 import rlr_validate, train_neural_net, correlated_ttest
from matplotlib import pyplot as plt
from scipy.linalg import svd
from matplotlib.pylab import figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid
from tabulate import tabulate


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

    #Extending attributes with seasons:
    label = list(attributes)
    label = np.concatenate((label, season_categories))
    attributes = np.asarray(label)

    # Creating Y column
    Y_raw = data_values[:, 0]
    Y = (Y_raw - np.mean(Y_raw))/np.std(Y_raw)

    print("Data loaded succesfully. \n")
    return X, Y, Y_raw, attributes, data_values, one_k

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

    # Creating empty result array [0]K1-Fold, [1-2]ANN MSE, [3-4]RLR MSE, [5]Baseline MSE:
    results = np.zeros((K1, 6))
    results_label = np.array(["Fold", "h-val", "h-MSE", "l-val", "l-MSE", "b-MSE"])

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
        h_test_error = np.empty((K2,len(h_units)))
        opt_lambda = 0
        opt_lambda_mse = float('inf')

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


            #### Training RLR model ####
            mse_l = []
            for lambda_n in lambdas:
                rlr = Ridge(alpha=lambda_n)
                rlr.fit(X_train_inn, y_train_inn)
                y_pred_rlr = rlr.predict(X_test_inn)
                mse_l.append(((y_test_inn - y_pred_rlr)**2).mean())

                # Determining the optimal lambda value
                mean_mse_l = np.mean(mse_l)
            
                if mean_mse_l < opt_lambda_mse:
                    opt_lambda = lambda_n
                    opt_lambda_mse = mean_mse_l

            #### Training ANN model(s) ####
            for i4, h in enumerate(h_units):
                # Choosing ANN by h_unit
                model = ANN_model(M, h)

                # Training the ANN
                net, final_loss, learning_curve = train_neural_net(model, loss_fn, X=X_train_tensor, y=y_train_tensor, n_replicates=n_replicates , max_iter=max_iter)

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


        #### Computing the LR (Baseline) ####
        # Training a LR without regularization i.e alpha=0
        baseline = LinearRegression()
        baseline.fit(X_train, y_train)
        y_pred_baseline = baseline.predict(X_test)

        #(LR) Calculate MSE for baseline model
        lr_mse = ((y_test - y_pred_baseline)**2).mean()
        ################################################


        # Record the results
        results[i, 0] = i+1
        results[i, 1] = opt_h
        results[i, 2] = round(opt_h_mse, 3)
        results[i, 3] = opt_lambda
        results[i, 4] = round(opt_lambda_mse, 3)
        results[i, 5] = round(lr_mse, 3)
    

    # Use function from toolbox to determine p-value and confidence interval
    p_RLR_ANN, CI_RLR_ANN = correlated_ttest(abs(results[:,2]-results[:,4]), 1/10, 0.05)
    p_ANN_base, CI_ANN_base = correlated_ttest(abs(results[:,2]-results[:,5]), 1/10, 0.05)
    p_RLR_base, CI_RLR_base = correlated_ttest(abs(results[:,4]-results[:,5]), 1/10, 0.05)
    
    # creating titles for the result output and
    # printing the array as a LaTeX table using tabulate
    results_w_labels = np.row_stack((results_label, results))
    result_table = tabulate(results_w_labels, headers='firstrow', tablefmt='latex_booktabs')

    # defining runtime
    end_time = time.time()
    runtime = end_time - start_time

    # printing results
    print("Runtime:", runtime, "seconds")
    print("\n Results:")
    print(result_table)
    print()
    print(f"ANN vs. RLR: \np = {p_RLR_ANN}, Confidence interval = {CI_RLR_ANN}\n")
    print(f"RLR vs. baseline: \np = {p_RLR_base}, Confidence interval = {CI_RLR_base}\n")
    print(f"ANN vs. baseline: \np = {p_ANN_base}, Confidence interval = {CI_ANN_base}\n")


# Comparing clasification models (Logreg, ANN, baseline):
def compare_classification_models(X, y_raw, K1, K2, lambdas, tree_depths, ozone_threshold, alpha, attributes):
    
    N, M = X.shape

    #discitizing the y-values
    y_dis = np.asarray([1 if ozone_value > ozone_threshold else 0 for ozone_value in y_raw])
    print(y_dis)

    # Create crossvalidation partition for the loops
    # rand.state = subtracting the same 'random' set every run
    CV_out = model_selection.KFold(K1, shuffle=True, random_state=42)
    CV_inn = model_selection.KFold(K2, shuffle=True, random_state=42)


    # Creating empty result array [0]K1-Fold, [1-2]RLR MSE, [3-4]ANN MSE, [5]Baseline MSE:
    results = np.zeros((K1, 6))
    results_label = np.array(["Fold", "opt-tree-depth", "ct-MSE", "l-val", "l-MSE", "b-MSE"])

    # print('Initiating training of model of type:\n{}\n'.format(str(model())))
    start_time = time.time()
    
    ###################################
    # Starting up the training:
    ###################################

    #### Initiating outer loop ####
    for i, (train_index, test_index) in enumerate(CV_out.split(X,y)): 
        print('\n Outer fold: {0}/{1}'.format(i+1,K1))

        # Fetch appropriate training and test data
        X_train, y_train, X_test, y_test = X[train_index], y_dis[train_index], X[test_index], y_dis[test_index]
        
        # Preallocate memory to hold error terms for classification tree "ct", linear reg "lr"
        ct_test_error = np.empty((K2,len(tree_depths)))
        lr_test_error = np.empty((K2,len(lambdas)))

        #### Initiating inner loop ####
        for i2, (train_index_inn, test_index_inn) in enumerate(CV_inn.split(X_train,y_train)): 
            print('\n Inner fold: {0}/{1}'.format(i2+1,K2))

            # Fetch appropriate training and test data
            X_train_inn, y_train_inn, X_test_inn, y_test_inn = X_train[train_index_inn], y_train[train_index_inn], X_train[test_index_inn], y_train[test_index_inn]

            # Converting testdata to PyTorch tensors
            X_train_tensor = torch.Tensor(X_train_inn)
            y_train_tensor = torch.Tensor(np.reshape(y_train_inn, (len(y_train_inn), -1)))
            X_test_tensor = torch.Tensor(X_test_inn)

            #### Training LR model ####
            
            for i3, lambda_n in enumerate(lambdas):
                # Instantiate new log reg model with different lambda values
                log_reg = LogisticRegression(C=lambda_n)

                model_log = log_reg.fit(X_train_inn, y_train_inn)
                # Calculate model prediction
                y_pred_log_reg = model_log.predict(X_test_inn)
                
                # Insert misclassified observation rate into array
                lr_test_error[i2,i3] = np.sum(y_pred_log_reg != y_test_inn) / len(y_test_inn)


            #### Training CT models ####

            for i4, tree_depth in enumerate(tree_depths):
                # Instantiate new tree model with different tree depths
                dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=tree_depth)
                model_dtc = dtc.fit(X_train_inn,y_train_inn)

                # Calculate model prediction
                y_pred_dtc = model_dtc.predict(X_test_inn)

                # Insert misclassified observation rate into array
                ct_test_error[i2, i4] = np.sum(y_pred_dtc != y_test_inn) / len(y_test_inn)

        # Convert np arrays to tensors
        X_train_tensor = torch.Tensor(X_train)
        y_train_tensor = torch.Tensor(np.reshape(y_train, (len(y_train), -1)))
        X_test_tensor = torch.Tensor(X_test)




        #(LR) Calculate the optimal lambda
        opt_lambda = lambdas[np.argmin(np.mean(lr_test_error,axis=0))] 
        #(CT) Calculate the optimal tree depth
        opt_tree_depth = tree_depths[np.argmin(np.mean(ct_test_error,axis=0))]

        #(LR) creating LR model with optimal 'opt_lambda'
        model_log = LogisticRegression(C=opt_lambda).fit(X_train_tensor, y_train_tensor)
        #(CT) creating CT model with optimal 'opt_tree_depth'
        model_dt = tree.DecisionTreeClassifier(criterion='gini', max_depth=opt_tree_depth).fit(X_train_tensor, y_train_tensor)

        #(LR) predicting y
        y_test_log_reg = model_log.predict(X_test_tensor)
        #(CT) predicting y
        y_test_dt = model_dt.predict(X_test_tensor)

        #(CT) Calculating the opt ct MSE
        opt_ct_mse =  np.sum(y_test_dt.flatten() != y_test.T) / len(y_test)
        #(CT) Calculating the opt lambda MSE
        opt_lambda_mse = np.sum(y_test_log_reg.flatten() != y_test.T) / len(y_test)


        ################################################


        #### Computing the Baseline with a no features model ####

        positives = len(np.where(y_train == 1)[0])
        negatives = len(np.where(y_train == 0)[0])
        
        # In case more positives than negatives
        if positives < negatives:
            test_error_baseline = len(np.where(y_test == 0)[0]) / len(y_test)
        
        # In case more neagatives than positives
        else:   
            test_error_baseline = len(np.where(y_test == 1)[0]) / len(y_test)
        
        ################################################


        # Record the results
        results[i, 0] = i+1
        results[i, 1] = opt_tree_depth
        results[i, 2] = round(opt_ct_mse, 3)
        results[i, 3] = opt_lambda
        results[i, 4] = round(opt_lambda_mse, 3)
        results[i, 5] = round(test_error_baseline, 3)
    

    # creating titles for the result output and
    # printing the array as a LaTeX table using tabulate
    results_w_labels = np.row_stack((results_label, results))
    result_table = tabulate(results_w_labels, headers='firstrow', tablefmt='latex_booktabs')

    # defining runtime
    end_time = time.time()
    runtime = end_time - start_time

    # printing results
    print("Runtime:", runtime, "seconds")
    print("\n Results:")
    print(result_table)

    # defining rho
    rho = 1/K1
    
    # Defining the model errors into individual arrays
    error_ct = np.asarray(results[:,2])
    error_lr = np.asarray(results[:,4])
    error_baseline = np.asarray(results[:,5])
    

    # Calculate the pairwise test error differences
    abs_ct_lr = abs(error_ct - error_lr)
    abs_ct_baseline = abs(error_ct - error_baseline)
    abs_lr_baseline = abs(error_lr - error_baseline)

    # Calculate p-value and confidence interval
    pval_ct_lr, ci_ct_lr = correlated_ttest(abs_ct_lr, rho, alpha)
    pval_ct_baseline, ci_ct_baseline = correlated_ttest(abs_ct_baseline, rho, alpha)
    pval_lr_baseline, ci_lr_baseline = correlated_ttest(abs_lr_baseline, rho, alpha)
    
    # Print results
    print('CT vs. LR: p-val =',  pval_ct_lr, 'CI =', ci_ct_lr)
    print('CT vs. Baseline: p-val =', pval_ct_baseline, 'CI =', ci_ct_baseline)
    print('LR vs. Baseline: p-val =', pval_lr_baseline, 'CI =', ci_lr_baseline)


    #####################################
    ### Training the logistical model ### 
    #####################################
    # the most common opt_lambda = 10.0 is used
    opt_lambda = 10.0
    
    # training the logistical model
    model = LogisticRegression(C=opt_lambda).fit(X, y_dis)
    
    # getting the attribute 
    print(attributes)
    print(model.coef_[0],3)



#############################################
####             Main code               ####
#############################################


#### Extracting base data:
X, y, y_raw, attributes, data_values, one_k = dataprep()   


#### Analysing 'Seasons' influence with PCA as for P.1:
# PCA(data_values, one_k)


#### Linear regression model, part A:
K = 10
K_inner = K
lambdas = np.power(10.,range(-2,8))
#linear_regression(X, y, attributes, K, lambdas)

#### Comparing models, part B:
h_units = range(1,11)     # number of hidden units
K1 = 10
K2 = 10
lambdas = range(1,100)
# compare_models(X, y, K1, K2, lambdas, h_units)

#### Comparing classification models:
tree_depths = range(1,11)     # classification tree depth
K1 = 10
K2 = 10
lambdas = np.power(10.,range(-2,8))
ozone_threshold = 20
alpha = 0.05 #used for 95 % CI
compare_classification_models(X, y_raw, K1, K2, lambdas, tree_depths, ozone_threshold, alpha, attributes)



