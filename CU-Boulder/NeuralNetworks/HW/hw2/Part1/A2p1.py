#%% libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.metrics import confusion_matrix  
import seaborn as sns
import random

# %%
class OneLayer_OneOutput():
    def __init__(self,
                 data_filename = "A2_Data_JasmineKobayashi.csv", 
                 data_normalizer = StandardScaler(),
                 hidden_units= 4,
                 randomize_initial_parameters = False,
                 verbose = True):
        # Read Data
        self.df = pd.read_csv(data_filename)
        
        # create X and y, assuming data has column "LABEL" as column with labels
        ## X matrix
        if data_normalizer is not None:
            self.data_scaler = data_normalizer
            X = self.data_scaler.fit_transform(self.df.drop(columns="LABEL",axis=1))
            print("X was normalized using {}".format(data_normalizer))
        else:
            self.data_scaler = None
            X = np.array(self.df.drop(columns=['LABEL'],axis=1))
            print("X was not normalized")

        self.X = X

        ## y
        self.y = np.array(self.df[['LABEL']])

        if verbose:
            print("Shape of X:", self.X.shape)
            print("X matrix: \n", self.X)
            print("Shape of y:", self.y.shape)
            print("y vector: \n", self.y)

        # Shape of W1
        w1_ncol = hidden_units        # Number of W1 cols should be number of hidden units in H1
        w1_nrow = self.X.shape[1]     # Number of W1 rows should be number of X columns (dim of X)
        # Shape of B (column vector)
        b_length = self.X.shape[0]    # length of X rows
        # Shape of W2:
        w2_ncol = 1                   # Number of W2 cols should be number of outputs (in this case, only one output)
        w2_nrow = hidden_units        # Number of W2 rows should be number of hidden units in H1
        # Shape of C (column vector)
        c_length = self.X.shape[0]    # length of X rows
        
        if randomize_initial_parameters: 
            # TODO: Finish potential code of randomized parameters
            # # bounds of randomized weights
            # bound = 1/np.sqrt(w1_nrow)   # I found online that supposedly this is a standard range for randomized weights
            # rand_w = [np.random.uniform(-bound,bound) for i in range(w1_ncol)]
            # self.W1 = np.array([rand_w]*w1_nrow)

            # self.W2 = np.array([]*w2_nrow)
            # print('All parameters were randomized')
            pass
        else:   # These are the parameters defined by the question 2 in Assignment2
            self.W1 = np.array([[1]*w1_ncol]*w1_nrow) # otherwise, all weights of W1 = 1
            self.B = np.array([[0]]*b_length)         # otherwise, all bias (B) = 0
            self.W2 = np.array([[2]*w2_ncol]*w2_nrow) # otherwise, all weights of W2 = 2
            self.C = np.array([[0]]*c_length)
            print("Parameters built for simple calculation (not randomized)")
        
        if verbose:
            # print("# of X cols (should be # of W1 rows):", self.X.shape[1])
            # print("# of hidden units (should be # of W1 cols):", hidden_units)
            print("W1 has shape:", self.W1.shape)
            print("W1 matrix: \n", self.W1)
            print("Shape of B:", self.B.shape)
            print("B (bias vector for Z1): \n", self.B)
            print("Shape of W2:",self.W2.shape)
            print("W2 matrix: \n", self.W2)
            print("Shape of C :", self.C.shape)
            print("C (bias vector for Z2): \n",self.C)


    def MSE_loss(self,y_hat,y):
        # L = mean((y_hat - y)^2)
        return np.mean((y_hat - y)**2)
        

    def lin_eq(self,X,W,B):
        assert X.shape[1] == W.shape[0], "Linear eq: z = X @ W + B. Shape of X: {}; Shape of W: {}".format(X.shape,W.shape)
        assert B.shape[0] == X.shape[0], "Linear eq: z = X @ W + B. Length of B should match X rows; Length B = {}; Shape of X = {}".format(B.shape[0],X.shape)
        return X @ W + B
    
    def Sigmoid(self,z,
                derivative = False):
        if derivative:
            return self.Sigmoid(z) * (1 - self.Sigmoid(z)) # dS/dz = S(z)(1-S(z))
        # Sigmoid
        return 1/(1 + math.e**(-z)) # S(z) = 1/(1 + e^(-z))
    

    def FeedForward(self,
                    activation = "sigmoid",
                    verbose = True):
        # Create H1
        # Z1 = X @ W1 + B
        self.Z1 = self.lin_eq(self.X,self.W1,self.B)
        if verbose:
            print("Shape of Z1:",self.Z1.shape)
            print("Z1 matrix: \n",self.Z1)
        self.H1 = self.Sigmoid(self.Z1)
        # TODO: Add flexibility with other activation functions
        if verbose: 
            print("Activation function:",activation)
            print("H1 matrix: \n", self.H1)

        # Z2 and y-hat
        # Z2 = H1 @ W2 + C
        self.Z2 = self.lin_eq(self.H1,self.W2,self.C)
        self.y_hat = self.Sigmoid(self.Z2)

        if verbose:
            print("Shape of Z2:", self.Z2.shape)
            print("Z2 matrix: \n", self.Z2)
            print("y_hat:\n",self.y_hat)
        
    
    # Gradient descent----------------------------------------------
    def grad_desc(self,y_hat,y,LR,
                  verbose = True):
        # Error = y^ - y = E
        self.E = y_hat - y                                      # shape = n x 1 

        # dy^/dZ2 = S(Z2)(1 - S(Z2)) = y^(1 - y^) = y^ - (y^)(y^) = Phi
        self.Phi = self.Sigmoid(self.Z2,derivative=True)        # shape = Z2.shape = n x 1
        
        # dL/dZ2  = [y^-y][S(Z2)(1 - S(Z2))] is a term that shows up in every gradient 
        # so it's going to be useful to save this
        self.dL_dZ2 = self.E * self.Phi                                   # shape = [n x 1]*[n x 1] => [n x 1]

        self.dH_dZ1 = self.Sigmoid(self.Z1,derivative=True)     # shape = Z1.shape = n x 4

        # dL/dC = [dL/dy^][dy^/dZ2][dZ2/C] 
        #       = [y^-y][S(Z2)(1 - S(Z2))][1]
        self.dZ2_dC = np.array([[1]]*self.C.shape[0])           # vector of 1s with same shape of C
        self.dL_dC = self.dL_dZ2 * self.dZ2_dC                            # should match shape of C (n x 1)

        # dL/dW2 = [dL/dy^][dy^/dZ2][dZ2/W2] 
        #        = [y^-y][S(Z2)(1 - S(Z2))][H1]
        self.dL_dW2 =  self.H1.T @ self.dL_dZ2                         # should match shape of W2 (4 x 1)

        # dL/dB = [dL/dy^][dy^/dZ2][dZ2/dH1][dH1/dZ1][dZ1/dB] 
        #       = [y^-y][S(Z2)(1 - S(Z2))][W2][S(Z1)(1-S(Z1))][1]
        #       = ([y^-y] * [S(Z2)(1 - S(Z2))]) * ([S(Z1)(1-S(Z1))] @ [W2]) * [1] 
        
        # dH1/dZ1 = S(Z1)(1-S(Z1)) = H1(1-H1)  # I'm going to call this matrix Omega
        self.Omega = self.Sigmoid(self.Z1,derivative = True)      # should match shape of Z1 (n x 4)
        self.dZ1_dB = np.array([[1]]*self.B.shape[0])           # vector of 1s with same shape of B
        # [n x n][n x 1][n x 1]
        self.dL_dB = self.dL_dZ2 * (self.Omega @ self.W2) * self.dZ1_dB               # should match shape of B (n x 1)
        # dL/dW1 = [dL/dy^][dy^/dZ2][dZ2/dH1][dH1/dZ1][dZ1/dW1] 
        #        = [y^-y][S(Z2)(1 - S(Z2))][W2][S(Z1)(1-S(Z1))][X] 
        #        = [X]^T @ [([[y^-y] * [S(Z2)(1 - S(Z2))]] @ [W2]) * [S(Z1)(1-S(Z1))]] 
        #        = [X]^T @ [([E * Phi] * [W2]^T) * Omega]
        self.dL_W1 = self.X.T @ (( (self.E * self.Phi) @ self.W2.T) * self.Omega)    # should match shape of W1 (3 x 4)

        # Gradient descent
        self.C = self.C - (LR*self.dL_dC)
        self.W2 = self.W2 - (LR*self.dL_dW2)
        self.B = self.B - (LR*self.dL_dB)
        self.W1 = self.W1 - (LR*self.dL_W1)

        if verbose:
            # print("# of X cols (should be # of W1 rows):", self.X.shape[1])
            # print("# of hidden units (should be # of W1 cols):", hidden_units)
            print("Updated W1 has shape:", self.W1.shape)
            print("Updated W1 matrix: \n", self.W1)
            print("Shape of updated B:", self.B.shape)
            print("Updated B (bias vector for Z1): \n", self.B)
            print("Shape of updated W2:",self.W2.shape)
            print("Updated W2 matrix: \n", self.W2)
            print("Shape of updated  C :", self.C.shape)
            print("Updated C (bias vector for Z2): \n",self.C)

    # Visualizations (confusion matrix & Lce plot)==================
    # Confusion matrix-------------------------------------------------------------
    def ConfusionMatrix(self, y_hat,y):
        self.prediction = y_hat
        self.prediction[y_hat >= .5] = 1
        self.prediction[y_hat < .5] = 0

        self.labels = y
        
        
        self.cm = confusion_matrix(self.labels, self.prediction)
        print(self.cm)
        plt.figure()
        ax= plt.subplot()
        sns.heatmap(self.cm, annot=True, fmt='g', ax=ax, cmap='Blues')  
        #annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")
        ax.xaxis.set_ticklabels(["0", "1"])
        ax.yaxis.set_ticklabels(["0", "1"])
    
    # Plot of Lce vs. Epochs--------------------------------------------------------
    def plot_LCE(self):
        plt.figure()
        plt.plot(np.arange(0,len(self.loss_record)),self.loss_record,'-',label="LR = {}".format(self.LR))
        plt.xlabel("epochs")
        plt.ylabel('Loss')
        plt.title("Loss over epochs")
        plt.legend()

    # Run multiple iterations (with specified epochs, etc.)========================================================
    def run_model(self, 
                  epochs,
                  LR= 1,
                  hidden_units= 4,
                  randomize_parameters = False,
                  activation = "sigmoid", 
                  verbose = True):
        
        # Save LCE values (for plotting)------------------------------------
        self.loss_record = []
        #-------------------------------------------------------------------
        self.LR = LR
        print("Running model for {} epochs...".format(epochs))
        for i in range(epochs):
            print("Epoch",i)
            self.FeedForward(activation= activation,
                             verbose=verbose)
            
            if verbose:
                print("MSE Loss:",self.MSE_loss(y_hat=self.y_hat,y=self.y))
            self.loss_record.append(self.MSE_loss(y_hat=self.y_hat,
                                                  y=self.y))
            # Update params with gradient descent---------------------------
            self.grad_desc(y_hat=self.y_hat,
                           y=self.y,
                           LR=self.LR,
                           verbose=verbose)
        print("Finished running all epochs")

        # Visualizations
        self.ConfusionMatrix(y_hat=self.y_hat,y=self.y)
        self.plot_LCE()

    def test_model(self,
                   test_data,
                   verbose = True):
        self.test_df = pd.read_csv(test_data)

        print("X is now input from test data")
        if self.data_scaler is not None:
            self.X = self.data_scaler.transform(self.test_df.drop(columns="LABEL",axis=1))
            print("Test input X was normalized using same normalizer as training ({})".format(self.data_scaler))
        else:
            self.X = self.test_df.drop(columns="LABEL",axis=1)

        self.y = np.array(self.test_df[['LABEL']])

        if verbose:
            print("Shape of X:", self.X.shape)
            print("X matrix: \n", self.X)
            print("Shape of y:", self.y.shape)
            print("y vector: \n", self.y)

        print("Testing model with test data")
        self.FeedForward()
        print("Finished testing model")

        self.ConfusionMatrix(y_hat=self.y_hat,
                             y=self.y)
        