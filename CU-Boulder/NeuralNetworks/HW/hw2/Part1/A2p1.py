#%% libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import confusion_matrix  
import seaborn as sns
import random

# %%
class OneLayer_OneOutput_FF():
    def __init__(self,
                 data_filename = "A2_Data_JasmineKobayashi.csv", 
                 scale_data = True,
                 verbose = True):
        # Read Data
        self.df = pd.read_csv(data_filename)
        
        # create X and y, assuming data has column "LABEL" as column with labels
        ## X matrix
        if scale_data:
            mm_scaler = MinMaxScaler()
            X = mm_scaler.fit_transform(self.df.drop(columns="LABEL",axis=1))
        else:
            X = np.array(self.df.drop(columns=['LABEL'],axis=1))
        self.X = X

        ## y
        self.y = np.array(self.df[['LABEL']])

        if verbose:
            print("Shape of X:", self.X.shape)
            print("X matrix: \n", self.X)
            print("Shape of y:", self.y.shape)
            print("y vector: \n", self.y)

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
                    hidden_units= 4,
                    randomize_parameters = False,
                    activation = "sigmoid",
                    verbose = True):
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
        
        if randomize_parameters: 
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
    def grad_desc(self,y_hat,y,X):
        pass

    # Back Prop
    def BackPropagation(self):
        pass

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
        # plt.figure()
        # plt.plot(np.arange(0,len(self.Lce_list)),self.Lce_list,'-',label="LR = {}".format(self.LR))
        # plt.xlabel("epochs")
        # plt.ylabel('Loss')
        # plt.title("Loss over epochs")
        # plt.legend()
        pass

    # Run multiple iterations (with specified epochs, etc.)========================================================
    def run_model(self, print_all_params = True):
        
        # # Save LCE values (for plotting)------------------------------------
        # self.Lce_list = []
        # #-------------------------------------------------------------------
        
        # for i in range(self.epochs):
        #     # Linear eq: apply parameters to data---------------------------
        #     self.z = self.lin_eq(X=self.X,w=self.w,b=self.b)
            
        #     # Sigmoid activation--------------------------------------------
        #     self.y_hat = self.Sigmoid(z=self.z)
            
        #     # Calculate loss------------------------------------------------
        #     loss = self.LCE(y_hat=self.y_hat,y=self.y)
        #     self.Lce_list.append(loss)
           
        #     # print info to check if doing as expected (if wanted)----------
        #     if print_all_params:
        #                     print("Epoch",i)
        #                     print("w vector: \n",self.w)
        #                     print("b = ",self.b)
        #                     print("y^: \n",self.y_hat)
        #                     print("LCE (Loss) = ", loss)
            
        #     # Update params with gradient descent---------------------------
        #     dL_dw,dL_db = self.grad_desc(y_hat=self.y_hat,y=self.y,X=self.X)
        #     # p_new = p_prev - (LR)gradient(p_prev)
        #     self.w = self.w - self.LR*dL_dw
        #     self.b = self.b - self.LR*dL_db
        # print("finished running all epochs")

        # # Visualizations
        # self.ConfusionMatrix(y_hat=self.y_hat,y=self.y)
        # self.plot_LCE()
        pass

