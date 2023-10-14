""" 
Code logistic regression using Python. DO NOT use a "package" but rather, code it by hand. Yes you can use simple packages like Numpy, etc. You cannot use a Logistic regression package. 

You must write code that will:
(a) Read in any .csv labeled dataset where the Label is called "LABEL". 
(b) Your code should create "y" (the label) as a numpy array of 0 and 1. So, if the labels are originally something like POS and NEG, you r code will need to determine this, and then update all POS to 1 and all NEG to 0. Your code should be able to do this for any two label options. 
(c) Your code will also create X which is the entire dataset (without the labels) as a numpy array. 
(d) Your code will initialize w and b and a learning rate. 
(e) Your code will need to contain a function for Sigmoid and for z, etc. 
(f) Your code will then need to perform gradient descent to optimize (reduce) LCE.
(g) Be sure that your code can run iteratively for as many epochs as you wish. 
(h) Show a final confusion matrix (see example below) AND show a graph of how the LCE reduces as epochs iterate (see example below)

Recommended: Use your code to see if your by-hand results from above are right. The goal here is to really understand all the steps, which steps are repeated (epochs), the loss function (LCE here), the activation function (Sigmoid here), updating the parameters, etc.  
"""
#%% libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import confusion_matrix  
import seaborn as sns


# %%
class OneLayer_OneOutput_FF():
    def __init__(self,
                 data_filename = "A2_Data_JasmineKobayashi.csv", 
                 hidden_units = 4, 
                 activation="sigmoid",
                 scale_data = True,
                 weight_init = "simple"):
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
        self.y = df[['LABEL']]

        # TODO: create W1
        if weight_init == "random":
            pass
        else:
            # self.W1 = np.array()
            pass
        # TODO: create H1

        # Other useful stuff-----------------------------------------------------------------------------------
        # self.epochs = epochs
        # self.LR = LR
        # self.y_hat = None
        # self.Lce_list = []
        
        pass

    def MSE_loss(self):
        # L = mean((y_hat - y)^2)
        pass
        
    def lin_eq(self,X,w,b):
        return X @ w + b
    
    def Sigmoid(self,z,derivative = False):
        if derivative:
            return self.Simoid(z) * (1 - self.Sigmoid(z)) # dS/dz = S(z)(1-S(z))
        # Sigmoid
        return 1/(1 + math.e**(-z)) # S(z) = 1/(1 + e^(-z))


    # Gradient descent----------------------------------------------requirement (f)
    def grad_desc(self,y_hat,y,X):
        error = y_hat-y
        m = self.n_obs
        
        dL_dw = (1/m)*(error.T@X)
        dL_db = (1/m)*error
        dL_db = np.mean(dL_db)
        return dL_dw.T, dL_db
    
    def LCE(self,y_hat,y):
        m = self.n_obs
        summation = np.sum(y*np.log(y_hat)+((1-y)*np.log(1-y_hat)))
        return (-1/m)*summation

    # Visualizations (confusion matrix & Lce plot)==================requirement (h)
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
        plt.plot(np.arange(0,len(self.Lce_list)),self.Lce_list,'-',label="LR = {}".format(self.LR))
        plt.xlabel("epochs")
        plt.ylabel('Loss')
        plt.title("Loss over epochs")
        plt.legend()

    # Run multiple iterations (with specified epochs, etc.)========================================================
    def run_model(self, print_all_params = True):
        
        # Save LCE values (for plotting)------------------------------------
        self.Lce_list = []
        #-------------------------------------------------------------------
        
        for i in range(self.epochs):
            # Linear eq: apply parameters to data---------------------------
            self.z = self.lin_eq(X=self.X,w=self.w,b=self.b)
            
            # Sigmoid activation--------------------------------------------
            self.y_hat = self.Sigmoid(z=self.z)
            
            # Calculate loss------------------------------------------------
            loss = self.LCE(y_hat=self.y_hat,y=self.y)
            self.Lce_list.append(loss)
           
            # print info to check if doing as expected (if wanted)----------
            if print_all_params:
                            print("Epoch",i)
                            print("w vector: \n",self.w)
                            print("b = ",self.b)
                            print("y^: \n",self.y_hat)
                            print("LCE (Loss) = ", loss)
            
            # Update params with gradient descent---------------------------
            dL_dw,dL_db = self.grad_desc(y_hat=self.y_hat,y=self.y,X=self.X)
            # p_new = p_prev - (LR)gradient(p_prev)
            self.w = self.w - self.LR*dL_dw
            self.b = self.b - self.LR*dL_db
        print("finished running all epochs")

        # Visualizations
        self.ConfusionMatrix(y_hat=self.y_hat,y=self.y)
        self.plot_LCE()