# ### First neural network in keras
# We will use simple data of mobile price range classifier. The dataset consists of 20 features and we need to predict the price range in which phone lies. These ranges are divided into 4 classes.
# <br>
# Link to dataset - https://www.kaggle.com/iabhishekofficial/mobile-price-classification 
# <br>
# Link to blog -
# 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#keras
import keras
from keras.models import Sequential
from keras.layers import Dense


class NNKeras():
    def __init__(self,data=None,test_size = 0.1):
        #dataset import
        self.dataset = data
        if self.dataset is None:
            self.dataset = pd.read_csv('data/train.csv')
        # dataset.head(10)
        # dataset.columns

        #Changing pandas dataframe to numpy array
        self.X = self.dataset.iloc[:,:20].values
        self.y = self.dataset.iloc[:,20:21].values

        #Normalizing the data
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        # print('Normalized data:')
        # print(X[0])

        #One hot encode
        
        ohe = OneHotEncoder()
        self.y = ohe.fit_transform(self.y).toarray()
        # print('One hot encoded array:')
        # print(y[0:5])

        #Train test split of model
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size = test_size,random_state = 0)


        self.model = Sequential()

    def build_model(self, 
                    units_in_layer=[16,12,4], 
                    activation=['relu','relu','softmax'],
                    visualize_model = True):
        for i,layer_units in enumerate(units_in_layer):
            if i == 0:  #if first hidden layer
                self.model.add(Dense(layer_units, input_dim=self.X.shape[1], activation=activation[i]))
            else:
                self.model.add(Dense(layer_units, activation=activation[i]))

        if visualize_model:
            #To visualize neural network
            self.model.summary()

    def train_model(self,
                    loss = 'categorical_crossentropy',
                    optimizer = 'adam',
                    metrics = ['accuracy'],
                    validate = True,
                    epochs = 100,
                    batch_size = 64):
        self.model.compile(loss= loss, optimizer=optimizer, metrics=metrics)

        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

        if validate:
            #Using test data as validation data.
            self.history1 = self.model.fit(self.X_train, self.y_train,
                                           validation_data = (self.X_test,self.y_test), 
                                           epochs=epochs, 
                                           batch_size=batch_size)


    def predict_test(self):
        self.y_pred = self.model.predict(self.X_test)
        #Converting predictions to label
        self.pred_list = list()
        for i in range(len(self.y_pred)):
            self.pred_list.append(np.argmax(self.y_pred[i]))
        
        #Converting one hot encoded test label to label
        test = list()
        for i in range(len(self.y_test)):
            test.append(np.argmax(self.y_test[i]))
        self.accuracy = accuracy_score(self.pred_list,test)
        print('Accuracy is:', self.accuracy*100)


    def visualize(self):
        plt.plot(self.history1.history['accuracy'])
        plt.plot(self.history1.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        plt.plot(self.history1.history['loss'])
        plt.plot(self.history1.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
