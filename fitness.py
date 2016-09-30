from keras.datasets import mnist  
from keras.models import model_from_json 
from scipy.spatial.distance import cdist
import numpy as np
import random

SVM = False

class Fitness:

    def __init__(self, name, model, target_image=0, target_output=0):
        # Load mnist dataset and choose letter.
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784)
        X_train = X_train.astype('float32') 
        X_train /= 255 
        
        self.target_output = target_output

        # Select one class. 
        #ZERO_train = X_train[y_train == target_image] 
        #self.target = random.choice(ZERO_train) 
        self.target = X_train[target_image]

        # Trained network.
        self.model = model 
        self.model_name = name

    def evaluate(self, individual):

        dist = cdist(np.atleast_2d(individual), np.atleast_2d(self.target))

        if (self.model_name == "CNN"):
            X = np.array([np.array(individual).reshape(1,28,28)])
        else:
            X = np.array([individual]) 
       
        if self.model_name.startswith("SVM") or self.model_name.startswith("DT"):
            model_output = self.model.predict_proba(X)
        else:
            model_output = self.model.predict(X)


        desired_output = np.zeros(10)
        desired_output[self.target_output] = 1.0 
        
        dist2 = cdist(np.atleast_2d(model_output), np.atleast_2d(desired_output))
            

        fit = dist*0.5 + 0.5*dist2
        #fit = dist 

        return fit, 
 
