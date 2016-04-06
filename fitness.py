from keras.datasets import mnist  
from keras.models import model_from_json 
from scipy.spatial.distance import cdist
import numpy as np

class Fitness:

    def __init__(self, models, target_image=0, target_output=0):
        # Load mnist dataset and choose letter.
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784)
        X_train = X_train.astype('float32') 
        X_train /= 255 
        
        self.target_output = target_output

        # Select one class. 
        self.target = X_train[target_image]
        

        # Trained network.
        self.models = models 

    def evaluate(self, individual):

        dist = cdist(np.atleast_2d(individual), np.atleast_2d(self.target))

        X = np.array([individual])
        res = [] 
        for model in models:
            r = self.model.predict(X)
            res.append(r)
        mean = np.zeros(len(res[0]))
        for r in res:
            mean = mean + r
        mean /= len(res) 
        desired_output = np.zeros(10)
        desired_output[self.target_output] = 1.0 
        dist2 = cdist(np.atleast_2d(res), np.atleast_2d(desired_output))

        fit = dist*0.5 + 0.5*dist2

        return fit, 
 
