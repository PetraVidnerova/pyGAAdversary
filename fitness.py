from keras.datasets import mnist  
from keras.models import model_from_json 
from scipy.spatial.distance import cdist
import numpy as np

class Fitness:

    def __init__(self, model, index=1):
        # Load mnist dataset and choose letter.
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784)
        X_train = X_train.astype('float32') 
        X_train /= 255 

        # Select one class. 
        self.target = X_train[index]
        
        # Trained network.
        self.model = model 

    def evaluate(self, individual):

        dist = cdist(np.atleast_2d(individual), np.atleast_2d(self.target))

        X = np.array([individual])
        res = self.model.predict(X)
        dist2 = cdist(np.atleast_2d(res), np.atleast_2d([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

        fit = dist*0.5 + 0.5*dist2

        return fit, 
 
