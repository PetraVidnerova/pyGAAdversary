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

        dist = cdist(np.atleast_2d(individual),
                     np.atleast_2d(self.target)).squeeze()
        dist /= self.target.size

        if (self.model_name == "CNN"):
            X = np.array([np.array(individual).reshape(28, 28, 1)])
        else:
            X = np.array([individual])

        if self.model_name.startswith("SVM") or self.model_name.startswith("DT"):
            model_output = self.model.predict_proba(X)
        else:
            model_output = self.model.predict(X)

        desired_output = np.zeros(10)
        desired_output[self.target_output] = 1.0

        dist2 = cdist(np.atleast_2d(model_output),
                      np.atleast_2d(desired_output)).squeeze()
        dist2 /= 10  # FIXME number of classes

#        print(self.target_output, model_output.argmax())
        success = self.target_output == model_output.argmax()

        if dist < 0.01:
            dist /= 100
        if success:
            dist2 /= 100

        # if not success:
        #     dist2 += 0.5 # penalty

        # fit = dist*0.9 + 0.1*dist2
        #fit = dist

        return dist, dist2, success
