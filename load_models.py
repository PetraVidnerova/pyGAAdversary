from keras.models import model_from_json
from keras.models import  load_model as keras_load_model
from sklearn.svm import SVC 
import pickle 
from tensorflow.keras.models import load_model as tf_load_model
from rbflayer import RBFLayer

#import sys
#sys.path.insert(0, '/home/petra/pyRBF/')
#from rbf import *

def load_MLP():
    # mlp = model_from_json(open("models/mlp.json").read())
    # mlp.load_weights("models/mlp_weights.h5") 
    mlp = keras_load_model("models/mlp_sigmoid.h5")
    mlp.summary()
    return mlp 

def load_CNN(): 
    # cnn = model_from_json(open("models/cnn.json").read()) 
    # cnn.load_weights("models/cnn_weights.h5") 
    cnn = keras_load_model("models/cnn_mnist.h5")
    return cnn 

def load_RBF():
    rbf = tf_load_model("models/rbf.h5",
                        custom_objects={'RBFLayer': RBFLayer})
    return rbf

def load_SVM(kernel):
    return pickle.load(open("models/svm_{0}.pickle".format(kernel), "rb")) 

def load_DT():
    return pickle.load(open("models/tree.pickle", "rb"))    

func_dict = {}  
func_dict["MLP"] = load_MLP 
func_dict["CNN"] = load_CNN 
func_dict["RBF"] = load_RBF
func_dict["DT"] = load_DT 
func_dict["SVM_linear"] = lambda: load_SVM("linear")
func_dict["SVM_rbf"] = lambda: load_SVM("rbf") 
func_dict["SVM_sigmoid"] = lambda: load_SVM("sigmoid")
func_dict["SVM_poly"] = lambda: load_SVM("poly")
func_dict["SVM_poly4"] = lambda: load_SVM("poly4")

def load_models():
    """ Create a dictionary of learned models. 
    """ 

    models_dict = {} 

    for name, func in func_dict.items():
        
        print(name, "...", end="")
        models_dict[name] = func() 
        print("ok") 

    return models_dict


def load_model(name):
    
    return func_dict[name]()
