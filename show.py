import numpy as np 
import matplotlib.pyplot as plot 
from keras.models import model_from_json 
from keras.datasets import mnist 
from matplotlib import rcParams

def probs_to_string(prob):
    # Pretty print vector of probabilities to string.
    res = ""
    for p in prob:
        res += "%.2f   " % p 
    return res


# Load matrix of adversary inputs.
# one image per line 
X = np.load("adversary_inputs_against_ensemble.npy") 

#  Load trained MLP. 
model = model_from_json(open("mlp.json").read()) 
model.load_weights("mlp_weights.h5")

# Load mnist data set. 
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
X_train = X_train.reshape(60000, 784)
X_train = X_train.astype('float32') 
X_train /= 255 

#lot.figure(figsize=(8.27,11.69))
plot.figure(1)
rcParams.update({'font.size': 8})
#f, pltarr = plot.subplots(5,2)

# For each image print prediction and show image.
for i in range(0,10):
#    INPUTS = np.array([X[i]])
#    res = model.predict(INPUTS)
    
 #   print(probs_to_string(res[0])) 
        
 #   x = X[i].reshape(28,28)
 #   plot.imshow(x, interpolation="none", cmap=plot.cm.Greys)
#    plot.title(probs_to_string(res[0]))

  #  filename = "ga_ensemble_%s.eps" % i
   # plot.savefig(filename, orientation = 'portrait')
    #plot.show() 

   # train = X_train[i].reshape(28, 28)
  #  plot.imshow(train, interpolation="none", cmap=plot.cm.Greys)

    res = model.predict(np.array([X_train[i]]))
    print(probs_to_string(res[0])) 
#    plot.title(probs_to_string(res[0]))

#    filename = "train_%s.eps" % i
 #   plot.savefig(filename, orientation = 'portrait')
    #plot.show()

#plot.show()

    

