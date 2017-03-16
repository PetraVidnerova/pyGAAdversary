import numpy as np
from load_models import load_models 
from show import predict 
import matplotlib.pyplot as plot 

def test_generalization(X, target, model_name, model):
    Y = predict(model_name, model, X)
    y = Y.argmax(axis=1)
    y = (y == target).astype(int)
    return y 
 
def generalization_matrix(model_name1, model_name, model):
    Z = None
    for i in range(10):
        try:
            X = np.load("adversary_inputs/adversary_inputs_against_{0}_{1}.npy".format(model_name1, i))
            z = test_generalization(X, i, model_name, model) 
        except:
            z = 0.5*np.ones(10)

        if Z is None:
            Z = z 
        else:
            Z = np.vstack((Z, z))

    np.fill_diagonal(Z, 0.5)
    return Z 
            
if __name__ == "__main__":

    models = load_models() 
    model_names = [ "MLP", "CNN", "SVM_sigmoid", "SVM_poly", "SVM_poly4", 
                    "SVM_linear", "SVM_rbf", "RBF", "DT"]

    f, pltarr = plot.subplots(9,9)
    f.tight_layout()
    #plot.axis('off')
    #for i in range(9):
    #   pltarr[8][i].set_xlabel(model_names[i])
    #   pltarr[i][0].set_ylabel(model_names[i])

    for target_model, i in zip(model_names, range(9)):
        for model, j  in zip(model_names, range(9)):
            print(target_model, model)
            Z = generalization_matrix(target_model, model, models[model])
            pltarr[i][j].imshow(Z, interpolation="none", cmap=plot.cm.Greys)
            # set off ticks and tick labels 
            plot.setp(pltarr[i][j].get_xticklabels(),visible=False)
            plot.setp(pltarr[i][j].get_yticklabels(),visible=False)
            pltarr[i][j].tick_params(axis=u'both', which=u'both',length=0)

    plot.savefig("generalization.eps")
    plot.show() 
