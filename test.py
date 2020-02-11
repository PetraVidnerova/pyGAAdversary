import matplotlib.pyplot as plt
import numpy as np
import sys

from load_models import load_model

MODEL = sys.argv[1]
ID = sys.argv[2]
# OUTPUT = sys.argv[2]
# IMAGE = sys.argv[3]

# result_names = [
#     f"results/adversary_sample_{MODEL}_{OUTPUT}_{IMAGE}_{round(treshold, 3)}.npy"
#     for treshold in np.linspace(0.001, 0.010, 10)
# ]

fig, ax = plt.subplots(10, 10)

for image in range(10):
    for class_ in range(10):
        filename = f"results/adversary_sample_{MODEL}_{class_}_{image}_{ID}.npy"
        print(filename)

        success = True
        try:
            X = np.load(filename)
        except FileNotFoundError:
            print("Does not exists.")
            success = False

        if success:
            cnn = load_model(MODEL)
            pred = cnn.predict(X.reshape(1, 28, 28, 1))

            print("Prediction: ", pred.argmax())
            if pred.argmax() == int(class_):
                print("Classification: CORRECT")
            else:
                print("Classification: WRONG")
                X = np.ones(28*28)
        else:
            X = np.zeros(28*28)

        ax[image][class_].set_yticklabels([])
        ax[image][class_].set_xticklabels([])
        ax[image][class_].imshow(X.reshape(28, 28), vmin=0.0, vmax=1.0,
                                 interpolation='none', cmap=plt.cm.Greys)


plt.savefig(f"adversary_sample_{MODEL}_{ID}", bbox_inches="tight")
plt.show()
