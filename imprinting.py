"""
In case of heavy imbalance amongst classes (e.g. classes with only 1 element),
even if features identified by network are enough to distinguish,
Fullyconnected layer doesn't get enough training to work well with those features.
If bottleneck features have dimension D and number of classes is N, FC layer = W of shape NxD.
We can replace FC layer with average bottleneck features of all N classes each having dimension D.

https://arxiv.org/abs/1712.07136
"""

import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

from tensorflow.keras.datasets import cifar10
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import normalize

def accuracy(model, x_test, y_test):
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)
    y_test = np.squeeze(y_test)
    correct = predictions[predictions==y_test]
    accuracy = len(correct) / len(predictions)
    return accuracy

def imprint(model, x_train, y_train):
    # create bottleneck network and embeddings
    n_classes = len(np.unique(y_train))
    embedder_layers = model.layers[0]
    embedder = keras.models.Model(inputs=embedder_layers.layers[0].input, outputs=embedder_layers.layers[-1].output)
    embeddings = embedder.predict(x_train, verbose=0)

    # create a matrix of per class average embedding
    indices = [[] for i in range(n_classes)]
    imprintings = np.zeros((n_classes, embeddings.shape[1]))

    for i, c in enumerate(y_train[:,0]):
        indices[c].append(i)

    for c, i_group in enumerate(indices):
        embeddings_subset = embeddings[i_group]
        average_embedding = np.average(embeddings_subset)
        imprintings[c] = average_embedding

    # replace weight matrix of dense layer with imprinted weights
    imprintings = normalize(imprintings)
    coefficients = imprintings.T
    model.layers[-1].set_weights([coefficients])

def get_model(n_classes, imshape):
    base_model = keras.applications.MobileNetV2(input_shape=imshape,
                                                include_top=False,
                                                pooling='avg',
                                                alpha=0.35)

    model = keras.Sequential([base_model, keras.layers.Dense(n_classes, activation='softmax', use_bias=False)])
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model

def test_imprinting(imprint_flag=False):
    ## Load model and train it
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    n_classes = len(np.unique(y_train))
    imshape = x_train.shape[1:]

    model = get_model(n_classes, imshape)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, shuffle=True, verbose=0, batch_size=128)
    if imprint_flag:
        imprint(model, x_train, y_train)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, shuffle=True, verbose=0, batch_size=128)

    print("Imprinting:{flag} Accuracy:{accuracy}".format(accuracy=accuracy(model, x_test, y_test),
                                                                       flag=imprint_flag))
if __name__ == '__main__':
    import sys
    flag = sys.argv[1].lower() == 'y'
    test_imprinting(flag)