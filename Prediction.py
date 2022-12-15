import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import cv2

def getPreds(numpy,model):
    # model = keras.models.load_model("model.h5")
    numbers = numpy
    # numbers = np.zeros(shape=(10,28,28))
    # for count, i in enumerate(range(0,10)):
    #     img = cv2.imread("image"+str(i)+".png")[:,:,0]
    #     print("PRE NUMPY")
    #     print(img)
    #     img = np.array([img])
    #     print("POST NUMPY")
    #     print(img)
    #     img = img/255
    #     print(img)
    #     numbers[count] = img
    #     print(count)
    #     print(i)
    #print(numbers)

    prediction = model.predict(numbers)

    preds = []
    for i in range(len(numbers)):
    #     plt.grid(False)
    #     # plt.imshow(test_images[i], cmap = plt.cm.binary)
    #     plt.imshow(numbers[i], cmap = plt.cm.binary)
    # # plt.xlabel("Actual: " + str(test_labels[i]))
        
    #     plt.title("prediction: " + str(np.argmax(prediction[i])))
    #     plt.show()
        preds.append(str(np.argmax(prediction[i])))
    # print(np.argmax(prediction[0]))
    return preds

