from json import load
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

model = VGG16()

idxs = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in idxs]
model = Model(inputs=model.inputs, outputs=outputs)

img = load_img("samples/car.jpg", target_size=(224, 224))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

feature_maps = model.predict(img)
square = 8

for feature_map in feature_maps:
    idx = 1

    for _ in range(square):
        for _ in range(square):
            axes = plt.subplot(square, square, idx)
            axes.set_xticks([])
            axes.set_yticks([])

            plt.imshow(feature_map[0, :, :, idx-1], cmap="gray")
            idx += 1

    plt.show()