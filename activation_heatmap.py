import cv2
import numpy as np
import pandas as pd
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mimg

img_path = "data/cat.jpg"
img = mimg.imread(img_path)
plt.imshow(img)


def process_image(input_img):
    img = image.load_img(input_img, target_size=(224, 224))
    np_arr = image.img_to_array(img)
    np_arr = np.expand_dims(np_arr, axis=0)
    np_arr = preprocess_input(np_arr)
    return np_arr


img = process_image(img_path)

model = VGG16(weights='imagenet')
model.summary()

preds = model.predict(img)
predictions = pd.DataFrame(decode_predictions(preds)[0], columns=[
                           'col1', 'category', 'probability']).iloc[:, 1:]
print('Prediction:', predictions.loc[0, 'category'])


def make_grad_cam(inputs):
    np_argmax = np.argmax(preds[0])
    output = model.output[:, np_argmax]

    # Get the last convolutional layer in our model
    last_conv_layer = model.get_layer('block5_conv3')

    # Get the gradients of the outputs
    gradients = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(gradients, axis=(0, 1, 2))

    iterate = K.function(
        [model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([inputs])

    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    return heatmap


def gen_heatmaps(image_input):
    heatmap = make_grad_cam(image_input)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap


def superimpose_heatmap(image_input, hmap, intense_factor):
    img = cv2.imread(image_input)

    # Resizing the heatmap
    heatmap = cv2.resize(hmap, (img.shape[1], img.shape[0]))

    # Converting the heatmap values into rgb values
    heatmap = np.uint8(255 * heatmap)

    # Applying a colormap to the heatmap, which in this notebook we will use COLORMAP_JET
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * intense_factor + img

    # Saving the superimposed image
    output = 'output/cat.jpg'
    cv2.imwrite(output, superimposed_img)

    print("Finished generating image!")
    print("Output: " + output)

    return output


if __name__ == "__main__":
    img = superimpose_heatmap(img_path, gen_heatmaps(img), 0.5)
    plt.imshow(mimg.imread(img))
    plt.title(predictions.loc[0, 'category'])
    plt.show()
