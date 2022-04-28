from tensorflow import keras
import random
import json
import requests
import numpy as np


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
test_images = test_images / 255.0

# reshape for feeding into the model
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

rando = random.randint(0, len(test_images)-1)

data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))


# send data using POST request and receive prediction result
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/test_model:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

for i in range(3):
    print(('The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
      class_names[np.argmax(predictions[i])], np.argmax(predictions[i]), class_names[test_labels[i]], test_labels[i])))