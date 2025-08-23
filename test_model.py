import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model
model = load_model("model.h5")

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# Load CIFAR-10 test data
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype("float32") / 255.0

# Pick a random sample
idx = np.random.randint(0, len(x_test))
img = x_test[idx]
true_label = class_names[y_test[idx][0]]

# Predict
img_input = np.expand_dims(img, axis=0)
prediction = model.predict(img_input)
predicted_label = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

# Show result
plt.imshow(img)
plt.title(f"True: {true_label}, Pred: {predicted_label}, Conf: {confidence:.2f}%")
plt.axis("off")
plt.show()
