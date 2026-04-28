import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load scaler parameters
scaler = StandardScaler()
scaler.mean_ = np.load("scaler_mean.npy")
scaler.scale_ = np.load("scaler_scale.npy")

# 1. Load numeric dataset (Iris)
iris = load_iris()

# Load the saved model
model = tf.keras.models.load_model("iris_model.h5")

# 8. Make a prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example iris flower
sample = scaler.transform(sample)  # Normalize
pred_logits = model.predict(sample)

print("Logits:", pred_logits)

pred_class = np.argmax(pred_logits, axis=1)

print("Predicted class:", pred_class, "->", iris.target_names[pred_class][0])
