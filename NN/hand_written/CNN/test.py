from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === Step 1: Load model ===
model = load_model("num_cnn_model.h5")

# === Step 2: Load custom image ===
img_path = "../numbers/99.png"

# Load, convert to grayscale, resize to 28x28
img = Image.open(img_path).convert("L").resize((28, 28))

# Invert image colors if needed (must be white digit on black background)
img = np.invert(img)

# Convert to array and normalize
img_array = np.array(img).astype("float32") / 255.0

# Reshape to match input shape (1, 28, 28, 1)
img_array = img_array.reshape(1, 28, 28, 1)

# === Step 3: Predict ===
prediction = model.predict(img_array)
predicted_label = np.argmax(prediction)

# === Step 4: Show result ===
plt.imshow(img_array[0].reshape(28, 28), cmap="gray")
plt.title(f"Predicted Digit: {predicted_label}")
plt.axis("off")
plt.show()
