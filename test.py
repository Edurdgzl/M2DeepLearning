from tensorflow import keras 
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pathlib
from tensorflow import nn, expand_dims
import numpy as np

new_base_dir = pathlib.Path("data")
test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180, 180),
    batch_size=32)


# Dataset test
test_model = keras.models.load_model("model.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


# Individual test
image = keras.utils.load_img('test.jpg', target_size=(180, 180))
img_arr = keras.utils.img_to_array(image)
img_arr = expand_dims(img_arr, 0)
predicts = test_model.predict(img_arr)
score = nn.softmax(predicts[0])
print(['female','male'][np.argmax(score)], 100 * np.max(score))