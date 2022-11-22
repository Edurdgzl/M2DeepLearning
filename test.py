import pathlib
import numpy as np
from tensorflow import keras
from tensorflow import expand_dims 
from tensorflow.keras.preprocessing import image_dataset_from_directory


new_base_dir = pathlib.Path("data")
test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180, 180),
    batch_size=32)


# Dataset test
test_model = keras.models.load_model("model.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


classes = ["Female", "Male"]


# Individual tests
def img_test(img):
    image = keras.utils.load_img(img, target_size=(180, 180))
    img_arr = keras.utils.img_to_array(image)
    img_arr = expand_dims(img_arr, 0)
    predicts = test_model.predict(img_arr)
    binary_class = np.where(predicts[0][0] > 0.5, 1, 0)
    print("Everything below 0.5 is labeled as female and everything above 0.5 is labeled as male.")
    print(predicts[0][0])
    print(classes[binary_class])


# Female test
test = img_test('tests/test_female.jpg')


# Male test
test = img_test('tests/test_male.jpg')


#Own image test
#Upload your jpg image into tests folder, insert image name over FILENAME and remove the comment

# test = img_test('tests/FILENAME.jpg')