from tensorflow import keras 
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pathlib


new_base_dir = pathlib.Path("data")


test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180, 180),
    batch_size=32)

test_model = keras.models.load_model("convnet_from_scratch_with_augmentation.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")