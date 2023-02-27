import tensorflow as tf
import os

def load_images_from_folder(folder_path, img_size):
    # Get all image filenames from folder
    file_names = os.listdir(folder_path)
    file_names = [os.path.join(folder_path, file) for file in file_names]

    # Create a dataset of file names
    dataset = tf.data.Dataset.from_tensor_slices(file_names)

    # Decode images and resize them
    def decode_and_resize(filename):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32)
        # Normalize the images to [-1, 1]
        image = (image / 127.5) - 1.0
        return image

    # Apply the decode_and_resize function to each element in the dataset
    dataset = dataset.map(decode_and_resize)

    return dataset
