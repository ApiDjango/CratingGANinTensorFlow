import tensorflow as tf
from models import make_generator_model, make_discriminator_model
from train_step import train_step
from image_loader import load_images_from_folder

# Set hyperparameters
BUFFER_SIZE = 60000
BATCH_SIZE = 256
IMG_SIZE = 28
NOISE_DIM = 100
EPOCHS = 50
DISC_LR = 2e-4
GEN_LR = 1e-4

# Load the dataset
train_dataset = load_images_from_folder('image', IMG_SIZE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Create the generator and discriminator models
generator = make_generator_model(NOISE_DIM)
discriminator = make_discriminator_model()

# Define the optimizers and loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(GEN_LR)
discriminator_optimizer = tf.keras.optimizers.Adam(DISC_LR)

# Train the model
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for image_batch in train_dataset:
        train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy, NOISE_DIM, BATCH_SIZE)

#generator.save('generator_model.h5')
generator.save_weights('generator_weights.h5')