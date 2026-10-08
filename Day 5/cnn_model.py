from sklearn.datasets import load_sample_images
import tensorflow as tf

images = load_sample_images()['images']

print(images[0].shape)

images = tf.keras.layers.CenterCrop(height=80, width=120)(images)
images = tf.keras.layers.Rescaling(scale=1./255)(images)

print(images.shape)

conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7)
fmaps = conv_layer(images)
print(fmaps.shape)

conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding='same')
fmaps = conv_layer(images)
print(fmaps.shape)

kernels, biases = conv_layer.get_weights()
print(kernels.shape, biases.shape)

max_pool = tf.keras.layers.MaxPooling2D(pool_size=2)
output = max_pool(images)
print(output.shape)

global_avg_pool = tf.keras.layers.GlobalAvgPool2D() 
print(global_avg_pool(images))