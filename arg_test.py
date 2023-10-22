import argparse
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pygad
import wandb
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# print(device_lib.list_local_devices())

import matplotlib.pyplot as plt
tf.random.set_seed(0)
np.random.seed(0)



parser = argparse.ArgumentParser(description="Description of your script")


parser.add_argument("--outlier_magnitude_factor", "-of", help="Outlier magnitude factor", default="10")
parser.add_argument("--latent_dim", "-ld", help="latent dimension", default="4")
parser.add_argument("--z_loss_weight", "-zl", help="Z loss weight", default="0.05")
parser.add_argument("--reg_loss_weight", "-rl", help="Reg loss weight", default="0.05")
parser.add_argument("--batch_size", "-b", help="batch size", default="128")
parser.add_argument("--epochs", "-e", help="epochs", default="100")
parser.add_argument("--num_samples", "-ns", help="Number of samples", default="5000")

args = parser.parse_args()


#########################################
####### Creating the dataset ############
#########################################



# Set the number of classes, dimensions, and samples per class
num_classes = 5
num_dimensions = 10  # Set the dataset to 3 dimensions
num_samples_per_class = int(args.num_samples)
num_outliers = 10
cluster_std_factor = 1.0  # Adjust this factor as needed
outlier_magnitude_factor = int(args.outlier_magnitude_factor)

# Generate random data for each class
datasets = []
for i in range(num_classes):
    data, _ = make_blobs(n_samples=num_samples_per_class, n_features=num_dimensions, centers=1, cluster_std=cluster_std_factor, random_state=i)
    datasets.append(data)

# Add outliers
outliers = []
outlier_data = []
for i in range(num_classes):
    # Generate outliers close to one class
    outlier_class = np.random.randint(0, num_classes)
    outlier_samples = np.random.rand(num_outliers, num_dimensions) * outlier_magnitude_factor
    outlier_samples += datasets[outlier_class][:num_outliers]  # Add outliers close to a class
    outlier_data.append(outlier_samples)
outliers.append(outlier_data)

# Combine data and outliers, and add binary outlier column
final_datasets = []
for i in range(num_classes):
    data = datasets[i]
    outliers_data = outliers[0][i]
    
    # Add binary outlier column (1 for outliers, 0 for non-outliers)
    data = np.column_stack((data, np.zeros(len(data))))
    outliers_data = np.column_stack((outliers_data, np.ones(len(outliers_data))))
    
    data_with_outliers = np.vstack((data, outliers_data))
    final_datasets.append(data_with_outliers)

# Shuffle the combined dataset
combined_dataset = np.vstack(final_datasets)
combined_dataset = shuffle(combined_dataset, random_state=42)

# Create the training set (all samples) and target set (classification assignments)
X_train = combined_dataset[:, :-1]  # Features (all dimensions except the outlier flag)
y_train = combined_dataset[:, -1]   # Target (outlier flag)

# Print the shapes of the training set and target set
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


######################################################
########## Building the RAE network ##################
######################################################

latent_dim = int(args.latent_dim)




encoder_inputs = keras.Input(shape=(num_dimensions,))
# x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
# x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Flatten()(x)
x = layers.Dense(20, activation="sigmoid")(encoder_inputs)
x = layers.Dense(18, activation="sigmoid")(x)
x = layers.Dense(16, activation="sigmoid")(x)
encoder_output = layers.Dense(latent_dim, activation="sigmoid")(x)
# z_mean = layers.Dense(latent_dim, name="z_mean")(x)
# z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
# z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, encoder_output, name="encoder")

latent_inputs = keras.Input(shape=(latent_dim,))
# x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
# x = layers.Reshape((7, 7, 64))(x)
# x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
x = layers.Dense(16, activation="sigmoid")(latent_inputs)
x = layers.Dense(18, activation="sigmoid")(x)
x = layers.Dense(20, activation="sigmoid")(x)
# x = layers.Dense(512, activation="relu")(x)
decoder_outputs = layers.Dense(num_dimensions, activation="linear")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")



class RAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.z_tracker = keras.metrics.Mean(name="z_loss")
        self.REG_tracker = keras.metrics.Mean(name="REG_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.z_tracker,
            self.REG_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
#             reconstruction_loss = tf.reduce_mean(
#                 tf.reduce_sum(
#                     keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
#                 )
#             )
            reconstruction_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(data,reconstruction)
            # reconstruction_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(data, reconstruction)
#             
            z_loss = K.mean(K.square(z), axis=[1])
    

#             gradients = tape.gradient(reconstruction, self.encoder(data))
    

#             gradient_norm = 0.0
#             for gradient in gradients:
#                 if gradient is not None:
#                     gradient_norm += tf.norm(gradient, ord=2)
    

            REG_loss = K.mean(K.square(K.gradients(K.square(reconstruction), z)))

            z_loss_w = float(args.z_loss_weight)
            REG_loss_w = float(args.reg_loss_weight)

            total_loss = reconstruction_loss +  z_loss_w * z_loss + REG_loss_w * REG_loss
            # total_loss = reconstruction_loss
        
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.z_tracker.update_state(z_loss)
            self.REG_tracker.update_state(REG_loss)
            del tape
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "z_loss": self.z_tracker.result(),
                "REG_loss": self.REG_tracker.result(),
            }
        

#####################################################
############### Training the Netwrok ################
#####################################################

batch_size = int(args.batch_size)
epochs = int(args.epochs)

tdata = np.concatenate([X_train], axis=0)
tdata = np.expand_dims(tdata, -1).astype("float32")

rae = RAE(encoder, decoder)
rae.compile(optimizer=tf.keras.optimizers.Adam())
history = rae.fit(tdata, epochs=epochs, batch_size=batch_size, verbose=0)



#################################################
############## Testing the Network ##############
#################################################

print(" ***************** Training phase Done! ****************** ")

outlier_indices = np.argwhere(y_train)
inlier_indices  = np.argwhere(1 - y_train)

outliers = X_train[outlier_indices,:]
inliers  = X_train[inlier_indices, :]

inliers_mean = []
outliers_mean= []

for i in range(inliers.shape[0]):
    
    sample = inliers[i,0,:].reshape([1,num_dimensions])

    z = rae.encoder(sample)
    reconstruction = rae.decoder(z)

    reconstruction_loss = tf.keras.losses.MeanSquaredError()(sample,reconstruction)
    
    inliers_mean.append(reconstruction_loss)

for i in range(outliers.shape[0]):
    
    sample = outliers[i,0,:].reshape([1,num_dimensions])

    z = rae.encoder(sample)
    reconstruction = rae.decoder(z)

    reconstruction_loss = tf.keras.losses.MeanSquaredError()(sample,reconstruction)
    
    outliers_mean.append(reconstruction_loss)

inliers_mean = np.array(inliers_mean)
outliers_mean = np.array(outliers_mean)

i_mean = np.mean(inliers_mean)
o_mean = np.mean(outliers_mean)

i_std = np.std(inliers_mean)
o_std = np.std(outliers_mean)

print(i_mean, i_std)
print(o_mean, o_std)


print(" ****************** Detecting phase ********************")

threshold = i_mean + 3*i_std

classes = []

for i in range(X_train.shape[0]):
    
    sample = X_train[i,:].reshape([1,num_dimensions])

    z = rae.encoder(sample)
    reconstruction = rae.decoder(z)

    reconstruction_loss = tf.keras.losses.MeanSquaredError()(sample,reconstruction)
    
    if reconstruction_loss > threshold:
        
        classes.append(1)
        
    else:
        
        classes.append(0)

classes = np.array(classes)

detected_num = np.sum(classes)

print("########## Number of outliers detected = ", detected_num)

# confusion = confusion_matrix(y_train, classes)

precision = precision_score(y_train, classes)
recall = recall_score(y_train, classes)
f1 = f1_score(y_train, classes)

# print("Confusion Matrix:")
# print(confusion)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
