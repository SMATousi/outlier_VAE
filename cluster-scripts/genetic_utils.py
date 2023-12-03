import argparse
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pygad
# import wandb
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# print(device_lib.list_local_devices())

import matplotlib.pyplot as plt
tf.random.set_seed(0)
np.random.seed(0)


def generate_dataset_refined(n_samples=100000, n_outliers=100, dimensions=20):
    # Generate inliers uniformly within a range
    inliers = np.random.uniform(-1, 1, size=(n_samples - n_outliers, dimensions))

    # Prepare to generate outliers
    outlier_samples = []
    outlier_indices = []
    outlier_dims = []

    # Define different clusters of outliers
    cluster_definitions = [
        (0, 7),    # First 7 dimensions
        (5, 10),   # 5 dimensions in the middle
        (3, 8),    # Another set of 5 dimensions, overlapping with the first
        (1, 6),    # 5 dimensions starting from second
        (6, 10),   # Last 4 dimensions
        (0, 4),    # First 4 dimensions
        (2, 7),    # 5 dimensions starting from third
        (4, 9),    # 5 dimensions starting near the middle
        (3, 6),    # 3 dimensions in the middle
        (7, 10)    # Last 3 dimensions
    ]

    # Adjust if the number of dimensions is different
    if dimensions != 10:
        scaling_factor = dimensions // 10
        cluster_definitions = [(start * scaling_factor, min(end * scaling_factor, dimensions)) for start, end in cluster_definitions]

    # Generate outliers for each cluster
    for start, end in cluster_definitions:
        for _ in range(n_outliers // len(cluster_definitions)):
            # Normal values for non-deviating dimensions
            normal_dims = list(set(range(dimensions)) - set(range(start, end)))
            outlier = np.random.uniform(-1, 1, dimensions)
            # More extreme values for the deviating dimensions
            outlier[start:end] = np.random.uniform(1, 10, end - start)
            
            outlier_samples.append(outlier)
            outlier_indices.append(len(inliers) + len(outlier_samples) - 1)
            outlier_dims.append((start, end))

    # Combine inliers and outliers
    dataset = np.vstack([inliers, np.array(outlier_samples)])

    return dataset, outlier_indices, outlier_dims


def train_VAE(data,
              latent_dim = 2,
              hidden_layer_n = [20,18,16],
              num_dims = 10,
              kl_loss_factor = 0.01,
              epochs = 100,
              batch_size = 128
              ):


    """
    Training the VAE on the data
    """

    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    latent_dim = latent_dim

    encoder_inputs = keras.Input(shape=(num_dims,))
    x = layers.Dense(num_dims, activation="tanh")(encoder_inputs)
    x = layers.Dense(hidden_layer_n[0], activation="tanh")(x)
    x = layers.Dense(hidden_layer_n[1], activation="tanh")(x)
    x = layers.Dense(hidden_layer_n[2], activation="tanh")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(hidden_layer_n[2], activation="tanh")(latent_inputs)
    x = layers.Dense(hidden_layer_n[1], activation="tanh")(x)
    x = layers.Dense(hidden_layer_n[0], activation="tanh")(x)
    decoder_outputs = layers.Dense(num_dims, activation="linear")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(
                name="reconstruction_loss"
            )
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.keras.losses.MeanSquaredError()(data,reconstruction)
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                total_loss = reconstruction_loss + kl_loss_factor * kl_loss
        
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
    
    creditdata = np.concatenate([data], axis=0)
    creditdata = np.expand_dims(creditdata, -1).astype("float32")

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    history = vae.fit(creditdata,epochs=epochs,batch_size=batch_size,verbose=1)

    return vae, history




def train_RAE(data,
              latent_dim = 2,
              hidden_layer_n = [20,18,16],
              num_dims = 10,
              z_loss_w = 0.01,
              REG_loss_w = 0.01,
              epochs = 100,
              batch_size = 128
              ):


    """
    Training the RAE on the data
    """


    encoder_inputs = keras.Input(shape=(num_dims,))
    x = layers.Dense(hidden_layer_n[0], activation="sigmoid")(encoder_inputs)
    x = layers.Dense(hidden_layer_n[1], activation="sigmoid")(x)
    x = layers.Dense(hidden_layer_n[2], activation="sigmoid")(x)
    encoder_output = layers.Dense(latent_dim, activation="sigmoid")(x)
    encoder = keras.Model(encoder_inputs, encoder_output, name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(hidden_layer_n[2], activation="sigmoid")(latent_inputs)
    x = layers.Dense(hidden_layer_n[1], activation="sigmoid")(x)
    x = layers.Dense(hidden_layer_n[0], activation="sigmoid")(x)
    decoder_outputs = layers.Dense(num_dims, activation="linear")(x)
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

                reconstruction_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(data,reconstruction)

                z_loss = K.mean(K.square(z), axis=[1])
        
                REG_loss = K.mean(K.square(K.gradients(K.square(reconstruction), z)))


                total_loss = reconstruction_loss +  z_loss_w * z_loss + REG_loss_w * REG_loss
            
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

    tdata = np.concatenate([data], axis=0)
    tdata = np.expand_dims(tdata, -1).astype("float32")

    rae = RAE(encoder, decoder)
    rae.compile(optimizer=tf.keras.optimizers.Adam())
    history = rae.fit(tdata,epochs=epochs,batch_size=batch_size,verbose=0)



    return rae, history 


def run_genetic_on_sample(inlier_samples,
                          outlier_sample,
                          rae_model,
                          num_dims,
                          num_generations = 40,
                          num_parents_mating = 2,
                          sol_per_pop = 20,
                          init_range_low = -2,
                          init_range_high = 5,
                          parent_selection_type = "tournament",
                          K_tournament = 5,
                          keep_parents = 1,
                          crossover_type = "single_point",
                          mutation_type = "random",
                          mutation_percent_genes = 5,
                            ):

    def fitness_func(ga_instance, solution, solution_idx):

        inliers = inlier_samples

        avg_ins = np.mean(inliers, axis=0)
        avg_ins = avg_ins.reshape([1,num_dims])

        particle = outlier_sample
        particle = particle.reshape([1,num_dims])

        avg_in_rec = []
        avg_in_z = []

        for index in range(inliers.shape[0]):

            candidate_inlier = inliers[index,:]
            candidate_inlier = candidate_inlier.reshape([1,num_dims])

            in_normal_subspace = solution
            in_bad_subspace = 1 - solution        

            in_remain = candidate_inlier * in_normal_subspace



            in_replace = in_bad_subspace * avg_ins

            in_candidate = in_remain + in_replace

            z = rae_model.encoder(in_candidate)
            in_candidate_rec = rae_model.decoder(z)


            rec_loss = tf.keras.losses.MeanSquaredError()(in_candidate,in_candidate_rec)
            z_loss = K.mean(K.square(z), axis=[1])

            avg_in_rec.append(rec_loss.numpy())
            avg_in_z.append(z_loss.numpy())

        avg_in_rec = np.array(avg_in_rec)
        avg_in_rec = np.mean(avg_in_rec)
        avg_in_z = np.array(avg_in_z)
        avg_in_z = np.mean(avg_in_z)


        out_normal_subspace = solution
        out_bad_subspace = 1 - solution

        out_remain = particle * out_normal_subspace



        out_replace = avg_ins * out_bad_subspace

        out_candidate = out_remain + out_replace


        z = rae_model.encoder(out_candidate)
        out_candidate_rec = rae_model.decoder(z)

        rec_loss = tf.keras.losses.MeanSquaredError()(out_candidate,out_candidate_rec)
        rec_loss = rec_loss.numpy()

        # fitness = rec_loss / (avg_in_rec + 10*avg_in_z)
        fitness = rec_loss / avg_in_rec
        print(fitness)

        return -fitness

    fitness_function = fitness_func
    num_generations = num_generations
    num_parents_mating = num_parents_mating
    sol_per_pop = sol_per_pop
    num_genes = num_dims
    init_range_low = init_range_low
    init_range_high = init_range_high
    parent_selection_type = parent_selection_type
    K_tournament = K_tournament
    keep_parents = keep_parents
    space = [[0,1] for i in range(num_genes)]
    crossover_type = crossover_type
    mutation_type = mutation_type
    mutation_percent_genes = mutation_percent_genes

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       K_tournament = K_tournament,
                       keep_parents=keep_parents,
                    #    keep_elitism=5,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
#                        on_generation=on_generation,
                       gene_space = space)
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    return solution


def rae_detect_outliers(data,
                        rae_model,
                        num_dims
                        ):

    data_mean = []

    for i in range(data.shape[0]):
            
        sample = data[i,:].reshape([1,num_dims])

        z = rae_model.encoder(sample)
        reconstruction = rae_model.decoder(z)

        reconstruction_loss = tf.keras.losses.MeanSquaredError()(sample,reconstruction)
        
        data_mean.append(reconstruction_loss)
    
    data_mean = np.array(data_mean)
    data_std = np.std(data_mean)

    threshold = i_mean + 3*i_std

    classes = []

    for i in range(data.shape[0]):
            
        sample = data[i,:].reshape([1,num_dims])

        z = rae_model.encoder(sample)
        reconstruction = rae_model.decoder(z)

        reconstruction_loss = tf.keras.losses.MeanSquaredError()(sample,reconstruction)
        
        if reconstruction_loss > threshold:
            
            classes.append(1)
            
        else:
            
            classes.append(0)

    classes = np.array(classes)

    return classes

def vae_detect_outliers(data,
                        vae_model,
                        num_dims
                        ):

    data_mean = []

    for i in range(data.shape[0]):
            
        sample = data[i,:].reshape([1,num_dims])
        sample = sample.astype('float32')

        z_mean, z_log_var, z = vae_model.encoder(sample)
        reconstruction = vae_model.decoder(z)

        reconstruction_loss = tf.keras.losses.MeanSquaredError()(sample,reconstruction)
        
        data_mean.append(reconstruction_loss)
    
    data_mean = np.array(data_mean)
    i_mean = np.mean(data_mean)
    i_std = np.std(data_mean)
    

    threshold = i_mean + 3*i_std

    classes = []

    for i in range(data.shape[0]):
            
        sample = data[i,:].reshape([1,num_dims])
        sample = sample.astype('float32')

        z_mean, z_log_var, z = vae_model.encoder(sample)
        reconstruction = vae_model.decoder(z)

        reconstruction_loss = tf.keras.losses.MeanSquaredError()(sample,reconstruction)
        
        if reconstruction_loss > threshold:
            
            classes.append(1)
            
        else:
            
            classes.append(0)

    classes = np.array(classes)

    return classes


def csv_data_loader(name):

    if name == "aloi":

        csv_file = './dataverse_files/aloi-unsupervised-ad.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]

    if name == "annthyroid":

        csv_file = './dataverse_files/annthyroid-unsupervised-ad.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]
    
    if name == "breast-cancer":

        csv_file = './dataverse_files/breast-cancer-unsupervised-ad.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]

    if name == "kdd99":

        csv_file = './dataverse_files/kdd99-unsupervised-ad.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]

    if name == "letter":

        csv_file = './dataverse_files/letter-unsupervised-ad.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]
    
    if name == "pen-global":

        csv_file = './dataverse_files/pen-global-unsupervised-ad.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]

    if name == "pen-local":

        csv_file = './dataverse_files/pen-local-unsupervised-ad.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]

    if name == "satellite":

        csv_file = './dataverse_files/satellite-unsupervised-ad.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]
    
    if name == "shuttle":

        csv_file = './dataverse_files/shuttle-unsupervised-ad.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]

    if name == "fashion-TB":

        csv_file = './dataverse_files/fashion_mnist_TB_combined.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]

    if name == "fashion-TP":

        csv_file = './dataverse_files/fashion_mnist_TP_combined.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]

    if name == "mnist-06":

        csv_file = '../dataverse_files/mnist_06_combined.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]

    if name == "mnist-25":

        csv_file = '../dataverse_files/mnist_25_combined.csv'
        data = pd.read_csv(csv_file)
        last_column = data.iloc[:, -1].values
        last_column = np.where(last_column == "o", 0, 1)
        labels = last_column
        data = np.array(data)
        data_n = data[:,:-1]

    if name == "creditcard":
        
        csv_file = '../dataverse_files/creditcard.csv'
        data = pd.read_csv(csv_file)
        data = np.array(data)
        labels = data[:,-1]
        data_n = data[:,:-1]
        
    
    if name == "musk":
        
        csv_file = '../dataverse_files/musk.csv'
        data = pd.read_csv(csv_file)
        data = np.array(data)
        labels = data[:,-1]
        data_n = data[:,:-1]

    return data_n, labels
    

 