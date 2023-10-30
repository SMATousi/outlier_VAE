from utils import *
import wandb


wandb.init(
    # set the wandb project where this run will be logged
    project="RAE_VAE_OD_MagTest", name="num_dim_test"
    
    # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.02,
#     "architecture": "CNN",
#     "dataset": "CIFAR-100",
#     "epochs": 20,
#     }
)

##################### Testing the Magnitude of the Outliers #######################

mags = [1.5, 2, 4, 6, 8, 10, 12, 16]
ws = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
dims = [5, 10, 15, 20, 25, 30]
latent_dims = [2, 4, 6, 8, 10, 15]

for num_dimensions in range(5,100):

    print("Starting the stage: ", num_dimensions)


    rae_precision, rae_recall, rae_f1 = run_RAE(latent_dim = 8, z_loss_weight=0.1, reg_loss_weight=0.1, epochs=200, num_dimensions=num_dimensions, num_samples=5000)
    
    vae_precision, vae_recall, vae_f1 = run_VAE(latent_dim = 8, epochs=200, num_dimensions=num_dimensions, num_samples=5000)

    wandb.log({"Metrics/num_dim": num_dimensions, "Metrics/RAE-Precision": rae_precision, "Metrics/RAE-recall": rae_recall, "Metrics/RAE-F1": rae_f1, "Metrics/VAE-Precision": vae_precision, "Metrics/VAE-recall": vae_recall, "Metrics/VAE-F1": vae_f1})







