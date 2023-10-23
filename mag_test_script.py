from utils import *
import wandb


wandb.init(
    # set the wandb project where this run will be logged
    project="RAE_OD_MagTest", name="Magnitude Test"
    
    # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.02,
#     "architecture": "CNN",
#     "dataset": "CIFAR-100",
#     "epochs": 20,
#     }
)

##################### Testing the Magnitude of the Outliers #######################

mags = [1.5, 2, 4, 8, 16]

for mag in mags:

    print("Starting the stage: ", mag)


    precision, recall, f1 = run_RAE(outlier_magnitude_factor = mag)

    wandb.log({"Metrics/Precision": precision, "Metrics/recall": recall, "Metrics/F1": f1})

    





