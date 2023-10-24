from utils import *
import wandb


wandb.init(
    # set the wandb project where this run will be logged
    project="RAE_OD_MagTest", name="Wtest, ep=200 num_samples = 5000"
    
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

for d in dims:

    print("Starting the stage: ", d)


    precision, recall, f1 = run_RAE(z_loss_weight=0.1, reg_loss_weight=0.1, epochs=200, num_dimensions=d, num_samples=5000)

    wandb.log({"Metrics/d": d, "Metrics/Precision": precision, "Metrics/recall": recall, "Metrics/F1": f1})







