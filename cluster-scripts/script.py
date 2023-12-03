from genetic_utils import *
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb


parser = argparse.ArgumentParser(description="A script with argparse options")


parser.add_argument("--runname", type=str, required=True)
parser.add_argument("--projectname", type=str, required=True)
parser.add_argument("--modelname", type=str, required=True)
parser.add_argument("--datasetname", type=str, required=True)
# parser.add_argument("--savingstep", type=int, default=10)
parser.add_argument("--epochs", type=int, default=100)
# parser.add_argument("--nottest", help="Enable verbose mode", action="store_true")


args = parser.parse_args()


arg_datasetname = args.datasetname
arg_epochs = args.epochs
arg_runname = args.runname
arg_projectname = args.projectname
arg_modelname = args.modelname



wandb.init(
    # set the wandb project where this run will be logged
    project=arg_projectname, name=arg_runname
    
    # track hyperparameters and run metadata
    # config={
    # "learning_rate": 0.02,
    # "architecture": "CNN",
    # "dataset": "CIFAR-100",
    # "epochs": 20,
    # }
)

std_ks = [1, 1.5, 2, 2.5, 3]

# Loading the data

data, lables = csv_data_loader(arg_datasetname)

num_dims = data.shape[1]


# Training the model



if arg_modelname == "VAE":

    model, history = train_VAE(data, num_dims=num_dims, latent_dim=1, hidden_layer_n=[512, 256, 128] )
    print("######################## Training Done ######################")


if arg_modelname == "RAE":

    model, history = train_RAE(data, num_dims=num_dims, latent_dim=2, hidden_layer_n=[512, 256, 128] )
    print("######################## Training Done ######################")


for std_k in std_ks:

    if arg_modelname == "VAE":
        classes = vae_detect_outliers(data, model, num_dims, std_k=std_k)

    if arg_modelname == "RAE":
        classes = rae_detect_outliers(data, model, num_dims, std_k=std_k)

    true_labels = 1 - lables
    predicted_labels = 1 - classes

    precision = precision_score(true_labels, predicted_labels)

    # Calculate recall
    recall = recall_score(true_labels, predicted_labels)

    # Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    wandb.log({"Metrics/Precision": precision, "Metrics/Recall": recall, "Metrics/F1 Score": f1, "Metrics/std_k": std_k})