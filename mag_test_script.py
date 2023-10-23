from utils import *



##################### Testing the Magnitude of the Outliers #######################


for mag in range(1,20):

    print("Starting the stage: ", mag)


    precision, recall, f1 = run_RAE(outlier_magnitude_factor = mag)


    


