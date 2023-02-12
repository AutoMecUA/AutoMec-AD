#!/usr/bin/python3

# Imports 
import argparse
import os
from statistics import mean
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from colorama import Fore, Style
import torch
from torch.nn import MSELoss
import yaml

#  custom imports
from src.dataset import Dataset
from src.results import SaveResults
from models.cnn_nvidia import Nvidia_Model
from models.cnn_rota import Rota_Model
from models.mobilenetv2 import MobileNetV2
from models.inceptionV3 import InceptionV3
from src.utils import SaveModel, SaveGraph
from src.visualization import DataVisualizer, ClassificationVisualizer


# Main code
def main():
    ########################################
    # Initialization                       #
    ########################################
    parser = argparse.ArgumentParser(description='Data Collector')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Visualize the loss')
    parser.add_argument('-d', '--dataset_name', type=str, required=True,
                        help='folder name of the dataset')
    parser.add_argument('-r', '--dataset_name', type=str, required=True,
                        help='folder name of the dataset')
    parser.add_argument('-fn', '--folder_name', type=str, required=True,
                        help='folder name where the model is stored')
    parser.add_argument('-mn', '--model_name', type=str, required=True,
                        help='model name')
    parser.add_argument('-c', '--cuda', default=0, type=int,
                        help='Number of cuda device')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    # General Path
    files_path=os.environ.get('AUTOMEC_DATASETS')
    #files_path=f'/home/andre/catkin_ws/src/AutoMec-AD/prometheus_driving/data/'
    # Image dataset paths
    dataset_path = f'{files_path}/datasets/{args["dataset_name"]}/'

    dataset_path = f'{files_path}/datasets/{args["dataset_name"]}/'
    columns = ['img_name','steering', 'velocity'] 
    df = pd.read_csv(os.path.join(dataset_path, 'driving_log.csv'), names = columns)

    del df["velocity"] # not in use, currently
    df.head()

    print(f'{Fore.BLUE}The dataset has {len(df)} images{Style.RESET_ALL}')

    device = f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu

    model = eval(args['model']) # Instantiate model
    model.to(device) # move the model variable to the gpu if one exists

    dataset_test = Dataset(df,dataset_path,augmentation=False)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args['batch_size'], shuffle=True)

    # Init visualization of loss
    if args['visualize']: # Checks if the user wants to visualize the loss
        test_visualizer = ClassificationVisualizer('Test Images')

    results = SaveResults(results_folder, args["model_folder"], args["testing_set"], args['overwrite'])

    for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_test), total=len(loader_test), desc=Fore.GREEN + 'Testing batches' +  Style.RESET_ALL):

        image_t = image_t.to(device=device, dtype=torch.float)
        label_t = label_t.to(device=device, dtype=torch.float).unsqueeze(1)

        # Apply the network to get the predicted ys
        label_t_predicted = model.forward(image_t)
        # Compute the error based on the predictions
        for idx in range(len(label_t_predicted)):
            print(label_t_predicted[idx].data.item() - label_t[idx].data.item())
            error = abs(label_t_predicted[idx].data.item() - label_t[idx].data.item())
            results.updateCSV(pos_error, rot_error)
            results.step()


        if args['visualize']:
            test_visualizer.draw(image_t, label_t, label_t_predicted)

    results.saveCSV()
    results.saveErrorsFig()
