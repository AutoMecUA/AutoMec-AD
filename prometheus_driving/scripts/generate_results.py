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
from src.utils import SaveModel, SaveGraph , LoadModel
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
    parser.add_argument('-r', '--results_name', type=str, required=True,
                        help='folder name of the results')
    parser.add_argument('-fn', '--folder_name', type=str, required=True,
                        help='folder name where the model is stored')
    parser.add_argument('-mn', '--model_name', type=str, required=True,
                        help='model name')
    parser.add_argument('-batch_size', '--batch_size', default=256, type=int,
                        help='Batch size')
    parser.add_argument('-m', '--model', default='Nvidia_Model()', type=str,
                        help='Model to use [Nvidia_Model(), Rota_Model(), MobileNetV2(), InceptionV3(), MyVGG(), ResNet()]')
    parser.add_argument('-c', '--cuda', default=0, type=int,
                        help='Number of cuda device')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    # General Path
    files_path=os.environ.get('AUTOMEC_DATASETS')
    results_folder = args['results_name']
    # Image dataset paths
    dataset_path = f'{files_path}/datasets/{args["dataset_name"]}/'
    if not os.path.exists(dataset_path):
        print(f'{Fore.RED}The dataset does not exist{Style.RESET_ALL}')
        exit()

    columns = ['img_name','steering', 'velocity'] 
    df = pd.read_csv(os.path.join(dataset_path, 'driving_log.csv'), names = columns)

    del df["velocity"] # not in use, currently
    df.head()

    print(f'{Fore.BLUE}The dataset has {len(df)} images{Style.RESET_ALL}')

    device = f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu

    model_path = f'{files_path}/models/{args["folder_name"]}/{args["model_name"]}.pkl'
    model = eval(args['model']) # Instantiate model
    model= LoadModel(model_path,model,device)
    model.eval()

    dataset_test = Dataset(df,dataset_path,augmentation=False)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args['batch_size'], shuffle=True)

    # Init visualization of loss
    if args['visualize']: # Checks if the user wants to visualize the loss
        test_visualizer = ClassificationVisualizer('Test Images')
    # Init results
    results = SaveResults(results_folder, args["folder_name"], args["dataset_name"])

    for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_test), total=len(loader_test), desc=Fore.GREEN + 'Testing batches' +  Style.RESET_ALL):

        image_t = image_t.to(device=device, dtype=torch.float)
        label_t = label_t.to(device=device, dtype=torch.float).unsqueeze(1)

        # Apply the network to get the predicted ys
        label_t_predicted = model.forward(image_t)
        # Compute the error based on the predictions
        for idx in range(len(label_t_predicted)):
            #print(label_t_predicted[idx].data.item() - label_t[idx].data.item())
            steering_error = abs(label_t_predicted[idx].data.item() - label_t[idx].data.item())
            results.updateCSV(steering_error, 0)
            results.step()


        if args['visualize']:
            test_visualizer.draw(image_t, label_t, label_t_predicted)

    results.saveCSV()
    results.saveErrorsFig()


if __name__ == '__main__':
    main()