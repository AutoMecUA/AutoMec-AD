#!/usr/bin/python3

# Imports 
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style
import torch
import glob

#  custom imports
from models.deeplabv3 import createDeepLabv3
from src.dataset_semantic import DatasetSemantic
from src.results import SaveResults
from src.utils import LoadModel
from src.visualization import ClassificationVisualizer


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
    parser.add_argument('-mn', '--model_name', type=str, required=True,
                        help='folder name where the model is stored')
    parser.add_argument('-batch_size', '--batch_size', default=256, type=int,
                        help='Batch size')
    parser.add_argument('-nw', '--num_workers', type=int, default=0, 
                        help='How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('-pff', '--prefetch_factor', type=int, default=2, 
                        help='Number of batches loaded in advance by each worker')
    parser.add_argument('-m', '--model', default='Nvidia_Model()', type=str,
                        help='Model to use [createDeepLabv3(outputchannels=1)]')
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

    dataset_RGB = glob.glob(dataset_path + '/leftImg8bit/train/*/*.png')
    dataset_seg = glob.glob(dataset_path + '/gtFine/train/*/*labelIds.png')
    dataset = list(zip(dataset_RGB, dataset_seg))

    device = f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu

    model_path = f'{files_path}/models/{args["model_name"]}/{args["model_name"]}.pkl'
    model = eval(args['model']) # Instantiate model
    model= LoadModel(model_path,model,device)
    model.eval()

    dataset_test = DatasetSemantic(dataset,augmentation=False)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args['batch_size'], shuffle=True , num_workers=args['num_workers'] , prefetch_factor=args['prefetch_factor'])

    # Init visualization of loss
    if args['visualize']: # Checks if the user wants to visualize the loss
        test_visualizer = ClassificationVisualizer('Test Images')
    # Init results
    results = SaveResults(results_folder, args["model_name"], args["dataset_name"])
    label_predicted = []
    label = []
    for batch_idx, (image_t, masks) in tqdm(enumerate(loader_test), total=len(loader_test), desc=Fore.GREEN + 'Testing batches' +  Style.RESET_ALL):

        image_t = image_t.to(device=device, dtype=torch.float)
        masks = masks.to(device=device, dtype=torch.float)

        # Apply the network to get the predicted ys
        masks_predicted = model.forward(image_t)
        # Compute the error based on the predictions
        if args['visualize']:
            test_visualizer.draw(image_t, masks, masks_predicted['out'])
    
    results.updateCSV(label_predicted, label , len(label_predicted))
    results.saveCSV()
    results.saveErrorsFig()


if __name__ == '__main__':
    main()