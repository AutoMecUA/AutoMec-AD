#!/usr/bin/python3

"""
    Script for creating statistics of a dataset, calculating the rgb mean and the rgb std.
    The dataset is loaded from the path specified in the parameter 'dataset_name'.
    The statistics are saved in the default path to the Automec folder set in the Environment variable.
"""

# Imports 
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style
import numpy as np
import cv2
import yaml

#  custom imports
from src.dataset import Dataset


# Main code
def main():
    ########################################
    # Initialization                       #
    ########################################
    parser = argparse.ArgumentParser(description='Data Collector')
    parser.add_argument('-d', '--dataset_name', type=str, required=True,
                        help='folder name of the dataset')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    # Dataset Path
    files_path=os.environ.get('AUTOMEC_DATASETS')
    dataset_path = f'{files_path}/datasets/{args["dataset_name"]}/'
    if not os.path.exists(dataset_path):
        print(f'{Fore.RED}The dataset does not exist{Style.RESET_ALL}')
        exit()

    # Loads the steering and velocity values from the driving_log.csv file to create the dataset class
    columns = ['img_name','steering', 'velocity'] 
    df = pd.read_csv(os.path.join(dataset_path, 'driving_log.csv'), names = columns)

    del df["velocity"] # not in use, currently
    df.head()

    print(f'{Fore.BLUE}The dataset has {len(df)} images{Style.RESET_ALL}')

    # Create dataset object
    dataset = Dataset(df,dataset_path)

    # Read YAML file with dataset info
    with open(dataset_path + "info.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    
    # Create statistics dictionary
    config['statistics'] = {'B' : {'max'  : np.empty((len(dataset))),
                                    'min'  : np.empty((len(dataset))),
                                    'mean' : np.empty((len(dataset))),
                                    'std'  : np.empty((len(dataset)))},
                            'G' : {'max'  : np.empty((len(dataset))),
                                    'min'  : np.empty((len(dataset))),
                                    'mean' : np.empty((len(dataset))),
                                    'std'  : np.empty((len(dataset)))},
                            'R' : {'max'  : np.empty((len(dataset))),
                                    'min'  : np.empty((len(dataset))),
                                    'mean' : np.empty((len(dataset))),
                                    'std'  : np.empty((len(dataset)))}}
    
    ########################################
    # Process each frame                   #
    ########################################
    for idx in range(len(dataset)):
        
        print(f'creating stats of frame {idx}')
        
        # Load RGB image
        cv_image = cv2.imread(f'{dataset.image_filenames_original[idx]}', cv2.IMREAD_UNCHANGED)
        
        blue_image = cv_image[:,:,0]/255
        green_image = cv_image[:,:,1]/255
        red_image = cv_image[:,:,2]/255
        
        ## B channel
        config['statistics']['B']['max'][idx] = np.max(blue_image)
        config['statistics']['B']['min'][idx] = np.min(blue_image)
        config['statistics']['B']['mean'][idx] = np.mean(blue_image)
        config['statistics']['B']['std'][idx] = np.std(blue_image)
        
        ## G channel
        config['statistics']['G']['max'][idx] = np.max(green_image)
        config['statistics']['G']['min'][idx] = np.min(green_image)
        config['statistics']['G']['mean'][idx] = np.mean(green_image)
        config['statistics']['G']['std'][idx] = np.std(green_image)
        
        ## R channel
        config['statistics']['R']['max'][idx] = np.max(red_image)
        config['statistics']['R']['min'][idx] = np.min(red_image)
        config['statistics']['R']['mean'][idx] = np.mean(red_image)
        config['statistics']['R']['std'][idx] = np.std(red_image)
                    
    ########################################
    # Save statistics                      #
    ########################################
    config['statistics']['B']['max']  = round(float(np.mean(config['statistics']['B']['max'])),5)
    config['statistics']['B']['min']  = round(float(np.mean(config['statistics']['B']['min'])),5)
    config['statistics']['B']['mean'] = round(float(np.mean(config['statistics']['B']['mean'])),5)
    config['statistics']['B']['std']  = round(float(np.mean(config['statistics']['B']['std'])),5)
    
    config['statistics']['G']['max']  = round(float(np.mean(config['statistics']['G']['max'])),5)
    config['statistics']['G']['min']  = round(float(np.mean(config['statistics']['G']['min'])),5)
    config['statistics']['G']['mean'] = round(float(np.mean(config['statistics']['G']['mean'])),5)
    config['statistics']['G']['std']  = round(float(np.mean(config['statistics']['G']['std'])),5)

    config['statistics']['R']['max']  = round(float(np.mean(config['statistics']['R']['max'])),5)
    config['statistics']['R']['min']  = round(float(np.mean(config['statistics']['R']['min'])),5)
    config['statistics']['R']['mean'] = round(float(np.mean(config['statistics']['R']['mean'])),5)
    config['statistics']['R']['std']  = round(float(np.mean(config['statistics']['R']['std'])),5)
    
    with open(f'{dataset_path}/info.yaml', 'w') as f:
            yaml.dump(config, f)


if __name__ == '__main__':
    main()