#!/usr/bin/env python3
import math
import os
import yaml
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from colorama import Fore, Style

from utils import write_transformation
from colorama import Fore
from datetime import datetime


class SaveResults():
    """
    class to save results
    """
    
    def __init__(self, output, model_path, seq_path, overwrite=False):
        
        # attribute initializer
        path=os.environ.get("AUTOMEC_DATASETS")
        self.output_folder = f'{path}/results/{output}'
        self.model_path = model_path
        self.seq_path = seq_path

        if os.path.exists(self.output_folder):
            print(Fore.YELLOW + f'Results folder already exists! Do you want to overwrite?' + Style.RESET_ALL)
            ans = input(Fore.YELLOW + "Y" + Style.RESET_ALL + "ES/" + Fore.YELLOW + "n" + Style.RESET_ALL + "o: ") # Asks the user if they want to resume training
        
        if not os.path.exists(self.output_folder):
            print(f'Creating folder {self.output_folder}')
            os.makedirs(self.output_folder)  # Create the new folder
        elif os.path.exists(self.output_folder) and ans.lower() in ['' , 'y' , 'yes']:
            print(f'Overwriting folder {self.output_folder}')
            shutil.rmtree(self.output_folder)
            os.makedirs(self.output_folder)  # Create the new folder
        else:
            print(f'{Fore.RED} {self.output_folder} already exists... Aborting SaveResults initialization! {Fore.RESET}')
            exit(0)
        
        
        
        dt_now = datetime.now() # current date and time
        config = {'user'       : os.environ["USER"],
                  'date'       : dt_now.strftime("%d/%m/%Y, %H:%M:%S"),
                  'model_path' : self.model_path,
                  'seq_path'   : self.seq_path}
        
        with open(f'{self.output_folder}/config.yaml', 'w') as file:
            yaml.dump(config, file)
        
        self.csv = pd.DataFrame(columns=('frame', 'steering_difference (rads)', 'steering_SE (rads²)' , 'steering_RSE (rads)'))
        
        print('SaveResults initialized properly')

    def updateCSV(self, steering_predicted, steering_labeled , num_frames):
        steering_se = np.square(np.subtract(steering_predicted, steering_labeled))
        self.steering_mse = sklearn.metrics.mean_squared_error(steering_labeled, steering_predicted, squared=True)
        self.steering_rmse = math.sqrt(self.steering_mse) 
        for frame_idx in range(num_frames):
            steering_error = abs(steering_predicted[frame_idx] - steering_labeled[frame_idx])
            steering_rse = math.sqrt(steering_se[frame_idx]) 
            row = pd.DataFrame({'frame' : f'{frame_idx:05d}', 
                                'steering_difference (rads)' : steering_error,
                                'steering_SE (rads²)' : steering_se[frame_idx],
                                'steering_RSE (rads)': steering_rse},index=[0])
            self.csv = pd.concat([self.csv , row],ignore_index=True)
        
        
    def saveCSV(self):
        # save averages values in the last row
        mean_row = pd.DataFrame({'frame'                 : 'mean_values', 
                                 'steering_difference (rads)'    : self.csv.mean(axis=0).loc['steering_difference (rads)'],
                                 'steering_SE (rads²)' : self.steering_mse,
                                 'steering_RSE (rads)': self.steering_rmse},index=[0])
        
        
        median_row = pd.DataFrame({'frame'                 : 'median_values', 
                      'steering_difference (rads)'    : self.csv.median(axis=0).loc['steering_difference (rads)'],
                      'steering_SE (rads²)' : self.csv.median(axis=0).loc['steering_SE (rads²)'],
                      'steering_RSE (rads)': self.csv.median(axis=0).loc['steering_RSE (rads)']},index=[0])
        
        #self.csv = self.csv.append(mean_row, ignore_index=True)  
        self.csv = pd.concat([self.csv , mean_row],ignore_index=True) 
        
        
        print(self.csv)
        self.csv.to_csv(f'{self.output_folder}/errors.csv', index=False, float_format='%.5f')

    def saveErrorsFig(self):
        frames_array = self.csv.iloc[:-2]['frame'].to_numpy().astype(int)
        
        diff_error_array = self.csv.iloc[:-2]['steering_difference (rads)'].to_numpy()
        se_error_array = self.csv.iloc[:-2]['steering_SE (rads²)'].to_numpy()
        rse_error_array = self.csv.iloc[:-2]['steering_RSE (rads)'].to_numpy()

        
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        ax1.plot(frames_array, diff_error_array, '#a2cffe',  label='Difference error')
        ax1.plot(frames_array, np.full((len(frames_array),1),self.csv.mean(axis=0).loc['steering_difference (rads)']), '#0343df', label=f'Mean error')
        ax2.plot(frames_array, se_error_array, '#8e82fe', label='Square error')
        ax2.plot(frames_array, np.full((len(frames_array),1),self.steering_mse), 'navy', label=f'Mean square error')
        ax3.plot(frames_array, rse_error_array, '#047495', label='Root square error')
        ax3.plot(frames_array, np.full((len(frames_array),1),self.steering_rmse), '#0504aa', label=f'Root Mean square error')
        ax3.set_xlabel('frame idx')
        ax3.set_ylabel('[rads]')
        ax2.set_ylabel('[rads²]')
        ax1.set_ylabel('[rads]')
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.savefig(f'{self.output_folder}/errors.png')
        
        