#!/usr/bin/env python3
import os
import yaml
import pandas as pd
import shutil
import matplotlib.pyplot as plt

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
        
        if not os.path.exists(self.output_folder):
            print(f'Creating folder {self.output_folder}')
            os.makedirs(self.output_folder)  # Create the new folder
        elif overwrite:
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
        
        self.frame_idx = 0 # make sure to save as 00000
        self.csv = pd.DataFrame(columns=('frame', 'position_error (m)', 'rotation_error (rads)'))
        
        print('SaveResults initialized properly')

    def updateCSV(self, steering_error, velocity_error):
        row = pd.DataFrame({'frame' : f'{self.frame_idx:05d}', 
                'steering_error (rads)' : steering_error,
                'velocity_error (m/s)' : velocity_error},index=[0])
        
        #self.csv = self.csv.append(row, ignore_index=True)  
        self.csv = pd.concat([self.csv , row],ignore_index=True)
        
        
    def saveCSV(self):
        # save averages values in the last row
        mean_row = pd.DataFrame({'frame'                 : 'mean_values', 
                    'steering_error (rads)'    : self.csv.mean(axis=0).loc["steering_error (rads)"],
                    'velocity_error (m/s)' : self.csv.mean(axis=0).loc["velocity_error (m/s)"]},index=[0])
        
        
        median_row = pd.DataFrame({'frame'                 : 'median_values', 
                      'steering_error (rads)'    : self.csv.median(axis=0).loc["steering_error (rads)"],
                      'velocity_error (m/s)' : self.csv.median(axis=0).loc["velocity_error (m/s)"]},index=[0])
        
        #self.csv = self.csv.append(mean_row, ignore_index=True)  
        self.csv = pd.concat([self.csv , mean_row],ignore_index=True) 
        
        
        print(self.csv)
        self.csv.to_csv(f'{self.output_folder}/errors.csv', index=False, float_format='%.5f')

    def saveErrorsFig(self):
        frames_array = self.csv.iloc[:-2]['frame'].to_numpy().astype(int)
        
        pos_error_array = self.csv.iloc[:-2]['steering_error (rads)'].to_numpy()
        rot_error_array = self.csv.iloc[:-2]['velocity_error (m/s)'].to_numpy()
        
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.suptitle('steering and velocity errors')
        ax1.plot(frames_array, pos_error_array, 'cyan',  label='steering error')
        ax2.plot(frames_array, rot_error_array, 'navy', label='velocity error')
        ax2.set_xlabel('frame idx')
        ax2.set_ylabel('[m/s]')
        ax1.set_ylabel('[rads]')
        ax1.legend()
        ax2.legend()
        plt.savefig(f'{self.output_folder}/errors.png')
        
        
    def step(self):
        self.frame_idx+=1
        