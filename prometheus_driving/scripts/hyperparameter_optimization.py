#!/usr/bin/python3

# Imports 
import os
from colorama import Fore, Style
import argparse
import sys


# Main code
def main():
    ########################################
    # Initialization                       #
    ########################################
    parser = argparse.ArgumentParser(description='Data Collector')
    parser.add_argument('-n_epochs', '--max_epoch', default=50, type=int,
                        help='Maximum number of epochs')
    parser.add_argument('-batch_size', '--batch_size', default=256, type=int,
                        help='Batch size')
    parser.add_argument('-c', '--cuda', default=0, type=int,
                        help='Number of cuda device')
    parser.add_argument('-d', '--dataset_name', type=str, required=True,
                        help='folder name of the dataset')
 
    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    # Default hyper parameters
    wd_default = 0
    dropout_default = 0
    lr_default = 0.001
    lr_step_size_default = 20

    # Define the hyper parameters to optimize
    hyperparameters = {
        'lr': [0.001, 0.0005 , 0.0001],
        'lr_step_size': [10, 20,30,40],
        'dropout': [0.1, 0.2, 0.3 ,0.4, 0.5],
        'wd': [0.0001, 0.001, 0.01, 0.1],
    }

    for lr_test in hyperparameters['lr']:
        wd = wd_default
        dropout = dropout_default
        lr_step_size = lr_step_size_default
        lr = lr_test
        print(Fore.BLUE + f'Running with lr={lr}, lr_step_size={lr_step_size}, dropout={dropout}, wd={wd}' + Style.RESET_ALL)
        os.system(f'rosrun prometheus_driving model_train.py -d {args["dataset_name"]} -fn lr_{lr} -m "LSTM(dropout={dropout})" -n_epochs {args["max_epoch"]} -batch_size {args["batch_size"]} -loss_f "MSELoss()" -nw 4 -lr {lr} -lr_step_size {lr_step_size} -wd {wd} -c {args["cuda"]}')

    for lr_step_size_test in hyperparameters['lr_step_size']:
        wd = wd_default
        dropout = dropout_default
        lr_step_size = lr_step_size_test
        lr = lr_default
        print(Fore.BLUE + f'Running with lr={lr}, lr_step_size={lr_step_size}, dropout={dropout}, wd={wd}' + Style.RESET_ALL)
        os.system(f'rosrun prometheus_driving model_train.py -d {args["dataset_name"]} -fn lr_step_size_{lr_step_size} -m "LSTM(dropout={dropout})" -n_epochs {args["max_epoch"]} -batch_size {args["batch_size"]} -loss_f "MSELoss()" -nw 4 -lr {lr} -lr_step_size {lr_step_size} -wd {wd} -c {args["cuda"]}')
    
    for dropout_test in hyperparameters['dropout']:
        wd = wd_default
        dropout = dropout_test
        lr_step_size = lr_step_size_default
        lr = lr_default
        print(Fore.BLUE + f'Running with lr={lr}, lr_step_size={lr_step_size}, dropout={dropout}, wd={wd}' + Style.RESET_ALL)
        os.system(f'rosrun prometheus_driving model_train.py -d {args["dataset_name"]} -fn dropout_{dropout} -m "LSTM(dropout={dropout})" -n_epochs {args["max_epoch"]} -batch_size {args["batch_size"]} -loss_f "MSELoss()" -nw 4 -lr {lr} -lr_step_size {lr_step_size} -wd {wd} -c {args["cuda"]}')

    for wd_test in hyperparameters['wd']:
        wd = wd_test
        dropout = dropout_default
        lr_step_size = lr_step_size_default
        lr = lr_default
        print(Fore.BLUE + f'Running with lr={lr}, lr_step_size={lr_step_size}, dropout={dropout}, wd={wd}' + Style.RESET_ALL)
        os.system(f'rosrun prometheus_driving model_train.py -d {args["dataset_name"]} -fn wd_{wd} -m "LSTM(dropout={dropout})" -n_epochs {args["max_epoch"]} -batch_size {args["batch_size"]} -loss_f "MSELoss()" -nw 4 -lr {lr} -lr_step_size {lr_step_size} -wd {wd} -c {args["cuda"]}')
    print(Fore.GREEN + 'Finished training all the models' + Style.RESET_ALL)
if __name__ == '__main__':
    main()