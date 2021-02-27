#!/usr/bin/env python

from csv import writer
from datetime import datetime

import csv

with open('good_train_data/27_02_21__13_58_02.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    i=0
    for row in csv_reader:
        print(row[len(row)-1])
        i+=1
        if i==100:
            break

    
