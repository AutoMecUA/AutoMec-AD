from csv import writer
from datetime import datetime

import csv

with open('20_02_21__16_15_20.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
    
        print(row[20399])
        #print(row[20402])

    
