# This script will process all json files in a folder and turn them into a readable cvs
import os
import pandas as pd


from csv import writer


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def getRowFromJson(path):
    pdObj = pd.read_json(
        path, orient="index")
    listOfVal0 = pdObj.iloc[:, 0].tolist()
    #print('Here comes : ', listOfVal0[0: 3:2])
    return listOfVal0[0:3:2]


def getFileNames(folderpath):
    filenames = []
    folder = os.fsencode(folderpath)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        # whatever file types you're using...
        if filename.endswith(('.json')):
            filenames.append(filename)

    filenames.sort()  # now you have the filenames and can do something with them
    return filenames


folderpath = '../hive_fast_driving/'
filenames = getFileNames(folderpath)

for name in filenames:
    path = '../hive_fast_driving/'+name
    if name[1] != '_':
        row = getRowFromJson(path)
        append_list_as_row('driving-log.csv', row)
