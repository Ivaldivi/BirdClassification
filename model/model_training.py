import torch
import os
import pandas as pd 

def gather_labels(starting_directory): 
    """Get all of the labels associated with a training set. The starting directory should contain
    folders that contain pictures for each of the categories that will be used in model training."""

    all_entries = os.listdir(starting_directory)
    labels = []
    for entry in all_entries: 
        full_path = os.path.join(starting_directory, entry)
        if os.path.isdir(full_path): 
            labels.append(entry)
    print(f"There are {len(labels)} labels")
    return labels

def gather_pictures(data_directory):
    """Gather the images and the labels to be used to create the model and store them in a pandas 
    dataframe"""
    data = []
    for root, dirs, files in os.walk(data_directory):
        label = root.split('/')[-1]
        for f in files:
            picture_path = root + '/' + f
            picture_name = f.split('.jpg')[0]
            picture_id = f.split('_')[-1][0:-4]
            data.append([label, picture_path, picture_name, picture_id])
    columns = ['Label', 'Picture_Path', 'Picture_Name', 'Picture_id']
    return pd.DataFrame(data, columns = columns)

def main():
    labels = gather_labels('../data')
    print(labels)
    df = gather_pictures('../data')
    print(df)

if __name__ == '__main__':
    main()