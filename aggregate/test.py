import os

data_dir = "../datasets/H_V"
for file in os.listdir(data_dir):
    if file.endswith('.data'):
        print(file)