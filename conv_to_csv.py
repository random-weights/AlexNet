import numpy as np
import pickle
import csv

import matplotlib.pyplot as plt
from scipy.misc import imshow
'''

This file is to experiment and verify the code


'''


def unpickle(fname):
    with open(fname,'rb') as fo:
        dict = pickle.load(fo)

    return dict

def main():

    file_str = "data/training/train_data_batch_"
    ls_fnames = [file_str+str(i) for i in range(1,11)]
    fh = open("D:/training.csv", 'a', newline='')

    for file_count in range(10):

        wr = csv.writer(fh,dialect='excel')

        val_data = unpickle(ls_fnames[file_count])
        print(val_data.keys())
        size = len(val_data['labels'])
        for i in range(size):
            img_data = list(val_data['data'][i])
            label = val_data['labels'][i]
            joined = [label]+img_data
            wr.writerow(joined)
            print("\r Line: ".format(i)+str(i),end  = '')
        print("\n")
        print("\r Finished File: ".format(file_count)+str(file_count),end = '')

    fh.close()


main()