import pickle
import csv


'''

This file is to experiment and verify the code


'''


def unpickle(fname):
    with open(fname,'rb') as fo:
        dict = pickle.load(fo)

    return dict

def main():

    file_str = "data/val_data"
    fh_xdata = open("data/x_val.csv", 'a', newline='')
    fh_ydata = open("data/y_val.csv",'a',newline = '')

    wr_x = csv.writer(fh_xdata,dialect='excel')
    wr_y = csv.writer(fh_ydata, dialect='excel')
    val_data = unpickle(file_str)
    print(val_data.keys())
    size = len(val_data['labels'])
    for i in range(size):
        img_data = list(val_data['data'][i])
        label = val_data['labels'][i]
        wr_x.writerow(img_data)
        wr_y.writerow([label])
        print("\r Line: ".format(i)+str(i),end  = '')

    fh_xdata.close()
    fh_ydata.close()


main()