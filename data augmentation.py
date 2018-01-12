import numpy as np
import matplotlib.pyplot as plt


def rand_crop(img_data):
    a = int(np.random.rand()*(64-52))
    ls_crop = []
    for c in range(0,3):
        for i in range(a,a+52):
            for j in range(a,a+52):
                ls_crop.append(img_data[c][i][j])

    img_crop = np.array(ls_crop).reshape(3,52,52)
    return img_crop


def mirror(img_data):
    ls_mirror = []
    for c in range(3):
        for i in range(len(img_data[c])):
            ls_mirror.append(list(reversed(img_data[c][i])))
    img_mirror = np.array(ls_mirror).reshape(3,52,52)
    return img_mirror


def main():
    with open("data/validation.csv") as fh:
        row = fh.readline()
        split_str = row.split(',')
        ls_img = []
        for i in range(1, len(split_str)):
            ls_img.append(int(split_str[i]))

        img_data = np.array(ls_img).reshape(3, 64, 64)

    img_crop = rand_crop(img_data)
    img_mirror = mirror(img_crop)
    plt.imshow(img_crop[0])
    plt.figure()
    plt.imshow(img_mirror[0])
    plt.show()


main()


