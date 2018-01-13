import tensorflow as tf
import numpy as np

class Data():

    def setFilePaths(self,x_path,y_path):
        self.x_path = x_path
        self.y_path = y_path

    def getSize(self):
        self.size = sum(1 for line in open(self.y_path))
        return self.size

    def getRandomData(self,batch_size = None):
        if batch_size is None:
            bsize = 128
        else:
            bsize = batch_size

        rand_indices = np.random.choice(self.size,bsize,replace = False)
        x_batch = []
        with open(self.x_path,'r') as fh:
            ptr = 0
            for item in rand_indices:
                for i in range(item-ptr):
                    fh.readline()
                    ptr += 1
                row = fh.readline().split(',')
                row = [int(x) for x in row]
                x_batch.append(row)
            self.x_batch = np.array(x_batch).astype(int)

        y_batch = []
        with open(self.y_path,'r') as fh:
            ptr = 0
            for item in rand_indices:
                for i in range(item-ptr):
                    fh.readline()
                    ptr += 1
                row = fh.readline().split(',')
                row = [int(y) for y in row]
                y_batch.append(row)
            self.y_batch = np.array(y_batch).astype(int)

    def getAllData(self):
        x_data = []
        with open(self.x_path, 'r') as fh:
            for _ in range(self.size):
                row = fh.readline().split(',')
                row = [int(x) for x in row]
                x_data.append(row)
        self.x_data = np.array(x_data).astype(int)

        y_data = []
        with open(self.y_path, 'r') as fh:
            for _ in range(self.size):
                row = fh.readline().split(',')
                row = [int(y) for y in row]
                y_data.append(row)
        self.y_data = np.array(y_data).astype(int)

gph = tf.Graph()
with gph.as_default():
    pass


with tf.Session(gph) as sess:
    sess.run(tf.global_variables_initializer())


