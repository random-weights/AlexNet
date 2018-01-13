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


batch_size = 128

gph = tf.Graph()
with gph.as_default():
    x = tf.placeholder('float',shape = [None,24,24,3])
    y_ = tf.placeholder('float',shape = [None,1000])

    ls_ker_dims = [[3,3,3,96],
                   [3,3,96,256],
                   [3,3,256,384],
                   [3,3,384,384],
                   [3,3,384,256],
                   [10*10*256,4096],
                   [4096,4096],
                   [4096,1000]]
    # initializing all kernels and biases including fc layer
    kernels = [0]*8
    biases = []*8
    for i in range(8):
        kernels[i] = tf.Variable(tf.random_normal(shape = ls_ker_dims[i],mean = 0.0, stddev=0.01))
        biases[i] = tf.Variable(1.0,shape = ls_ker_dims[i][-1])

    amap = tf.nn.relu(tf.nn.conv2d(x,kernels[0],strides = [1,1,1,1],padding = 'VALID'))
    layer1 = tf.nn.max_pool(amap,[1,3,3,1],[1,1,1,1],padding = "VALID")

    amap = tf.nn.relu(tf.nn.conv2d(layer1,kernels[1],[1,1,1,1],padding = "VALID"))
    layer2 = tf.nn.max_pool(amap,[1,3,3,1],[1,1,1,1],'VALID')

    layer3 = tf.nn.relu(tf.nn.conv2d(layer2,kernels[2],[1,1,1,1],'VALID'))
    layer4 = tf.nn.relu(tf.nn.conv2d(layer3,kernels[3],[1,1,1,1],'VALID'))
    layer5 = tf.nn.relu(tf.nn.conv2d(layer4,kernels[4],[1,1,1,1],'VALID'))

    out_dims = [batch_size,10,10,256]
    layer5_flat = tf.reshape(layer5,shape = [batch_size,out_dims[1]*out_dims[2]*out_dims[3]])

    layer6 = tf.relu(tf.matmul(layer5_flat,kernels[5]) + biases[5])
    layer7 = tf.relu(tf.matmul(layer6, kernels[6]) + biases[6])
    layer8 = tf.matmul(layer7, kernels[7]) + biases[7]

    soft_max = tf.nn.softmax(layer8)
    

with tf.Session(gph) as sess:
    sess.run(tf.global_variables_initializer())


