import tensorflow as tf
import numpy as np

class Data():

    def __init__(self):
        self.batch_size = 128

    def setFilePaths(self,data_path):
        self.data_path = data_path
        print("Data Path set")

    def getRandomData(self,batch_size = None):
        if not (batch_size is None):
            self.batch_size = batch_size

        rand_index = int(np.random.sample()*(1281166 - self.batch_size))
        file_numb = int(rand_index//100000)
        rand_index = rand_index % 100000

        x_fname = self.data_path+"x_train_"+str(file_numb)+".csv"
        y_fname = self.data_path+"y_train_"+str(file_numb)+".csv"

        x_batch = []
        with open(x_fname,'r') as fh:
            for _ in range(rand_index):
                fh.readline()
            for i in range(self.batch_size):
                row = fh.readline().split(',')
                row = [int(x) for x in row]
                x_batch.append(row)
            self.x_batch = np.array(x_batch).astype(int)

        y_batch = []
        with open(y_fname,'r') as fh:
            for _ in range(rand_index):
                fh.readline()
            for i in range(self.batch_size):
                row = fh.readline().split(',')
                row = [int(y) for y in row]
                y_batch.append(row)
            self.y_batch = np.array(y_batch).astype(int)

    def getAllData(self):
        x_data = pd.read_csv(self.x_path,sep = ',',header = None)
        self.x_data = np.array(x_data).astype(int)
        y_data = pd.read_csv(self.y_path, sep=',', header=None)
        self.y_data = np.array(y_data).astype(int)

    def rand_crop(self,img_data):
        '''
        Expects img_data in the shape 3,32,32 as np array
        :param img_data:
        :return: cropped image of shape 3,24,24 as np array
        '''
        a = int(np.random.rand() * (32 - 24))
        ls_crop = []
        for c in range(0, 3):
            for i in range(a, a + 24):
                for j in range(a, a + 24):
                    ls_crop.append(img_data[c][i][j])

        img_crop = np.array(ls_crop).reshape(3, 24, 24)
        return img_crop

    def mirror(self,img_data):
        '''
        Expects img_data in shape 3,24,24 as np array
        :param img_data:
        :return: mirrored image along vertical axis as 3,24,24 np array
        '''
        ls_mirror = []
        for c in range(3):
            for i in range(len(img_data[c])):
                ls_mirror.append(list(reversed(img_data[c][i])))
        img_mirror = np.array(ls_mirror).reshape(3, 24, 24)
        return img_mirror

    def augmentData(self):
        y_index = 0
        y_final_batch = []
        x_final_batch = []
        for img in self.x_batch:
            for _ in range(5):
                img = img.reshape(3,32,32)
                img_crop = self.rand_crop(img)
                img_mirror = self.mirror(img_crop)
                x_final_batch.append(img_crop)
                x_final_batch.append(img_mirror)
                y_final_batch.append(self.y_batch[y_index])
                y_final_batch.append(self.y_batch[y_index])
            y_index +=1
        self.y_final_batch = np.array(y_final_batch).astype(int)
        self.x_final_batch = np.array(x_final_batch).astype(int)


batch_size = 128
epochs = 100

gph = tf.Graph()
with gph.as_default():
    x = tf.placeholder('float',shape = [None,3,24,24])
    y_ = tf.placeholder('float',shape = [None,1000])

    x_reshaped = tf.transpose(x,perm = [0,2,3,1])
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
    biases = [0]*8
    for i in range(8):
        kernels[i] = tf.Variable(tf.random_normal(shape = ls_ker_dims[i],mean = 0.0, stddev=0.01))
        biases[i] = tf.Variable(tf.constant(1.0,shape = [ls_ker_dims[i][-1]]))

    amap = tf.nn.relu(tf.nn.conv2d(x_reshaped,kernels[0],strides = [1,1,1,1],padding = 'VALID'))
    layer1 = tf.nn.max_pool(amap,[1,3,3,1],[1,1,1,1],padding = "VALID")

    amap = tf.nn.relu(tf.nn.conv2d(layer1,kernels[1],[1,1,1,1],padding = "VALID"))
    layer2 = tf.nn.max_pool(amap,[1,3,3,1],[1,1,1,1],'VALID')

    layer3 = tf.nn.relu(tf.nn.conv2d(layer2,kernels[2],[1,1,1,1],'VALID'))
    layer4 = tf.nn.relu(tf.nn.conv2d(layer3,kernels[3],[1,1,1,1],'VALID'))
    layer5 = tf.nn.relu(tf.nn.conv2d(layer4,kernels[4],[1,1,1,1],'VALID'))

    out_dims = [batch_size*10,10,10,256]
    layer5_flat = tf.reshape(layer5,shape = [out_dims[0],out_dims[1]*out_dims[2]*out_dims[3]])

    layer6 = tf.nn.relu(tf.matmul(layer5_flat,kernels[5]) + biases[5])
    layer7 = tf.nn.relu(tf.matmul(layer6, kernels[6]) + biases[6])
    layer8 = tf.matmul(layer7, kernels[7]) + biases[7]

    soft_max = tf.nn.softmax(layer8)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = layer8,labels = y_))
    opt = tf.train.AdamOptimizer()
    train = opt.minimize(loss)

with tf.Session(graph = gph) as sess:
    sess.run(tf.global_variables_initializer())
    train_data = Data()
    train_data.setFilePaths("D:/data/training/")

    for i in range(100):
        train_data.getRandomData(batch_size)
        train_data.augmentData()
        x_batch = train_data.x_final_batch
        y_batch = train_data.y_final_batch
        dict = {x: x_batch,y_:y_batch}
        _,cost = sess.run([train,loss],dict)
        print("Epoch: ",i,"\tCost: ",cost)
