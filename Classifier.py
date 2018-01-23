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
    y_true = tf.placeholder('float',shape = [None,1000])
    y_true_cls = tf.argmax(y_true,axis = 1)

    x_reshaped = tf.transpose(x,perm = [0,2,3,1])

    kern_size = [3,3]
    ls_kern_count = [96,256,384,384,256]

    #initializer for kernels and biases
    kern_init = tf.random_normal(mean = 0.0,stddev=0.01)
    bias_init = tf.zeros_initializer()

    conv1 = tf.layers.conv2d(x_reshaped,filters = ls_kern_count[0],
                             kern_size = kern_size, strides = [1,1],padding = "valid",
                             activation = tf.nn.relu,use_bias = True,
                             kernel_initializer=kern_init,
                             bias_initializer=bias_init,
                             trainable=True,name = "conv1")
    pool1 = tf.layers.max_pooling2d(conv1,kern_size,[1,1],padding="valid",name = "pool1")

    conv2 = tf.layers.conv2d(pool1,filters = ls_kern_count[1],
                             kern_size = kern_size, strides = [1,1],padding = "valid",
                             activation = tf.nn.relu,use_bias = True,
                             kernel_initializer=kern_init,
                             bias_initializer=bias_init,
                             trainable=True,name = "conv2")
    pool2 = tf.layers.max_pooling2d(conv2,kern_size,[1,1],padding="valid",name = "pool2")

    conv3 = tf.layers.conv2d(pool2,filters = ls_kern_count[2],
                             kern_size = kern_size, strides = [1,1],padding = "valid",
                             activation = tf.nn.relu,use_bias = True,
                             kernel_initializer=kern_init,
                             bias_initializer=bias_init,
                             trainable=True,name = "conv3")
    conv4 = tf.layers.conv2d(conv3, filters=ls_kern_count[3],
                             kern_size=kern_size, strides=[1, 1], padding="valid",
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=kern_init,
                             bias_initializer=bias_init,
                             trainable=True, name="conv4")
    conv5 = tf.layers.conv2d(conv4, filters=ls_kern_count[4],
                             kern_size=kern_size, strides=[1, 1], padding="valid",
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=kern_init,
                             bias_initializer=bias_init,
                             trainable=True, name="conv5")

    flat_tensor = tf.layers.flatten(conv5,name = "flat_tensor")

    #fully connected layer
    ls_units = [4096,4096,1000]
    fc1 = tf.layers.dense(flat_tensor,units = ls_units[0],activation = tf.nn.relu,
                          use_bias = True,kernel_initializer=kern_init,
                          bias_initializer=bias_init,trainable=True,name = "fc1")
    fc2 = tf.layers.dense(fc1, units=ls_units[1], activation=tf.nn.relu,
                          use_bias=True, kernel_initializer=kern_init,
                          bias_initializer=bias_init, trainable=True, name="fc1")
    logits = tf.layers.dense(fc2,units = ls_units[2],activation=None,use_bias = True,
                             kernel_initializer=kern_init, bias_initializer=bias_init,
                             trainable=True,name = "logits")

    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred,axis = 1)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = y_true))
    opt = tf.train.AdamOptimizer()
    train = opt.minimize(loss)

with tf.Session(graph = gph) as sess:
    sess.run(tf.global_variables_initializer())
    train_data = Data()
    train_data.setFilePaths("D:/data/training/")

    for i in range(epochs):
        train_data.getRandomData(batch_size)
        train_data.augmentData()
        x_batch = train_data.x_final_batch
        y_batch = train_data.y_final_batch
        dict = {x: x_batch,y_true:y_batch}
        _,cost = sess.run([train,loss],dict)
        print("Epoch: ",i,"\tCost: ",cost)
