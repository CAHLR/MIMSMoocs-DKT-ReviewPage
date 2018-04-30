from keras.models import Model
from keras.layers import Input, Dropout, Masking, Dense, Embedding
from keras.layers import Embedding
from keras.layers.core import Flatten, Reshape
from keras.layers import LSTM
from keras.layers import merge
from keras.layers.merge import multiply
from keras.callbacks import EarlyStopping
from keras import backend as K
from theano import tensor as T
from theano import config
from theano import printing
from theano import function
from keras.layers import Lambda
import theano
import numpy as np
import pdb
from math import sqrt
from keras.callbacks import Callback

class TestCallback(Callback):

    def __init__(self, test_data = [[],[],[]]):
        """
        x_test:
        y_test_order:
        y_test:

        """
        self.x_test, self.y_test_order, self.y_test = test_data

    def on_epoch_end(self, epoch, logs={}):
        """

        :param epoch: number of user defined epochs for training
        :param logs:
        :return:
        """
        y_pred = self.model.predict([self.x_test, self.y_test_order])
        avg_rmse, avg_acc = self.rmse_masking(self.y_test, y_pred)
        print('\nTesting avg_rmse: {}\n'.format(avg_rmse))
        print('\nTesting avg_acc: {}\n'.format(avg_acc))

    def rmse_masking(self, y_true, y_pred):
        """

        :param y_true: validation data actual label
        :param y_pred: model predicted label
        :return:
        """
        mask_matrix = np.sum(self.y_test_order, axis=2).flatten()
        num_users, max_responses = np.shape(self.x_test)[0], np.shape(self.x_test)[1]
        # we want y_pred and y_true both to be matrix of 2 dim.
        if len(y_pred.shape) and len(y_true.shape) == 3:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
        rmse = []
        acc = []
        padding_num = 0
        for user in range(num_users):
            diff_sq, response, correct = 0, 0, 0
            for i in range(user * max_responses, (user + 1) * max_responses):
                if mask_matrix[i] == 0:
                    break
                if y_true[i] == 1 and y_pred[i] >0.5:
                    correct += 1
                elif y_true[i] == 0 and y_pred[i] < 0.5:
                    correct += 1
                elif y_true[i] == -1:
                    padding_num += 1
                response += 1
                diff_sq += (y_true[i] - y_pred[i]) ** 2
            if response != 0:
                acc.append(correct/float(response))
                rmse.append(sqrt(diff_sq/float(response)))
        try:
            return sum(rmse)/float(len(rmse)), sum(acc)/float(len(acc))
        except:
            pdb.set_trace()

    def rmse_masking_on_batch(self, y_true, y_pred, y_order):
        num_users, max_responses = np.shape(y_order)[0], np.shape(y_order)[1]
        mask_matrix = np.sum(y_order, axis=2).flatten()
        #we want y_pred and y_true both to be matrix of 2 dim.
        if len(y_pred.shape) and len(y_true.shape) == 3:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
        rmse = []
        acc = []
        padding_num = 0
        for user in range(num_users):
            diff_sq, response, correct = 0, 0, 0
            for i in range(user * max_responses, (user + 1) * max_responses):
                if mask_matrix[i] == 0:
                    break
                if y_true[i] == 1 and y_pred[i] >0.5:
                    correct += 1
                elif y_true[i] == 0 and y_pred[i] < 0.5:
                    correct += 1
                elif y_true[i] == -1:
                    padding_num += 1
                response += 1
                diff_sq += (y_true[i] - y_pred[i]) ** 2
            if response != 0:
                acc.append(correct/float(response))
                rmse.append(sqrt(diff_sq/float(response)))
        try:
            return rmse, acc
        except:
            pdb.set_trace()

class DKTnet():

    def __init__(self, 
                input_dim, 
                input_dim_order, 
                hidden_layer_size, 
                batch_size, 
                epochs,
                x_train=[], 
                y_train=[], 
                y_train_order=[],
                validation_split=0.0,
                validation_data=None,
                optimizer='adam',
                callbacks=None):
        """

        :param input_dim: dimension of the input at one timestamp (dimension of x_t= 2*num_skills)
        :param input_dim_order: dimension of the one-hot representation of problem to check order of occurence(=num_skills)
        :param hidden_layer_size: number of nodes in hidden layer
        :param x_train: 3D matrix of size (samples, number of timestamp/sequence length, dimension of input vec (x_t) )
        :param y_train: a matrix of responses (samples,number of timestamp/sequence length)
        :param y_train_order: shape of output equal to number of timesteps
        :param validation_split:
        :param validation_data:
        :param optimizer:
        :param callbacks:

        """
        ## input dim is the dimension of the input at one timestamp (dimension of x_t)
        self.input_dim = int(input_dim) #2* num_skills

        ## input_dim_order is the dimension of the one-hot representation of problem
        ## CHECK: order of occurence of responses should be according to timestamp
        self.input_dim_order = int(input_dim_order)#num_skills

        self.hidden_layer_size = int(hidden_layer_size)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)

        ## x_train is a 3D matrix of size (samples, number of timestamp, dimension of input vec (x_t) )
        ## in cognitive tutor # of students * # total responses * # input_dim
        self.x_train = x_train
        ## y_train is a matrix of (samples one hot representation according to problem output value at each timestamp (y_t) )
        self.y_train = y_train
        ## y_train_order is the one hot representation of problem according to timestamp starting from
        ## t=1 if training starts at t=0
        self.y_train_order = y_train_order
        # users: no of student datapoints
        self.users = np.shape(x_train)[0]
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.callbacks = callbacks
        print ("DKTnet initialization done")

    def build_train_on_batch(self):
        ## 1. First layer for the input (x_t), creates a tensor object
        x = Input(batch_shape=(None, None, self.input_dim), name='x')

        ## 2. Mask unknown or anomalous valued timesteps in x
        # the timestep will be masked (skipped) if all values in the input tensor
        #  at that timestep are equal to mask_value
        masked = Masking(mask_value=-1)(x)

        ## 3. Add a lstm layer, return sequences is True to allow output have same
        # dimension as number of timesteps in input
        lstm_out = LSTM(self.hidden_layer_size, return_sequences=True)(masked)

        ## 4. Add a fully connected layer on lstm layer

        dense_out = Dense(self.input_dim_order, activation='sigmoid')(lstm_out)
        ## 5. Create  a tensor object --not sure if its required
        y_order = Input(batch_shape=(None, None, self.input_dim_order), name='y_order')
        merged = multiply([dense_out, y_order])

        def reduce_dim(x):
            #Custom tensor arithmatic from backend K
            # chooses the max value from the output of previous layer
            # this will reduce dimension from skill_num to 1 for each timestep
            x = K.max(x, axis=2, keepdims=True)
            return x

        def reduce_dim_shape(input_shape):
            shape = list(input_shape)
            shape[-1] = 1
            return tuple(shape)

        ## 6. Chooses the max value from the output of previous layer
            # this will reduce dimension from skill_num to 1 for each timestep
        reduced = Lambda(reduce_dim, output_shape=reduce_dim_shape)(merged)

        ## 7. Creates model object with specified input and output
        self.model = Model(inputs=[x, y_order], outputs=reduced)

        ## 8. Compile model by assigning loss function for backpropagtion
        self.model.compile(optimizer=self.optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        print('Summary of the model')
        self.model.summary()

    def train_on_batch(self, x_train, y_train, y_train_order):

        self.model.train_on_batch([x_train, y_train_order], y_train)



    def test_on_batch(self, x_val, y_val, y_val_order):
        """
       Test the model on a single batch of samples.
       :param x_train:
       :param y_train:
       :param y_train_order:
       :return: Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics).
       The attribute model.metrics_names will give you the display labels for the scalar outputs.
       """
        print(self.model.metrics_names)
        return self.model.test_on_batch([x_val, y_val_order], y_val)

    def predict(self, x_val, y_val_order):
        return self.model.predict([x_val, y_val_order])

    def build(self):
        """ Go through train on batch function above"""
        ## first layer for the input (x_t)
        x = Input(batch_shape=(None, None, self.input_dim), name='x')
        masked = Masking(mask_value= -1, input_shape = (None, None, self.input_dim))(x)
        lstm_out = LSTM(self.hidden_layer_size, return_sequences = True)(masked)
        dense_out = Dense(self.input_dim_order, activation='sigmoid')(lstm_out)
        y_order = Input(batch_shape=(None, None, self.input_dim_order), name='y_order')
        merged = multiply([dense_out, y_order])

        def reduce_dim(x):
            x = K.max(x, axis = 2, keepdims=True)
            return x

        def reduce_dim_shape(input_shape):
            shape = list(input_shape)
            shape[-1] = 1
            return tuple(shape)

        reduced = Lambda(reduce_dim, output_shape=reduce_dim_shape)(merged)
        self.model = Model(inputs=[x, y_order], outputs=reduced)
        self.model.compile(optimizer=self.optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        print('Summary of the model')
        self.model.summary()

    def fit_data(self):
        self.model.fit([self.x_train, self.y_train_order], 
                self.y_train, 
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=self.callbacks,
                validation_split=self.validation_split, 
                validation_data=self.validation_data,
                shuffle=True)