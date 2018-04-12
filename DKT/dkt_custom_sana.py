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



class DKTcustomloss():

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
        
        

        
    def LossModel(y_train,dense_out):

        dense_output= Input(batch_shape=(None, None, self.input_dim_order), name='predicted_prob')
        y_order=Input(batch_shape=(None, None, self.input_dim_order), name='skill_array')
   
        merged = multiply([dense_out, y_order])
         ## 6. Chooses the max value(which is a non zero number in merged) from the output of merged layer
        # this will reduce dimension from skill_num to 1 for each timestep

        reduced = Lambda(reduce_dim, output_shape=reduce_dim_shape)(merged)
        cce = tf.losses.softmax_cross_entropy(self.y_train,reduced)
        loss_m = Model(inputs=[dense_output,y_order], outputs=cce)
        #it's important to make this model not trainable if it has weights 
        #(you should probably set these weights manually if that's the case)    
        loss_m.trainable = False
        return loss_m

    def build_train_on_batch(self):
            ## 1. First layer for the input (x_t), creates a tensor object
            x = Input(batch_shape=(None, None, self.input_dim), name='x')

            ## 2. Mask unknown or anomalous valued timesteps in x
            # the timestep will be masked (skipped) if all values in the input tensor
            #  at that timestep are equal to mask_value
            masked = Masking(mask_value=0)(x)

            ## 3. Add a lstm layer, return sequences is True to allow output have same
            # dimension as number of timesteps in input
            lstm_out = LSTM(self.hidden_layer_size, return_sequences=True)(masked)

            ## 4. Add a fully connected layer on lstm layer, 
            ### this gives us probabilities of all events at differnt timesteps
            dense_out = Dense(self.input_dim_order, activation='sigmoid')(lstm_out)
            

            
            
            def get_probability_of_timestep_event(x):
                #Custom tensor arithmatic from backend K
                # chooses the max value from the output of previous layer
                # this will reduce dimension from one hot of skill_num to 1 for each timestep
                x = K.max(x, axis=2, keepdims=True)
                return x

        
             ## 5.1 Get problem event encoding ONLY: 
            #  ASSUMPTION:that skill dict has problem events only from keys 1:input_dim_order
            
            def get_skill_array(x):
                return x[:,:,:self.input_dim_order]
  
            
#             y_order=Lambda(get_skill_array,output_shape=lambda s: (s[0], s[1],self.input_dim_order))(x)
            
#             ## 5.2 In dense output only retain probability of event at that timestep all others 0
#             merged = multiply([dense_out, y_order])

#             ## 6. Chooses the max value from the output of merged which is the prob
#                 # this will reduce dimension from skill_num to 1 for each timestep

#             reduced = Lambda(get_probability_of_timestep_event , output_shape=lambda s: (s[0], s[1],1))(merged)
    
#             ## 7. Creates model object with specified input and output
#             self.model = Model(inputs=x, outputs=reduced)

#             ## 8. Compile model by assigning loss function for backpropagtion
#             self.model.compile(optimizer=self.optimizer,
#                                loss='binary_crossentropy',
#                                metrics=['accuracy'])

#             print('Summary of the model')
#             self.model.summary()


            
            def custom_loss(y_train,dense_out):
                '''
                using crossentropy, choose from dense_out, the probabilities corresponding to event attemted
                at that timesetep. Create a tensor object of that accepts skill_array i. e one hot encoded skill sequence
                then mask the dense_out using that skill array
                '''
                print("using custom loss.....")

                
                y_order=Lambda(get_skill_array,output_shape=lambda s: (s[0], s[1],self.input_dim_order))(x)
            

                ## 5.2 In dense output only retain probability of event at that timestep all others 0
                merged = multiply([dense_out, y_order])

                ## 6. Chooses the max value from the output of merged which is the prob
                    # this will reduce dimension from skill_num to 1 for each timestep

                reduced = Lambda(get_probability_of_timestep_event , output_shape=lambda s: (s[0], s[1],1))(merged)

                cce = K.mean(K.binary_crossentropy(y_train, reduced ))#, axis=-1)
                return cce


            
            def custom_metric_accuracy(y_train,dense_out):
                '''
                using crossentropy, choose from dense_out, the probabilities corresponding to event attemted
                at that timesetep. Create a tensor object of that accepts skill_array i. e one hot encoded skill sequence
                then mask the dense_out using that skill array
                '''
                print("using custom metric.....")
                def get_response_array(x):
                    return x[:,:,-self.input_dim_order:]
                
                
                
                actual_response=Lambda(get_response_array,output_shape=lambda s: (s[0], s[1],self.input_dim_order))(x)
                print(actual_response.get_shape(),"actual response")
                def idx_max_prob(x):
                    max_idx= K.argmax(x,axis=-1)
#                         convert to one hot, to tackle,incorrect question argmax
                    one_hot_idx=tf.one_hot(max_idx,
                                self.input_dim_order,
                                on_value=1.0,
                                off_value=0.0,
                                axis=-1)
                    return one_hot_idx
        
                max_predicted_prob = Lambda(idx_max_prob,output_shape=lambda s: (s[0], s[1],self.input_dim_order))(dense_out)
                print( max_predicted_prob.get_shape()," max_predicted_prob")
                # we use merging, we want to find how many 1-1 matchings are there
                merged_metric = multiply([actual_response,  max_predicted_prob])


                accuracy = K.mean(K.sum(merged_metric,axis=-1))
                print(accuracy.get_shape())
                return accuracy


            ## 7.Creates model object with specified input and output
                #CHANGED: OUTPUTS: TO GET probabilities at all timesteps as output.

            self.model = Model(inputs=x, outputs=dense_out)

            ## 8. Compile model by assigning loss function for backpropagtion
            self.model.compile(optimizer=self.optimizer,
                               loss=custom_loss,
                               metrics=[custom_metric_accuracy])

            print('Summary of the model')
            self.model.summary()
   

    def train_on_batch(self, x_train,y_train,y_train_order):
# y_train_order: not used
        self.model.train_on_batch(x_train, y_train)




    def test_on_batch(self, x_val, y_val,y_train_order):
        
        # y_train_order: not used
        """
       Test the model on a single batch of samples
       :return: Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics).
       The attribute model.metrics_names will give you the display labels for the scalar outputs.
       """
        print(self.model.metrics_names)
        return self.model.test_on_batch(x_val, y_val)

    def predict(self, x_val,y_train_order):
#              # y_train_order: not used
        return self.model.predict_on_batch(x_val)