"""
Cleaned, NILMTK-free Im2Seq implementation
Author: Adapted from Bousbiat Hafsa
"""

import math
import pandas as pd
import numpy as np
from collections import OrderedDict
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

from .batchgenerator import BatchGenerator  


class ApplianceNotFoundError(Exception):
    pass


class Im2Seq:
    def __init__(self, params):
        self.img_method = params.get('img_method', 'gasf')
        self.MODEL_NAME = "Im2Seq_" + self.img_method
        self.models = OrderedDict()
        self.sequence_length = params.get('sequence_length')
        self.img_size = params.get('img_size', self.sequence_length)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size', 512)
        self.appliance_params = params.get('appliance_params', {})
        self.model_type = params.get('model_type', "simple")
        self.mains_min = 0
        self.mains_max = 14000

    def partial_fit(self, train_main, train_appliances):
        print(f"Training Im2Seq model with method: {self.img_method}")

        if not self.appliance_params:
            self.set_appliance_params(train_appliances)

        train_main, train_appliances = self.call_preprocessing(train_main, train_appliances, 'train')

        train_main = pd.concat(train_main, axis=0).values.reshape((-1, self.sequence_length))

        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs, axis=0)
            app_df_values = app_df.values.reshape((-1, self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))

        for appliance_name, power in new_train_appliances:
            if appliance_name not in self.models:
                print("Creating model for:", appliance_name)
                self.models[appliance_name] = self.return_network()

            model = self.models[appliance_name]

            print(f"train_main shape: {train_main.shape}, power shape: {power.shape}")
            print(f"Training {appliance_name}...")

            
            if len(train_main) > 10:
                filepath = f"Im2Seq-{self.img_method}-{self.sequence_length}-{self.model_type}-{appliance_name}-weights.h5"
                checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min')
                train_x, val_x, train_y, val_y = train_test_split(train_main, power, test_size=0.15, random_state=10)
                
                batch_generator = BatchGenerator(train_x, train_y, self.batch_size, self.img_method, self.img_size)
                val_generator = BatchGenerator(val_x, val_y, self.batch_size, self.img_method, self.img_size)
                print(f"Starting model.fit for {appliance_name}, epochs={self.n_epochs}")
                print(f"Steps per epoch: {len(train_y) // self.batch_size}")



                model.fit(batch_generator,
                          steps_per_epoch=len(train_y) // self.batch_size,
                          epochs=self.n_epochs,
                          verbose=1,
                          shuffle=True,
                          callbacks=[checkpoint],
                          validation_data=val_generator,
                          validation_steps=len(val_y) // self.batch_size)

                model.load_weights(filepath)

    def disaggregate_chunk(self, test_main_list):
        test_predictions = []
        for test_mains_df in test_main_list:
            disaggregation_dict = {}
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length))
            for appliance, model in self.models.items():
                batch_generator = BatchGenerator(test_main_array, None, self.batch_size, self.img_method, self.img_size)
                prediction = model.predict(batch_generator)
                
                l = self.sequence_length
                n = len(prediction) + l - 1
                sum_arr = np.zeros(n)
                counts_arr = np.zeros(n)

                for i in range(len(prediction)):
                    sum_arr[i:i + l] += prediction[i].flatten()
                    counts_arr[i:i + l] += 1
                
                sum_arr = np.divide(sum_arr, counts_arr, out=np.zeros_like(sum_arr), where=counts_arr != 0)
                prediction = self.appliance_params[appliance]['mean'] + (sum_arr * self.appliance_params[appliance]['std'])
                prediction = np.clip(prediction, 0, None)
                disaggregation_dict[appliance] = pd.Series(prediction)

            results = pd.DataFrame(disaggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    
    '''
    def return_network(self):
        model = Sequential()
        model.add(Conv2D(8, 4, strides=2, activation='linear', input_shape=(self.img_size, self.img_size, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.sequence_length * 8, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.sequence_length * 8, activation='relu'))
        model.add(Reshape((self.sequence_length, 8)))
        model.add(Conv1D(1, 4, padding='same', activation='linear'))
        model.compile(loss='mse', optimizer=SGD(learning_rate=1e-2, momentum=0.8))
        return model
    '''

    def return_network(self):
        reg = tf.keras.regularizers.l2(1e-4)
        S = self.img_size

        model = Sequential()
        # ↓ S → ~S/2
        model.add(Conv2D(16, 3, strides=2, padding='same', activation='relu',
                        kernel_regularizer=reg,
                        input_shape=(S, S, 1)))
        # ↓ ~S/4
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        # ↓ ~S/8
        model.add(Conv2D(32, 3, strides=2, padding='same', activation='relu',
                        kernel_regularizer=reg))

        # ↓ ~S/16  (extra downsampling before Flatten)
        model.add(tf.keras.layers.AveragePooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Flatten())                                      # much smaller now
        model.add(Dense(self.sequence_length * 4, activation='relu',
                        kernel_regularizer=reg))
        model.add(Dense(128, activation='relu', kernel_regularizer=reg))
        model.add(Dense(self.sequence_length * 4, activation='relu',
                        kernel_regularizer=reg))
        model.add(Reshape((self.sequence_length, 4)))
        model.add(Conv1D(1, 3, padding='same', activation='linear',
                        kernel_regularizer=reg))

        opt = SGD(learning_rate=1e-2, momentum=0.8)
        model.compile(loss=tf.keras.losses.Huber(delta=1.0), optimizer=opt)
        return model


    
    def call_preprocessing(self, mains_lst, submeters_lst, method):
        processed_mains_lst = []
    
        for mains in mains_lst:
            new_mains = mains.values.flatten()
            units_to_pad = self.sequence_length // 2
            new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant')
            new_mains = np.array([
                new_mains[i:i + self.sequence_length]
                for i in range(len(new_mains) - self.sequence_length + 1)
            ])
    
            if method == 'train':
                self.mains_mean = np.mean(new_mains)
                self.mains_std = np.std(new_mains)
    
            new_mains = (new_mains - self.mains_mean) / self.mains_std
            processed_mains_lst.append(pd.DataFrame(new_mains))
    
        if method == 'test':
            return processed_mains_lst
    
        appliance_list = []
        for app_name, app_df_lst in submeters_lst:
            app_mean = self.appliance_params[app_name]['mean']
            app_std = self.appliance_params[app_name]['std']
            processed_app_dfs = []
            for app_df in app_df_lst:
                new_app = app_df.values.flatten()
                new_app = np.pad(new_app, (units_to_pad, units_to_pad), 'constant')
                new_app = np.array([
                    new_app[i:i + self.sequence_length]
                    for i in range(len(new_app) - self.sequence_length + 1)
                ])
                new_app = (new_app - app_mean) / app_std
                processed_app_dfs.append(pd.DataFrame(new_app))
            appliance_list.append((app_name, processed_app_dfs))
    
        return processed_mains_lst, appliance_list


    def set_appliance_params(self, train_appliances):
        for app_name, df_list in train_appliances:
            data = pd.concat(df_list, axis=0).values.flatten()
            self.appliance_params[app_name] = {
                'mean': np.mean(data),
                'std': np.std(data)
            }
