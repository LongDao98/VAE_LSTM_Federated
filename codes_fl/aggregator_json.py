import os
import tensorflow as tf
import json
from data_loader import DataGenerator
from models import VAEmodel, lstmKerasModel
from trainers import vaeTrainer
from utils import process_config, create_dirs, get_args, save_config
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Aggregator:
    def __init__(self, vae_trainers, global_vae_trainer, lstm_models, global_lstm_model, config, weights):
        self.global_lstm_model = global_lstm_model
        self.vae_trainers = vae_trainers
        self.global_vae_trainer = global_vae_trainer
        self.lstm_models = lstm_models
        self.config = config
        self.num_comm_rounds = self.config["num_comm_rounds"]
        self.weights = weights

    def aggregate_vae(self):
        if self.config['TRAIN_VAE']:
            if self.config['vae_epochs_per_comm_round'] > 0:
                for comm_round in range(self.config['num_comm_rounds']):
                    self.train_vars_VAE_of_clients = []
                    for vae_trainer in self.vae_trainers:
                        for i in range(len(vae_trainer.model.train_vars_VAE)):
                            #vae_trainer.model.train_vars_VAE[i].load(self.global_vae_trainer.model.train_vars_VAE[i].eval(self.global_vae_trainer.sess), vae_trainer.sess)
                            #cac model client lay weight tu model global theo tung lop
                            #lay weight tung lop cua global model
                            sent_weight = self.global_vae_trainer.model.train_vars_VAE[i].eval(self.global_vae_trainer.sess)
                            #luu weight vao json
                            with open('test.json','w') as f:
                                json.dump(sent_weight.tolist(),f)#,indent=4)
                            #lay weight tu file
                            with open('test.json','r') as f:
                                received_weight = json.load(f)
                            #model client set weight
                            vae_trainer.model.train_vars_VAE[i].load(np.array(received_weights), vae_trainer.sess)
                        #train model client
                        vae_trainer.train()
                        self.train_vars_VAE_of_clients.append(vae_trainer.model.train_vars_VAE)
                    for i in range(len(self.vae_trainers[0].model.train_vars_VAE)):
                        # tong hop model global theo tung lop mot
                        # dau tien set weight = 0
                        global_train_var_eval = np.zeros_like(self.vae_trainers[0].model.train_vars_VAE[i].eval(self.vae_trainers[0].sess))
                        #sau do lay weight tung model, nhan voi he so roi cong vao nhau
                        #o day model client phai lay weight ra, gui di bang file json
                        #model global doc file json nhan dc roi tinh FedAvg
                        #tuong tu nhu tren, Luong tu lam nhe
                        for j in range(len(self.vae_trainers)):
                            global_train_var_eval += np.multiply(self.weights[j], self.vae_trainers[j].model.train_vars_VAE[i].eval(self.vae_trainers[j].sess))
                        self.global_vae_trainer.model.train_vars_VAE[i].load(global_train_var_eval, self.global_vae_trainer.sess)
                        # print(self.global_vae_trainer.model.train_vars_VAE[i].eval(self.global_vae_trainer.sess))
                #save model global
                self.global_vae_trainer.model.save(self.global_vae_trainer.sess)

    def aggregate_lstm(self):
        if self.config['TRAIN_LSTM']:
            if self.config['lstm_epochs_per_comm_round'] > 0:
                for comm_round in range(self.config['num_comm_rounds']):
                    lstm_weights = []
                    #lay weight cua model global, global_weights la sent_weights
                    global_weights = self.global_lstm_model.lstm_nn_model.get_weights()
                    #luu vao file json, coi nhu gui di
                    with open('test2.json','w') as f:
                        json.dump([ x.tolist() for x in global_weights],f)
                    for i in range(len(self.lstm_models)):
                        lstm_model = self.lstm_models[i]

                        # produce the embedding of all sequences for training of lstm model
                        # process the windows in sequence to get their VAE embeddings
                        # lstm_model.produce_embeddings(self.vae_trainers[i].model, self.vae_trainers[i].data, self.vae_trainers[i].sess)
                        lstm_model.produce_embeddings(self.global_vae_trainer.model, self.global_vae_trainer.data, self.global_vae_trainer.sess)

                        # Create a basic model instance
                        # lstm_nn_model = lstm_model.create_lstm_model(self.config)
                        lstm_nn_model = lstm_model.lstm_nn_model
                        #lstm_nn_model.set_weights(global_weights)
                        #doc weight tu file json, coi nhu nhan ve tu server
                        with open('test2.json','r') as f:
                            received_weights_lstm = json.load(f)
                        #client set weight theo cai nhan duoc
                        lstm_nn_model.set_weights([np.array(x) for x in received_weights_lstm])
                        lstm_nn_model.summary()   # Display the model's architecture
                        # checkpoint path
                        checkpoint_path = self.config['checkpoint_dir_lstm']\
                                          + "cp_{}.ckpt".format(lstm_model.name)
                        # Create a callback that saves the model's weights
                        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                          save_weights_only=True,
                                                                          verbose=1)
                        # load weights if possible
                        # lstm_model.load_model(lstm_nn_model, checkpoint_path)

                        # start training
                        if self.config['lstm_epochs_per_comm_round'] > 0:
                            lstm_model.train(lstm_nn_model, cp_callback)
                        lstm_weights.append(lstm_nn_model.get_weights())
                        # set globel_weights = 0
                        # global_weights = np.subtract(global_weights, global_weights)
                    for i in range(len(self.lstm_models)):
                        #o day lai lay weight lstm tu cac client gui len roi FedAvg
                        #roi set weights cua global model bang cai tinh dc
                        #tuong tu nhu tren, Luong tu lam nhe
                        if i == 0:
                            global_weights = np.multiply(lstm_weights[i], self.weights[i])
                        else:
                            global_weights += np.multiply(lstm_weights[i], self.weights[i])
                    #sua indentation cho nay
                    self.global_lstm_model.lstm_nn_model.set_weights(global_weights)
                        # make a prediction on the test set using the trained model
                        # lstm_embedding = lstm_nn_model.predict(lstm_model.x_test, batch_size=self.config['batch_size_lstm'])
                        # print(lstm_embedding.shape)
                                    # save global lstm model
                glb_checkpoint_path = self.config['checkpoint_dir_lstm'] + "cp_{}.ckpt".format(self.global_lstm_model.name)
                self.global_lstm_model.lstm_nn_model.save_weights(glb_checkpoint_path)