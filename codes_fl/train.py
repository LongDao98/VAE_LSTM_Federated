import os
import tensorflow as tf
from data_loader_FL import DataGenerator as generator_fl
from data_loader import DataGenerator
from models import VAEmodel, lstmKerasModel
from trainers import vaeTrainer
from aggregator import Aggregator
from utils import process_config, create_dirs, get_args, save_config
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config['result_dir'], config['checkpoint_dir'], config['checkpoint_dir_lstm']])
    # save the config in a txt file
    save_config(config)
    # create tensorflow session
    sess1 = tf.Session(config=tf.ConfigProto())
    sess2 = tf.Session(config=tf.ConfigProto())
    #sess3 = tf.Session(config=tf.ConfigProto())
    sess_global = tf.Session(config=tf.ConfigProto())
    # create your data generator
    #config1 = config
    #config2 = config
    #config3 = config
    #config1['dataset'] = 'TEK14'
    #config2['dataset'] = 'TEK16'
    #config3['dataset'] = 'TEK17'
    #data1 = DataGenerator(config1)
    #data2 = DataGenerator(config2)
    #data3 = DataGenerator(config3)
    data1 = generator_fl(config,1)
    data2 = generator_fl(config,2)

    # create a CNN model
    model_vae_1 = VAEmodel(config, "Client1")
    model_vae_2 = VAEmodel(config, "Client2")
    #model_vae_3 = VAEmodel(config3, "Client3")
    model_vae_global = VAEmodel(config, "Global")
    model_vae_1.load(sess1)
    model_vae_2.load(sess2)
    #model_vae_3.load(sess3)
    model_vae_global.load(sess_global)

    # create a trainer for VAE model
    trainer_vae1 = vaeTrainer(sess1, model_vae_1, data1, config)
    trainer_vae2 = vaeTrainer(sess2, model_vae_2, data2, config)
    #trainer_vae3 = vaeTrainer(sess3, model_vae_3, data3, config3)
    trainer_vae_global = vaeTrainer(sess_global, model_vae_global, data1, config)
    vae_trainers = [trainer_vae1, trainer_vae2]#, trainer_vae3]
    lstm_model_1 = lstmKerasModel("Client1", config)
    lstm_model_2 = lstmKerasModel("Client2", config)
    #lstm_model_3 = lstmKerasModel("Client3", config3)
    lstm_model_global = lstmKerasModel("Global", config)
    lstm_models = [lstm_model_1, lstm_model_2]#, lstm_model_3]
    weights = [3,7]
    weights = weights / np.sum(weights)
    aggregator = Aggregator(vae_trainers, trainer_vae_global, lstm_models, lstm_model_global, config, weights)
    aggregator.aggregate_vae()
    aggregator.aggregate_lstm()
    # model_vae.load(sess)
    # # here you train your model
    # if config['TRAIN_VAE']:
    #     if config['num_epochs_vae'] > 0:
    #         trainer_vae.train()

    # if config['TRAIN_LSTM']:
    #     # create a lstm model class instance
    #     lstm_model = lstmKerasModel(data)

    #     # produce the embedding of all sequences for training of lstm model
    #     # process the windows in sequence to get their VAE embeddings
    #     lstm_model.produce_embeddings(config, model_vae, data, sess)

    #     # Create a basic model instance
    #     lstm_nn_model = lstm_model.create_lstm_model(config)
    #     lstm_nn_model.summary()   # Display the model's architecture
    #     # checkpoint path
    #     checkpoint_path = config['checkpoint_dir_lstm']\
    #                       + "cp.ckpt"
    #     # Create a callback that saves the model's weights
    #     cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                       save_weights_only=True,
    #                                                       verbose=1)
    #     # load weights if possible
    #     lstm_model.load_model(lstm_nn_model, config, checkpoint_path)

    #     # start training
    #     if config['num_epochs_lstm'] > 0:
    #         lstm_model.train(config, lstm_nn_model, cp_callback)

    #     # make a prediction on the test set using the trained model
    #     lstm_embedding = lstm_nn_model.predict(lstm_model.x_test, batch_size=config['batch_size_lstm'])
    #     print(lstm_embedding.shape)

    #     # visualise the first 10 test sequences
    #     for i in range(10):
    #         lstm_model.plot_lstm_embedding_prediction(i, config, model_vae, sess, data, lstm_embedding)
    sess1.close()
    sess2.close()
    #sess3.close()
    sess_global.close()


if __name__ == '__main__':
    main()
