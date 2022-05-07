# 联邦学习
#

import json
import torch, random
from fl_server import *
from fl_client import *
import models, datasets
import time

# The basic process of horizontal federated learning implemented:
#
# 1. The server generates the initialization model according to the configuration,
# and the client cuts the dataset horizontally without overlapping in sequence.
# 2. The server sends the global model to the client.
# 3. The client receives the global model (from the server) and returns the local
# parameter difference to the server through local iterations.
# 4. The server aggregates the difference between each client to update the model,
# and then evaluates the current model performance
# If the performance is not up to standard, repeat the process of 2, otherwise end.
# print(__name__)
if __name__ == '__main__':

    # load configuration file
    with open("conf.json", 'r') as f:
        conf = json.load(f)
    # Load dataset： train datasets, eval datasets
    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
    # Start the server
    server = Server(conf, eval_datasets)
    # client List
    clients = []
    # Add N clients to the list according to the conf configuration file
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))
    # 开始时间
    start_time = time.time()
    print("begin time:", start_time)

    # for the convenience of implementation, the implementation does not use network communication
    # to simulate the communication between the client and the server, but simulates it locally in a circular manner.
    print("begin global model training \n")
    # Global model training
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # Each training is to randomly sample k from the clients list for this round of training
        candidates = random.sample(clients, conf["k"])
        # weight accumulation
        weight_accumulator = {}
        # Initialize empty model parameter weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # Generate a 0 matrix of the same size as the parameter matrix
            weight_accumulator[name] = torch.zeros_like(params)

        # Traverse clients, each client trains the model locally
        for c in candidates:
            diff = c.local_train(server.global_model)
            # print("client:", diff )
            # Update the overall weight according to the client's parameter difference dictionary
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        # model parameter aggregation
        server.model_aggregate(weight_accumulator)
        # model evaluation
        acc, loss = server.model_eval()

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    end_time = time.time()
    print("end time:", end_time)

    end_time_calc = round(time.time() - start_time, 4)

    print('Execution time: {} seconds'.format(end_time_calc))
