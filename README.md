#### Matrix Standalone (Federated Learning Project)
A simple version of the horizontal federated learning project to achieve image classification learning. This project is simulated locally, and does not involve network communication details and failure processing. It only involves model aggregation functions, which can quickly verify the relevant algorithms and capabilities of federated learning.

### Operating Manual
```shell
python3 fl_integration.py
```

#### Simulation Ensemble Run(fl_integration) 

It will define a server object and multiple client objects respectively to simulate horizontal federation training scenarios.


#### Server（fl_server）

Perform model aggregation on the local model uploaded by the selected client.



#### Client（fl_client）

Receive commands and global models from the server, and use local data to train local models.


##### Functional screenshot

<img src="./doc/img/20220507-024348.jpg" width="930">


##### Results screenshot

<img src="./doc/img/acc.jpg" width="930">

<img src="./doc/img/loss.jpg" width="930">

---