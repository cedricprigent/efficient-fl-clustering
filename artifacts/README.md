### Artifacts

* [config](config) contains config files to run experiments locally.
* [data_preparation](data_preparation) contains data preparation files.
* [scripts](scripts) contains scripts used to evaluate the impact of randomized quantization on the clustering accuracy of our approach.
* [strategies](strategies) contains server strategies: AEPreTraining (autoencoder pretraining strategy), ClusterEmbedding (one-shot clustering with K-Means), IFCA (dynamic clustering) TensorboardStrategy (FedAvg).
* [utils](utils) contains utilitary functions and classes.

### Running experiments locally

**Dependencies and generation of client partitions**
* Installing dependencies
```bash
$ virtualenv -p python3 venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

* Downloading dataset and generating client partitions
```
# mnist
$ python3 data_preparation/partition_data.py --dataset mnist --n_partitions 100 --alpha 10

# cifar10
$ python3 data_preparation/partition_data.py --dataset cifar10 --n_partitions 100 --alpha 10

# pacs (10 partitions per domain)
$ python3 data_preparation/PACS.py --n_partitions 10 --alpha 10
```

> Comments regarding PACS and FEMNIST preparation: Before running the data_preparation script, PACS dataset should first be downloaded. FEMNIST is available through the LEAF benchmark. Additional preparation for FEMNIST partitions are provided in the notebooks.


**Running experiments**

Several configuration files are provided to run experiments with the different datasets: config-mnist.json, config-cifar10.json, config-femnist.json, config-pacs.json, config-pretraining.json.

* Starting the FL server
```
$ python3 server.py --config_file config/config-mnist.json
``` 

* Starting a federated client
```
$ python3 client.py --config_file config/config-mnist.json
```

* Starting multiple federated clients
```
$ bash scripts/launch_clients.sh config/config-mnist.json
```

**Configuring experiments**

We detail here the arguments to use for running each strategy.

|                   | strategy        | client              | compression |
|-------------------|-----------------|---------------------|-------------|
| AE Pretraining    | "ae-pretraining"| "AETrainerClient"   | "None"      |
| Ours (federated)  | "testencoding"  | "EncodingClient"    | "AE"        |
| Ours (centralized)| "testencoding"  | "EncodingClient"    | "FashionMNIST" or "Cifar100" |
| IFCA              | "ifca"          | "IFCAClient"        | "None"      |
| LADD              | "testencoding"  | "EncodingClient"    | "StyleExtraction" |


**Configuration used for AE pre-training experiments**

We detail here the arguments used for our AE pre-training experiments. 

|          | Sampled clients | Number of rounds | Local epochs |
|----------|-----------------|------------------|--------------|
| MNIST    | 50              | 10               | 3            |
| FEMNIST  | 75              | 20               | 3            |
| CIFAR-10 | 50              | 10               | 1            |
| PACS     | 20              | 30               | 3            |

**Evaluating randomized quantization impacts on clustering**
```
$ python3 scripts/test_random_quantization_mnist.py
$ python3 scripts/test_random_quantization_cifar10.py
```