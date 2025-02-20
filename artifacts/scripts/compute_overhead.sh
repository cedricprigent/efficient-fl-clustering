python3 data_preparation/partition_data.py --dataset mnist
python3 compute_embedding_vector_overhead.py --dataset MNIST

python3 data_preparation/partition_data.py --dataset cifar10
python3 compute_embedding_vector_overhead.py --dataset CIFAR10

tar -xzf /tmp/app/data/femnist/partitions.tar.gz -C /tmp/app/data
python3 compute_embedding_vector_overhead.py --dataset FEMNIST

python3 data_preparation/PACS.py --n_partitions 10
python3 compute_embedding_vector_overhead.py --dataset PACS

sleep infinity