config=$1
n_clients=$2
device='cuda:0'
indices=$(seq 0 $n_clients)

for i in $indices
do
	python3 client.py --config_file $config&
done
