# PINT: position-encoding injective temporal graph nets

This is the official repo for "Provably expressive temporal graph networks", published at NeurIPS 2022. 

Since we follow the same evaluation setup as TGNs, our code builds on top of the [TGN repository](https://github.com/twitter-research/tgn) --- we even left some of the original comments :sweat_smile:.

## Data
You can download the datasets from:
* Wikipedia: http://snap.stanford.edu/jodie/wikipedia.csv
* UCI: http://konect.cc/networks/opsahl-ucforum/
* Reddit: http://snap.stanford.edu/jodie/reddit.csv
* LastFM: http://snap.stanford.edu/jodie/lastfm.csv
* Enron: https://www.cs.cmu.edu/~./enron/

Once you do so, place the files in the 'data' folder and run, e.g:
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
```


## Preprocessing

To run PINT on any dataset, we first precompute the positional features. We'll use wikipedia as a running example.
We start off doing so for the training data:
```{bash}
python preproc.py --data wikipedia --r_dim 4 --data_split train
```
The flag 'r-dim' sets the dimension of positional features. 

Then, we do the same for the test and validation splits:
```{bash}
python preproc.py --data wikipedia --r_dim 4 --data_split val_ind
python preproc.py --data wikipedia --r_dim 4 --data_split val_trans
python preproc.py --data wikipedia --r_dim 4 --data_split test_ind
python preproc.py --data wikipedia --r_dim 4 --data_split test_trans
```
We note that the four commands above can be run in parallel.

Finally, we join the files created in the previous steps via:
```{bash}
python preproc.py --data wikipedia --r_dim 4 --data_split join
```

## Running PINT

With the precomputed positional features at hand, we run PINT using the following commands.

For Wikipedia:
```{bash}
python train.py --data wikipedia --n_layer 2 --use_memory --beta 0.0001 --n_epoch 50 --patience 5 --n_runs 10 --n_degree 10 --memory_dim 172
```

For UCI:
```{bash}
python train.py --data uci --n_layer 2 --use_memory --beta 0.00001 --n_epoch 50 --patience 5 --n_runs 10 --n_degree 10 --memory_dim 100
```
For Reddit:
```{bash}
python train.py --data reddit --n_layer 2 --use_memory --beta 0.00001 --n_epoch 50 --patience 5 --n_runs 10 --n_degree 10 --memory_dim 172
```
For LastFM:
```{bash}
python train.py --data lastfm --n_layer 2 --use_memory --beta 0.0001  --n_epoch 50 --patience 5 --n_runs 10 --n_degree 10 --memory_dim 172
```
For Enron:
```{bash}
python train.py --data enron --n_layer 2 --use_memory --beta 0.00001  --n_epoch 50 --patience 5 --n_runs 10 --n_degree 20 --memory_dim 32
```
## Dependencies

The code in this repository is relatively well-contained. In **Jan 16, 2023** we've installed all dependencies using the following commands:
```{bash}
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install scikit-learn
pip install pandas
```
Nonetheless, we provide a yml file with further details on our environment.


## Cite
```
@inproceedings{PINT2022,
  title={Provably expressive temporal graph networks},
  author={A. H. Souza and D. Mesquita and S. Kaski and V. Garg},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```





