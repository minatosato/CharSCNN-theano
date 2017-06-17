# CharSCNN
## About
This is implementation of CharSCNN and SCNN[Ćıcero Nogueira dos Santos 2014].  
This repository is heavily based on [satwantrana/CharSCNN](https://github.com/satwantrana/CharSCNN).  
Experimental results are shown in [this **Japanese** post](http://qiita.com/hogefugabar/items/93fcb2bc27d7b268cbe6).



## Requirement
- Python 2 >= 2.6
- NumPy >= 1.7.1
- SciPy >= 0.11
- Theano == 0.9.0
- scikit-learn == 0.17
	- for dividing dataset into training set and test set


## Usage

```
git clone https://github.com/hogefugabar/CharSCNN-theano
```

Download [this tweets file from satwantrana/CharSCNN](https://raw.githubusercontent.com/satwantrana/CharSCNN/master/tweets_clean.txt) and add to cloned repository directory.  
e.g., 
```
cd CharSCNN-theano
wget https://raw.githubusercontent.com/satwantrana/CharSCNN/master/tweets_clean.txt
```

To run SCNN;
```
python scnn.py
```

To run CharSCNNl;
```
python charscnn.py
```

Note: default dataset size (i.e., `num_sent`) is set to 20,000 in `load.py` or `char_load.py`

## Reference
Ćıcero Nogueira dos Santos and Máıra Gatti. 2014. Deep convolutional neural networks for sentiment analysis of short texts. In Proceedings of the 25th International Conference on Computational Linguistics (COLING), Dublin, Ireland.