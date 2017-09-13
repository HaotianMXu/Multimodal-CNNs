# Multimodal-CNNs
Implementation of ACM-BCB 2016 paper [Text Classification with Topic-based Word Embedding and Convolutional Neural Networks](http://www.cs.wayne.edu/~mdong/ACMBCB16.pdf)

## Requirements
* Python==2.7
* Theano==0.7
* Lasagne==0.1
* Pre-trained word2vec vectors (CNN-gn in the paper) and Skip-gram model are [available](https://code.google.com/p/word2vec/)
* Latent Dirichlet Allocation model can be downloaded from [here](https://github.com/blei-lab/lda-c/)

## Preprocessing
To process raw textual data,
    python process_data_channel.py /path/to/wordvec1 /path/to/wordvec2 LengthofWordvec1 LengthofWordvec2 classnum /path/to/data_class0 /path/to/data_class1  
OR
    python process_data_multi.py /path/to/wordvec1 /path/to/wordvec2 LengthofWordvec1 LengthofWordvec2 classnum /path/to/data_class0 /path/to/data_class1

## Running CNN-channel or CNN-concat models
    python conv_channel_w2v.py epochnum batchsize
OR
    python conv_multi_w2v.py epochnum batchsize
You can choose epochnum=25 and batchsize=64 as an example.

## Acknowledgment
The preprocessing code is adapted from [here](https://github.com/yoonkim/CNN_sentence).

