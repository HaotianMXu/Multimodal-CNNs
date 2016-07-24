import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

def build_data_cv(data_folder, cv, clean_string=True):
    ceil=20000##########################################################
    revs = []
    vocab = defaultdict(float)
    for i in range(len(data_folder)):
        with open(data_folder[i], "rb") as f:
            for line in f:       
                rev = []
                rev.append(line.strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                words = set(orig_rev.split())
                wordslst=list(orig_rev.split())
                if len(wordslst)<=ceil:
                    for word in words:
                        vocab[word] += 1
                    datum  = {"y":i, 
                          "text": orig_rev,                             
                          "num_words": len(orig_rev.split()),
                          "split": np.random.randint(0,cv)}
                elif len(wordslst)>ceil:
                    new_rev=[]
                    for w in range(ceil):
                        vocab[wordslst[w]]+=1
                        new_rev.append(wordslst[w])
                    new_orig_rev = " ".join(new_rev).lower()
                    new_orig_rev = new_orig_rev.strip()
                    datum  = {"y":i, 
                          "text": new_orig_rev,                             
                          "num_words": len(new_orig_rev.split()),
                          "split": np.random.randint(0,cv)}
                revs.append(datum)
    return revs, vocab
    
def get_W(word_vecs, vocab,k):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        if word not in vocab:
            continue
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def get_W2(word_vecs,word_vecs2,vocab,k):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    W2 = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W2[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        if word not in vocab:
            continue
        if word not in word_vecs2:
            W2[i]=np.random.uniform(-0.25,0.25,k) 
        else:
            W2[i] = word_vecs2[word]
        i += 1
    return W2
    
def add_unknown_words(word_vecs, vocab, k,min_df=1):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k) 

def load_bin_vec(fname, vocab):
    """
    Loads dimx1 word vecs from bin word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs
 
def load_txt_vec(fname, vocab,dim):
    """
    Loads dimx1 word vecs from word2vec
    """
    word_vecs = {}
    vec=np.zeros(dim)
    with open(fname, "rb") as f:
        while True:
            line=f.readline()
            if not line:
                break
            tokens=line.strip().split()
            if len(tokens)!=(dim+1):#wrong!!!
                print 'ERROR VEC SIZE %d, should be %d' %(len(tokens),dim+1)
                break
            for t in range(dim):
                vec[t]=float(tokens[t+1])
            word_vecs[tokens[0]]=vec
    return word_vecs 

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__": 
    #print sys.argvs 
    s=''
    for i in xrange(len(sys.argv)):
        s=s+' '+sys.argv[i]
    print s
  
    np.random.seed(3333)
    w2v_file = sys.argv[1]  
    w2v_file2=sys.argv[2]  
    data_folder=[]
    dim=int(sys.argv[3])
    dim2=int(sys.argv[4])
    classnum=int(sys.argv[5])
    for cn in xrange(classnum):
        data_folder.append(sys.argv[6+cn])

    print "loading data..."    
    revs, vocab = build_data_cv(data_folder, cv=5, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    avg_l=np.mean(pd.DataFrame(revs)["num_words"])
    mid_l=np.median(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "average sentence length:"+str(avg_l)
    print "median sentence length:"+str(mid_l)
    print "loading vectors...",
    
    if w2v_file[-4:]=='.bin':
        w2v = load_bin_vec(w2v_file, vocab)
    elif w2v_file[-4:]=='.txt':
        w2v = load_txt_vec(w2v_file, vocab,dim)
    if w2v_file2[-4:]=='.bin':
        w2v2 = load_bin_vec(w2v_file2, vocab)
    elif w2v_file2[-4:]=='.txt':
        w2v2 = load_txt_vec(w2v_file2, vocab,dim2)
    
    print "word2vec loaded!"
    print "num words already in "+w2v_file+": " + str(len(w2v))
    print "num words already in "+w2v_file2+": " + str(len(w2v2))
    add_unknown_words(w2v, vocab,dim)
    W, word_idx_map = get_W(w2v,vocab,dim)#word representation1
    W2 = get_W2(w2v,w2v2,vocab,dim2)#word representation2

    print "shape of vec1="+str(W.shape[1])
    print "shape of vec2="+str(W2.shape[1])
    cPickle.dump([revs, W, W2, word_idx_map,classnum, max_l], open("mr.p", "wb"))
    print "dataset created!"
    
