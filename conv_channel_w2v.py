# -*- coding: utf-8 -*-
"""
lasagne implementation
"""

import cPickle
import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import sys
from collections import OrderedDict
from sklearn import metrics


#warnings.filterwarnings("ignore")  

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return T.cast(shared_x,'int32'), T.cast(shared_y, 'int32')

def shared_dataset_CPU(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        CPU
        """
        data_x, data_y = data_xy
        shared_x = theano.tensor._shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.tensor._shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return T.cast(shared_x,'int32'), T.cast(shared_y, 'int32')
        
def get_idx_from_sent(sent, word_idx_map, max_l=51, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, filter_h)   
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train,dtype="int32")
    test = np.array(test,dtype="int32")
    ###shuffle
    np.random.shuffle(train)
    np.random.shuffle(test)
    return [train, test]  

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        #print param.name
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words') and (param.name!='Words2'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
        
def negative_log_likelihood(prediction,target):
    return -T.mean(T.log(prediction)[T.arange(target.shape[0]), target])

    
def build_cnn(sens,sen_length, W_embed_top, W_embed_bot,dim_top,dim_bot,h_top,h_bot,class_num,batch_size):
    filter_height=[3,4,5]
    dropout_rate=0.5
    hidden_unit=100

    # Input layer: word indexes of a sentence
    
    l_in = lasagne.layers.InputLayer(shape=(None,1,sen_length),
                                        input_var=sens)                                 
    #top half                                   
    l_embed_top=lasagne.layers.EmbeddingLayer(l_in,h_top,dim_top,W=W_embed_top)
    
    l_top_conv1=lasagne.layers.Conv2DLayer(l_embed_top,num_filters=hidden_unit,
                                           filter_size=(filter_height[0],dim_top),stride=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            W=lasagne.init.Normal())
                                            
    l_top_conv2=lasagne.layers.Conv2DLayer(l_embed_top,num_filters=hidden_unit,
                                           filter_size=(filter_height[1],dim_top),stride=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            W=lasagne.init.Normal())    
    
    l_top_conv3=lasagne.layers.Conv2DLayer(l_embed_top,num_filters=hidden_unit,
                                           filter_size=(filter_height[2],dim_top),stride=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            W=lasagne.init.Normal())       

    
    #bot half
    l_embed_bot=lasagne.layers.EmbeddingLayer(l_in,h_bot,dim_bot,W=W_embed_bot)
    
    l_bot_conv1=lasagne.layers.Conv2DLayer(l_embed_bot,num_filters=hidden_unit,
                                           filter_size=(filter_height[0],dim_bot),stride=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            W=lasagne.init.Normal())
                                            
    l_bot_conv2=lasagne.layers.Conv2DLayer(l_embed_bot,num_filters=hidden_unit,
                                           filter_size=(filter_height[1],dim_bot),stride=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            W=lasagne.init.Normal())    
    
    l_bot_conv3=lasagne.layers.Conv2DLayer(l_embed_bot,num_filters=hidden_unit,
                                           filter_size=(filter_height[2],dim_bot),stride=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            W=lasagne.init.Normal())       
   
    
    #merge
    l_conv1=lasagne.layers.ElemwiseMergeLayer([l_top_conv1,l_bot_conv1],T.add)
    l_conv2=lasagne.layers.ElemwiseMergeLayer([l_top_conv2,l_bot_conv2],T.add)
    l_conv3=lasagne.layers.ElemwiseMergeLayer([l_top_conv3,l_bot_conv3],T.add)
    #pool
    l_pool1=lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(sen_length-filter_height[0]+1, 1))
    l_pool2=lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(sen_length-filter_height[1]+1, 1))
    l_pool3=lasagne.layers.MaxPool2DLayer(l_conv3, pool_size=(sen_length-filter_height[2]+1, 1))
    #merge
    all_layers=[l_pool1,l_pool2,l_pool3]
    l_comb=lasagne.layers.ConcatLayer(all_layers,axis=1)
    l_drop=lasagne.layers.dropout(l_comb, p=dropout_rate)
    #fully connected
    l_out=lasagne.layers.DenseLayer(l_drop,
                                    num_units=class_num,
                                    nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def train_cnn(dataset, W1, W2,class_num,num_epochs,batch_size):
    print("Building model and compiling functions...")
    #print("learning rate: "+str(learning_rate))
    print("num_epoch: "+str(num_epochs))
    print("batch_size: "+str(batch_size))
    input_var = T.itensor3('inputs')
    target_var = T.ivector('targets')
    index=T.iscalar()
    sen_length=len(dataset[0][0])-1
    train=dataset[0]
    test=dataset[1]
    #pad train set to n*batch_size
    if train.shape[0] % batch_size > 0:
        extra_data_num = batch_size - train.shape[0] % batch_size
        train_set = np.random.permutation(train)   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(train,extra_data,axis=0)
    else:
        new_data = train
    n_batches = new_data.shape[0]/batch_size
    train_x=new_data[:,:sen_length]
    train_x=train_x.reshape(train_x.shape[0],1,train_x.shape[1])
    train_y=new_data[:,-1]
    
    test_x=test[:,:sen_length]
    test_x=test_x.reshape(test_x.shape[0],1,test_x.shape[1])
    test_y=test[:,-1]   
    
    train_set_x,train_set_y=shared_dataset_CPU((train_x,train_y))
    test_set_x,test_set_y=shared_dataset_CPU((test_x,test_y))
    
    h,dim=W.shape
    h2,dim2=W2.shape
    Words = theano.shared(value = W, name = "Words")
    Words2 = theano.shared(value = W2, name = "Words2")    
    
    network = build_cnn(sens=input_var,sen_length=sen_length, 
                        W_embed_top=Words, W_embed_bot=Words2,dim_top=dim,dim_bot=dim2,
                        h_top=h,h_bot=h2,
                        class_num=class_num,batch_size=batch_size)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)  
    loss=lasagne.objectives.categorical_crossentropy(prediction,target_var)
    loss=T.mean(loss)

    # Create update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)          
    grad_updates = sgd_updates_adadelta(params, loss, rho=0.95,epsilon=1e-6,norm_lim=9)
    
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss=lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
    test_loss=T.mean(test_loss)

    test_predict_class=T.argmax(test_prediction, axis=1)

    train_fn = theano.function([index], loss, updates=grad_updates,
          givens={
            input_var: train_set_x[index*batch_size:(index+1)*batch_size],
              target_var: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)

    test_fn = theano.function([], [test_loss, test_predict_class],
                              givens={input_var:test_set_x,target_var:test_set_y},
                              allow_input_downcast=True)

    # we define 0 in wordmap as padding character. keep it as zero
    zero_vec_tensor = T.vector()
    zero_vec_tensor2 = T.vector()
    zero_vec = np.zeros(dim)
    zero_vec2 = np.zeros(dim2)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    set_zero2 = theano.function([zero_vec_tensor2], updates=[(Words2, T.set_subtensor(Words2[0,:], zero_vec_tensor2))], allow_input_downcast=True)
    
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    best_test=0.0
    if num_epochs==-1:
        num_epochs=1000000
        test_loss_last=0.0
    for epoch in xrange(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch_inx in xrange(n_batches):
            train_err += train_fn(batch_inx)
            train_batches += 1
            set_zero(zero_vec)
            set_zero2(zero_vec2)

        test_loss_current,tpc=test_fn()
        tpc=np.array(tpc).tolist()

        f1_test=metrics.f1_score(test_y, tpc,average='macro')
        if f1_test>best_test:
            best_test=f1_test
            cr=metrics.precision_recall_fscore_support(test_y,tpc)
            best_perf=str(np.mean(cr[2]))+","+str(np.mean(cr[0]))+","+str(np.mean(cr[1]))+","+"details"+","+str(cr)
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:,{:.6f}".format(train_err / train_batches),"  test loss:,{:.6f}".format(test_loss_current*1.0))#format(test_err / test_batches))
        print("  test f1:,{:.6f}".format(f1_test))
        if num_epochs==1000000:
            if epoch==0:
                test_loss_last=test_loss_current
            elif test_loss_last-test_loss_current<1e-05:
                break
            else:
                test_loss_last=test_loss_current
    return best_perf
    
if __name__=="__main__":
    #print sys.argvs learning_rate,num_epochs,batch_size
    s=''
    for i in xrange(len(sys.argv)):
        s=s+' '+sys.argv[i]
    print s
    num_epochs=int(sys.argv[1])
    batch_size=int(sys.argv[2])
    np.random.seed(3333)
    print "loading data...",
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, class_num, max_len = x[0], x[1], x[2], x[3], int(x[4]), int(x[5])
    print "data loaded!"

    r = xrange(0,5)#5 fold    
    result=[]
    for i in r:
        print("cv-"+str(i))
        dataset = make_idx_data_cv(revs, word_idx_map, i,max_l=max_len, filter_h=5)
        best_perf=train_cnn(dataset,W, W2,class_num,
                            num_epochs=num_epochs,
                            batch_size=batch_size)
        result.append(best_perf)
    print("Result:")
    for i in r:
        print result[i]


























































