import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time
import h5py

"""
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

"""
class Params():
    def __init__(init_scale=0.1,learning_rate=1.0,max_grad_norm = 5,
    num_steps = 20,
    hidden_size = 200,
    max_epoch = 4,
    max_max_epoch = 13,
    keep_prob = 1.0,
    lr_decay = 0.5,
    batch_size = 20,
    vocab_size = 10000):

        self.init_scale = init_scale
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.max_epoch = max_epoch
        self.max_max_epoch = max_max_epoch
        self.keep_prob = keep_prob
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.vocab_size = vocab_size



def inference(x, is_training, num_steps, reuse=None):

  print("\nnum_steps : %d, is_training : %s, reuse : %s" %
                                                (num_steps, is_training, reuse))
  initializer = tf.random_uniform_initializer(-init_scale, init_scale)
  with tf.variable_scope("model", reuse=reuse):
    tl.layers.set_name_reuse(reuse)
    network = tl.layers.EmbeddingInputlayer(
                inputs = x,
                vocabulary_size = vocab_size,
                embedding_size = hidden_size,
                E_init = tf.random_uniform_initializer(-init_scale, init_scale),
                name ='embedding')
    network = tl.layers.DropoutLayer(network, keep=keep_prob, is_fix=True, is_train=is_training, name='drop1')
    network = tl.layers.RNNLayer(network,
                cell_fn=tf.contrib.rnn.BasicLSTMCell, #tf.nn.rnn_cell.BasicLSTMCell,
                cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True},
                n_hidden=hidden_size,
                initializer=tf.random_uniform_initializer(-init_scale, init_scale),
                initial_state=None,
#                initial_state=tf.contrib.rnn.LSTMStateTuple(tf.constant(dtype=tf.float32,value=np.random.randn(batch_size,hidden_size)),tf.constant(dtype=tf.float32,value=np.random.randn(batch_size,hidden_size))),
                n_steps=num_steps,
                return_last=False,
                name='basic_lstm1')
    lstm1 = network
    network = tl.layers.DropoutLayer(network, keep=keep_prob, is_fix=True, is_train=is_training, name='drop2')
    network = tl.layers.RNNLayer(network,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,#tf.nn.rnn_cell.BasicLSTMCell,
                cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True},
                n_hidden=hidden_size,
                initializer=tf.random_uniform_initializer(-init_scale, init_scale),
                initial_state=None,
                # initial_state=tf.contrib.rnn.LSTMStateTuple(tf.constant(dtype=tf.float32,value=np.random.randn(batch_size,hidden_size)),tf.constant(dtype=tf.float32,value=np.random.randn(batch_size,hidden_size))),
                n_steps=num_steps,
                return_last=False,
                return_seq_2d=True,
                name='basic_lstm2')
    lstm2 = network
      
      # network = tl.layers.ReshapeLayer(network,
      #       shape=[-1, int(network.outputs._shape[-1])], name='reshape')
    network = tl.layers.DropoutLayer(network, keep=keep_prob, is_fix=True, is_train=is_training, name='drop3')
    network = tl.layers.DenseLayer(network,
                  n_units=vocab_size,
                  W_init=tf.random_uniform_initializer(-init_scale, init_scale),
                  b_init=tf.random_uniform_initializer(-init_scale, init_scale),
                  act = tf.identity, name='output')
    return network, lstm1, lstm2

 

def loss_fn(outputs, targets):
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [outputs],
        [tf.reshape(targets, [-1])],
        [tf.ones_like(tf.reshape(targets, [-1]), dtype=tf.float32)])
    cost = tf.reduce_sum(loss) / batch_size
    return cost

def loss_fn_weights(outputs, targets, weights):
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [outputs],
        [tf.reshape(targets, [-1])],
        [tf.ones_like(tf.reshape(targets, [-1]), dtype=tf.float32)])
    cost = tf.reduce_sum(loss) / batch_size
    return cost

# def accuracy(outputs,targets):
#     pred_targets=tf.argmax(outputs,1);
#     bool_arr=tf.equal(tf.cast(pred_targets,tf.int32),tf.reshape(targets,[-1]));
#     acc=tf.reduce_sum(tf.cast(bool_arr,tf.float32));
#     return acc;
def accuracy(outputs,targets):
    pred_targets=tf.argmax(outputs,1);
    bool_arr=tf.equal(tf.cast(tf.reshape(pred_targets,targets.shape),tf.int32),targets);
    return bool_arr;


def Train(sess,train_data,network,lstm1,lstm2):
        new_lr_decay = lr_decay ** max(i - max_epoch, 0.0);    #decay initial learning rate
        sess.run(tf.assign(lr, learning_rate * new_lr_decay))

        print("Epoch: %d/%d Learning rate: %.3f" % (i + 1, max_max_epoch, sess.run(lr)))
        epoch_size = ((len(train_data) // batch_size) - 1) // num_steps
        start_time = time.time()
        costs = 0.0; iters = 0
        train_acc=0.0;
        # reset all states at the begining of every epoch
        state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
        state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)
        for step, (x, y) in enumerate(tl.iterate.ptb_iterator(train_data,
                                                    batch_size, num_steps)):
            feed_dict = {input_data: x, targets: y,
                        lstm1.initial_state.c: state1[0],
                        lstm1.initial_state.h: state1[1],
                        lstm2.initial_state.c: state2[0],
                        lstm2.initial_state.h: state2[1],
                        }
            # For training, enable dropout
            feed_dict.update( network.all_drop )
            _train_acc,_cost, state1_c, state1_h, state2_c, state2_h, _ = sess.run([acc_train,cost,
                                            lstm1.final_state.c,
                                            lstm1.final_state.h,
                                            lstm2.final_state.c,
                                            lstm2.final_state.h,
                                            train_op],
                                            feed_dict=feed_dict
                                            )
            state1 = (state1_c, state1_h)
            state2 = (state2_c, state2_h)

            costs += _cost; iters += num_steps
            train_acc += _train_acc/(batch_size*num_steps);
            print "train_iter:",step,"acc:",train_acc/step,"this time:",_train_acc,"/",batch_size*num_steps;
            if step % (epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                    (step * 1.0 / epoch_size, np.exp(costs / iters),
                    iters * batch_size / (time.time() - start_time)))
        train_perplexity = np.exp(costs / iters);
        return train_perplexity,train_acc/step;
#def adamodel_acc(sess,train_data,network,lstm1,lstm2,n_models):
    



def ada_data_eval(sess,data,network,lstm1,lstm2,nboost_iters,model):

    ada_accum=[];
    goals=[];
    accur=[];
    acc_list=[];
    for j in range(nboost_iters):
        load_params = tl.files.load_npz(name='adabst_model_'+str(model)+'_iteration_'+str(j)+'.npz');
        tl.files.assign_params(sess, load_params, network);
        state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
        state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)
        step=0;accum=[];accs=[];

        for (x, y) in tl.iterate.ptb_iterator(valid_data,batch_size, num_steps):
            step=step+1;
            feed_dict = {input_data: x, targets: y,
                            lstm1.initial_state.c: state1[0],
                            lstm1.initial_state.h: state1[1],
                            lstm2.initial_state.c: state2[0],
                            lstm2.initial_state.h: state2[1],
                            }
            [acc,pr,state1_c, state1_h, state2_c, state2_h]=sess.run([acc_valid,pred_targets_valid,lstm1.final_state.c,
                                lstm1.final_state.h,
                                lstm2.final_state.c,
                                lstm2.final_state.h],feed_dict=feed_dict);
            state1 = (state1_c, state1_h)
            state2 = (state2_c, state2_h)
            # make outs one-hot
            # er=np.zeros((batch_size*num_steps,10000));
            # er[range(batch_size*num_steps),pr]=1;
            # accum.append(alphas[j]*er);
            accum.append(pr);
            accs.append(acc);
            if(j==nboost_iters-1):
                goals.append(y.reshape(-1));
            print "Adabst model:",j,", Batch:",step,", Accuracy:",acc,"/",batch_size*num_steps;
        acc_list.append(np.stack(accs));
        ada_accum.append(np.stack(accum));
    roger=np.stack(ada_accum);
    goals=np.array(goals);
    alphas=np.load('alphas_'+str(model)+'.npy');
    h5f = h5py.File('model_data_'+str(model)+'.h5', 'w')
    h5f.create_dataset('roger', data=roger)
    h5f.create_dataset('goals', data=goals)
    h5f.create_dataset('alphas', data=alphas)
    h5f.create_dataset('model_accuracies', data=np.stack(acc_list))
    h5f.close()

def predict_adaboost(model):
    h5f = h5py.File('model_data_'+str(model)+'.h5', 'r')
    roger=h5f['roger'][:];
    alphas=h5f['alphas'][:];
    goals=h5f['goals'][:];
    accur=h5f['model_accuracies'][:];
    h5f.close()
    step=roger.shape[1];
    nboost_iters=roger.shape[0];
    bsxns=roger.shape[2];
    pred=np.zeros((nboost_iters,step,bsxns));
    for a in range(step):
        for b in range(bsxns):
            for y in range(nboost_iters):
                er=np.zeros((y+1,10000))
                er[range(y+1),roger[:y+1,a,b]]=alphas[:y+1];
                pred[y,a,b]=np.argmax(np.sum(er,0));             #adaboost predictions using ada-models 1:y
    # for a in range(step):
    #     accur.append(len(np.where(pred[a,:]==goals[a,:])[0]));
    h5f = h5py.File('model_data_'+str(model)+'.h5', 'a')
    h5f.create_dataset('adabst_predictions', data=pred);
    # h5f.create_dataset('adabst_accuracies', data=np.array(accur));
    h5f.close()
    # preds=np.argmax(np.array(accum),2);
    # return accur;

def ada_data_eval1(sess,data,alphas,network,lstm1,lstm2,nboost_iters):
    
    accum=[];
    goals=[];
    nboost_iters=15;
    cond=0;
    for (x, y) in tl.iterate.ptb_iterator(data,batch_size, num_steps):
        goals.append(y.reshape(-1));
        adder=np.zeros((batch_size*num_steps,10000));

        for j in range(nboost_iters):
            load_params = tl.files.load_npz(name='adabst_model_1_iteration_'+str(j+1)+'.npz');
            tl.files.assign_params(sess, load_params, network);
            state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
            state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)
            feed_dict = {input_data: x, targets: y,
                            lstm1.initial_state.c: state1[0],
                            lstm1.initial_state.h: state1[1],
                            lstm2.initial_state.c: state2[0],
                            lstm2.initial_state.h: state2[1],
                            }
            [pr,state1_c, state1_h, state2_c, state2_h]=sess.run([pred_targets_valid,lstm1.final_state.c,
                                lstm1.final_state.h,
                                lstm2.final_state.c,
                                lstm2.final_state.h],feed_dict=feed_dict);

            # make outs one-hot
            # er=np.zeros((batch_size*num_steps,10000));
            # er[range(batch_size*num_steps),pr]=1;
            # adder=adder+alphas[j]*er;
            accum.append(pr);
        accum.append(adder)
    preds=np.argmax(np.array(accum),2);
    goals=np.array(goals);
    accur=np.where(preds!=goals)[0]
    return accur;
def eval_traindata(sess,train_data,network,lstm1,lstm2):
        train_acc=0.0;step=0;
        state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
        state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)
        batch_acc=[];

        for (x, y) in tl.iterate.ptb_iterator(train_data,batch_size, num_steps):
            feed_dict = {input_data: x, targets: y,
                        lstm1.initial_state.c: state1[0],
                        lstm1.initial_state.h: state1[1],
                        lstm2.initial_state.c: state2[0],
                        lstm2.initial_state.h: state2[1],
                        }
            # For training, enable dropout
            #feed_dict.update( network.all_drop )
            _train_acc, state1_c, state1_h, state2_c, state2_h = sess.run([acc_valid,
                                            lstm1.final_state.c,
                                            lstm1.final_state.h,
                                            lstm2.final_state.c,
                                            lstm2.final_state.h,
                                            ],feed_dict=feed_dict)
            state1 = (state1_c, state1_h)
            state2 = (state2_c, state2_h)
            batch_acc.append(_train_acc/(batch_size*num_steps));
            train_acc += _train_acc*1.0/(batch_size*num_steps);
            step=step+1;
            print "Complete train acc:\n",step,"acc:",train_acc/step,"this time:",_train_acc,"/",batch_size*num_steps;
        return train_acc/(step+1),np.array(batch_acc);

def Train_new(sess,train_data,network,lstm1,lstm2):
        start_time = time.time()
        costs = 0.0; iters = 0
        step=0;
        train_acc=0.0;
        state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
        state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)
        for (x, y) in train_data:
            feed_dict = {input_data: x, targets: y,
                        lstm1.initial_state.c: state1[0],
                        lstm1.initial_state.h: state1[1],
                        lstm2.initial_state.c: state2[0],
                        lstm2.initial_state.h: state2[1],
                        }
            # For training, enable dropout
            feed_dict.update( network.all_drop )
            _train_acc,_cost, state1_c, state1_h, state2_c, state2_h, _ = sess.run([acc_train,cost,
                                            lstm1.final_state.c,
                                            lstm1.final_state.h,
                                            lstm2.final_state.c,
                                            lstm2.final_state.h,
                                            train_op],
                                            feed_dict=feed_dict
                                            )
            state1 = (state1_c, state1_h)
            state2 = (state2_c, state2_h)
            #_train_acc=np.sum(train_acc_bool);
            costs += _cost; iters += num_steps;
            train_acc += _train_acc*1.0/(batch_size*num_steps);
            step=step+1;
            print "Train_Batch No:",step,"acc till now:",train_acc/step,"acc this batch:",_train_acc,"/",batch_size*num_steps;
            #if step % (epoch_size // 10) == 10:
                # print("%.3f perplexity: %.3f speed: %.0f wps" %
                #     (step * 1.0 / epoch_size, np.exp(costs / iters),
                #     iters * batch_size / (time.time() - start_time)))
        train_perplexity = np.exp(costs / iters);
        return train_perplexity,train_acc/step;       
def Train_ada1(sess,train_data,network,lstm1,lstm2):
        model=2;
        data=[];target=[];
        split_ratio=0.3;
        nboost_iters=15;
        for (x, y) in tl.iterate.ptb_iterator(train_data,batch_size, num_steps):
            data.append(x);
            target.append(y);
        n_trains=len(data);
        n_trainchunks=int(split_ratio*n_trains);
        alpha=[];
        prob_list=[np.ones(n_trains)/n_trains];
        # reset all states at the begining of every epoch
        state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
        state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)
        for j in range(nboost_iters):
            # tl.files.save_npz(network.all_params , name='adabst_model_'+str(model)+'_iteration_'+str(j)+'.npz');   
            prob=prob_list[j];
            cum_prob = np.cumsum(prob);
            sample_indices = np.searchsorted(cum_prob, np.random.rand(n_trainchunks));
            train_chunks=[[data[ind],target[ind]]for ind in sample_indices];

            for ep in range(12):
                new_lr_decay = lr_decay ** max(ep - max_epoch, 0.0);    #decay initial learning rate
                sess.run(tf.assign(lr, learning_rate * new_lr_decay))
                train_perplexity, train_acc=Train_new(sess,train_chunks,network,lstm1,lstm2);
                print("Epoch: %d/%d Train Perplexity: %.3f, Epoch_train_acc: %.3f" % (ep + 1, max_max_epoch,train_perplexity,train_acc))
            com_train_acc,batch_acc=eval_traindata(sess,train_data,network,lstm1,lstm2);
            tl.files.save_npz(network.all_params , name='adabst_model_'+str(model)+'_iteration_'+str(j+1)+'.npz');
            t=0.3;
            mask=batch_acc<t;

            print "Updating data distribution\n";
            #update err, weights and alphas
            eps=np.sum(prob[mask]);
            alpha.append(np.log((1-eps)/eps)+np.log(9999));
            np.save('prob_'+str(j),prob);
            prob=np.multiply(prob,np.exp(alpha[j]*mask));
            prob=np.multiply(prob,np.exp(-1*alpha[j]*np.logical_not(mask)));
            prob=prob/np.sum(prob);
            prob_list.append(prob);
def Train_ada(sess,train_data,network,lstm1,lstm2,model,nboost_iters):
    data=[];target=[];
    split_ratio=0.3;
    for (x, y) in tl.iterate.ptb_iterator(train_data,batch_size, num_steps):
        data.append(x);
        target.append(y);
    n_trains=len(data);
    n_trainchunks=int(split_ratio*n_trains);
    alpha=[];
    prob_list=[np.ones(n_trains)/n_trains];
    # reset all states at the begining of every epoch
    for j in range(nboost_iters):
        #   
        prob=prob_list[j];
        np.save('prob_'+str(j),prob);
        cum_prob = np.cumsum(prob);
        sample_indices = np.searchsorted(cum_prob, np.random.rand(n_trainchunks));
        train_chunks=[[data[ind],target[ind]]for ind in sample_indices];
        sess.run(tf.variables_initializer(tf.trainable_variables()))   
        # state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
        # state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)

        for ep in range(2):
            new_lr_decay = lr_decay ** max(ep - max_epoch, 0.0);    #decay initial learning rate
            sess.run(tf.assign(lr, learning_rate * new_lr_decay))
            train_perplexity, train_acc=Train_new(sess,train_chunks,network,lstm1,lstm2);
            print("Epoch: %d/%d Train Perplexity: %.3f, Epoch_train_acc: %.3f" % (ep + 1, max_max_epoch,train_perplexity,train_acc))
        com_train_acc,batch_acc=eval_traindata(sess,train_data,network,lstm1,lstm2);
        t=0.2;
        mask=batch_acc<t;
        tl.files.save_npz(network.all_params , name='adabst_model_'+str(model)+'_iteration_'+str(j)+'.npz'); 
        print "Updating data distribution\n";
        #update err, weights and alphas
        eps=np.sum(prob[mask]);
        alpha.append(np.log((1-eps)/eps)+np.log(int((1-split_ratio)*n_trainchunks)));
        prob=np.multiply(prob,np.exp(alpha[j]*mask));
        prob=np.multiply(prob,np.exp(-1*alpha[j]*np.logical_not(mask)));
        prob=prob/np.sum(prob);
        prob_list.append(prob);
        np.save('alphas_'+str(model)+'.npy',np.array(alpha));
    

def Validate_ada(sess,valid_data,network_val,lstm1_val,lstm2_val):
        start_time = time.time()
        costs = 0.0; iters = 0
        valid_acc=0.0;
        # reset all states at the begining of every epoch

        state1 = tl.layers.initialize_rnn_state(lstm1_val.initial_state)
        state2 = tl.layers.initialize_rnn_state(lstm2_val.initial_state)
        for step, (x, y) in enumerate(tl.iterate.ptb_iterator(valid_data,
                                                    batch_size, num_steps)):
            feed_dict = {input_data: x, targets: y,
                        lstm1_val.initial_state.c: state1[0],
                        lstm1_val.initial_state.h: state1[1],
                        lstm2_val.initial_state.c: state2[0],
                        lstm2_val.initial_state.h: state2[1],
                        }
            _valid_acc,_cost, state1_c, state1_h, state2_c, state2_h, _ = \
                                    sess.run([acc_valid,cost_val,
                                            lstm1_val.final_state.c,
                                            lstm1_val.final_state.h,
                                            lstm2_val.final_state.c,
                                            lstm2_val.final_state.h,
                                            tf.no_op()],
                                            feed_dict=feed_dict
                                            )
            state1 = (state1_c, state1_h)
            state2 = (state2_c, state2_h)
            costs += _cost; iters += num_steps
            valid_acc+=_valid_acc/(batch_size*num_steps);
            print "valid_iter:",step,"acc:",valid_acc/step,"this time:",_valid_acc,"/",batch_size*num_steps;
        valid_perplexity = np.exp(costs / iters);
        return valid_perplexity,valid_acc/step;

def Test_ada(sess,test_data,network_test,lstm1_test,lstm2_test):
    print("Evaluation")
    # Testing
    # go through the test set step by step, it will take a while.
    costs = 0.0; iters = 0;
    test_acc=0.0;
    # reset all states at the begining
    state1 = tl.layers.initialize_rnn_state(lstm1_test.initial_state)
    state2 = tl.layers.initialize_rnn_state(lstm2_test.initial_state)
    for step, (x, y) in enumerate(tl.iterate.ptb_iterator(test_data,
                                            batch_size=1, num_steps=1)):
        feed_dict = {input_data_test: x, targets_test: y,
                    lstm1_test.initial_state.c: state1[0],
                    lstm1_test.initial_state.h: state1[1],
                    lstm2_test.initial_state.c: state2[0],
                    lstm2_test.initial_state.h: state2[1],
                    }
        _test_acc,_cost, state1_c, state1_h, state2_c, state2_h = \
                                sess.run([acc_test,cost_test,
                                        lstm1_test.final_state.c,
                                        lstm1_test.final_state.h,
                                        lstm2_test.final_state.c,
                                        lstm2_test.final_state.h,
                                        ],
                                        feed_dict=feed_dict
                                        )
        state1 = (state1_c, state1_h)
        state2 = (state2_c, state2_h)
        costs += _cost; iters += 1
        test_acc+=_test_acc;
        print "test_iter:",step,"acc:",test_acc/step;
    test_perplexity = np.exp(costs / iters);
    return test_perplexity,test_acc/step;



init_scale=0.1;
learning_rate=1.0;
max_grad_norm = 5,
num_steps = 20;
hidden_size = 200;
max_epoch = 4;
max_max_epoch = 13;
keep_prob = 1;
lr_decay = 0.5;
batch_size = 10;
vocab_size = 10000;
# Load PTB dataset
train_data, valid_data, test_data, vocab_size = tl.files.load_ptb_dataset()
# train_data = train_data[0:int(100000/5)]    # for fast testing
# print('len(train_data) {}'.format(len(train_data))) # 929589 a list of int
# print('len(valid_data) {}'.format(len(valid_data))) # 73760  a list of int
# print('len(test_data)  {}'.format(len(test_data)))  # 82430  a list of int
# print('vocab_size      {}'.format(vocab_size))      # 10000
sess = tf.InteractiveSession()
input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
targets = tf.placeholder(tf.int32, [batch_size, num_steps])
input_data_test = tf.placeholder(tf.int32, [1, 1])
targets_test = tf.placeholder(tf.int32, [1, 1])


network, lstm1, lstm2 = inference(input_data,is_training=True, num_steps=num_steps, reuse=None)


# Inference for Validating
network_val, lstm1_val, lstm2_val = inference(input_data,is_training=False, num_steps=num_steps, reuse=True)
# Inference for Testing (Evaluation)
network_test, lstm1_test, lstm2_test = inference(input_data_test,is_training=False, num_steps=1, reuse=True)


cost = loss_fn(network.outputs, targets)
cost_val = loss_fn(network_val.outputs, targets)
cost_test = loss_fn(network_test.outputs, targets_test);#, 1, 1)


pred_targets_train=tf.argmax(network.outputs,1);
train_acc_bool=tf.equal(tf.cast(tf.reshape(pred_targets_train,targets.shape),tf.int32),targets);
acc_train=tf.reduce_sum(tf.cast(train_acc_bool,tf.float32));

pred_targets_valid=tf.argmax(network_val.outputs,1);
valid_acc_bool=tf.equal(tf.cast(tf.reshape(pred_targets_valid,targets.shape),tf.int32),targets);
acc_valid=tf.reduce_sum(tf.cast(valid_acc_bool,tf.float32));

pred_targets_test=tf.argmax(network_test.outputs,1);
test_acc_bool=tf.equal(tf.cast(tf.reshape(pred_targets_test,targets_test.shape),tf.int32),targets_test);
acc_test=tf.reduce_sum(tf.cast(test_acc_bool,tf.float32));
# acc_train=accuracy(network.outputs,targets);
# acc_valid=accuracy(network_val.outputs,targets);
# acc_test=accuracy(network_test.outputs,targets_test);

with tf.variable_scope('learning_rate'):
    lr = tf.Variable(0.0, trainable=False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),max_grad_norm)
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.apply_gradients(zip(grads, tvars))

tl.layers.initialize_global_variables(sess)

# network.print_params()
# network.print_layers()
# tl.layers.print_all_variables()

# load_params = tl.files.load_npz(name='model_1_chkpoint_12.npz');
# tl.files.assign_params(sess, load_params, network);

#model dropouts
# 1: 1
# 2: 0.2
# 3: 0.47
# 4: 0.74

# for i in range(max_max_epoch):
#     train_perplexity, train_acc=Train(sess,train_data,network,lstm1,lstm2);
#     print("Epoch: %d/%d Train Perplexity: %.3f, Epoch_train_acc: %.3f" % (i + 1, max_max_epoch,train_perplexity,train_acc))

#     valid_perplexity, valid_acc=Validate(sess,valid_data[:20000],network_val,lstm1_val,lstm2_val);
#     print("Epoch: %d/%d Valid Perplexity: %.3f, Epoch_valid_acc: %.3f" % (i + 1, max_max_epoch,
#                                                         valid_perplexity,valid_acc))
#     start_time = time.time()
#     test_perplexity, test_acc=Test(sess,test_data[:20000],network_test,lstm1_test,lstm2_test);
#     print("Epoch: %d/%d Test Perplexity: %.3f took %.2fs, test_acc: %.3f" % (i + 1, max_max_epoch,test_perplexity, time.time() - start_time, test_acc));

#     tl.files.save_npz(network.all_params , name='model_'+str(j)+'_chkpoint_'+str(i)+'.npz');                                                 

# model=1;
# data=[];target=[];
# split_ratio=0.3;
# nboost_iters=15;
# for (x, y) in tl.iterate.ptb_iterator(train_data,batch_size, num_steps):
#     data.append(x);
#     target.append(y);
# n_trains=len(data);
# n_trainchunks=int(split_ratio*n_trains);
# alpha=[];
# prob_list=[np.ones(n_trains)/n_trains];
# # reset all states at the begining of every epoch
# for j in range(nboost_iters):
#     #   
#     prob=prob_list[j];
#     np.save('prob_'+str(j),prob);
#     cum_prob = np.cumsum(prob);
#     sample_indices = np.searchsorted(cum_prob, np.random.rand(n_trainchunks));
#     train_chunks=[[data[ind],target[ind]]for ind in sample_indices];
#     sess.run(tf.variables_initializer(tf.trainable_variables()))   
#     # state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
#     # state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)

#     for ep in range(3):
#         new_lr_decay = lr_decay ** max(ep - max_epoch, 0.0);    #decay initial learning rate
#         sess.run(tf.assign(lr, learning_rate * new_lr_decay))
#         train_perplexity, train_acc=Train_new(sess,train_chunks,network,lstm1,lstm2);
#         print("Epoch: %d/%d Train Perplexity: %.3f, Epoch_train_acc: %.3f" % (ep + 1, max_max_epoch,train_perplexity,train_acc))
#     com_train_acc,batch_acc=eval_traindata(sess,train_data,network,lstm1,lstm2);
#     t=0.22;
#     mask=batch_acc<t;
#     tl.files.save_npz(network.all_params , name='adabst_model_'+str(model)+'_iteration_'+str(j)+'.npz'); 
#     print "Updating data distribution\n";
#     #update err, weights and alphas
#     eps=np.sum(prob[mask]);
#     alpha.append(np.log((1-eps)/eps)+np.log(4646));
#     prob=np.multiply(prob,np.exp(alpha[j]*mask));
#     prob=np.multiply(prob,np.exp(-1*alpha[j]*np.logical_not(mask)));
#     prob=prob/np.sum(prob);
#     prob_list.append(prob);
# np.save('alphas.npy',np.array(alpha));
model=1;
nboost_iters=15;
# Train_ada(sess,train_data,network,lstm1,lstm2,model,nboost_iters);
# alphas=np.load('alphas_'+str(model)+'.npy');
ada_data_eval(sess,valid_data,network_val,lstm1_val,lstm2_val,nboost_iters,model);
# predict_adaboost(model);
model=2;
ada_data_eval(sess,valid_data,network_val,lstm1_val,lstm2_val,nboost_iters,model);
# predict_adaboost(model);


