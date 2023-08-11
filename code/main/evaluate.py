import tensorflow as tf
import LTN2_tb_3 as ltn
import numpy as np
from LTN2_tb_3 import Forall, Exists, Equiv, Implies, And, Or, Not
import os
import pandas as pd

train_data_path = ''
test_data_path = ''
model_meta_path = ''
model_path = ''
np.random.seed(0)
# load data
###################################################
# positive ########################################
# single
single = np.load(train_data_path+'single.npy').astype('float32')

rule_0_train = np.load(train_data_path+'rule_0_train.npy').astype('float32')
rule_1_train = np.load(train_data_path+'rule_1_train.npy').astype('float32')

# negative ########################################

non_rule_0_train = np.load(train_data_path+'non_rule_0_train.npy').astype('float32')

###################################################
# load test_data
rule_0_test = np.load(test_data_path+'rule_0_test.npy').astype('float32')
rule_1_test = np.load(test_data_path+'rule_1_test.npy').astype('float32')

# negative ########################################

non_rule_0_test = np.load(test_data_path+'non_rule_0_test.npy').astype('float32')
###################################################
# tensor size parameters
nDimVar = 600
sDimVar = 300
pDimVar = 600
###################################################
# ltn variables
# single
x = ltn.variable('x',sDimVar)
y = ltn.variable('y',sDimVar)
z = ltn.variable('z',sDimVar)
# positive 
rule_0_train_ltn = ltn.variable('rule_0_train',pDimVar)
rule_1_train_ltn = ltn.variable('rule_1_train',pDimVar)

non_rule_0_train_ltn = ltn.variable('non_rule_0_train',pDimVar)
###################################################

# add single layers
x_nn = ltn.addlayer_xyz('x_nn',sDimVar,x)
y_nn = ltn.addlayer_xyz('y_nn',sDimVar,y)
z_nn = ltn.addlayer_xyz('z_nn',sDimVar,z)
# add layers basic positive
rule_0_nn = ltn.addlayer('rule_0_nn',pDimVar,rule_0_train_ltn)
rule_1_nn = ltn.addlayer('rule_1_nn',pDimVar,rule_1_train_ltn)

# add layers basic negative
nlxy_nn = ltn.addlayer('non_0_nn',pDimVar,non_rule_0_train_ltn)


# load data
nr_random_bbs = 25
def get_feed_dict_test():
    feed_dict = {}

    feed_dict[x] = single[np.random.choice(len(single), nr_random_bbs, replace=True)].astype(np.float32)
    feed_dict[y] = single[np.random.choice(len(single), nr_random_bbs, replace=True)].astype(np.float32)
    feed_dict[z] = single[np.random.choice(len(single), nr_random_bbs, replace=True)].astype(np.float32)

    feed_dict[rule_0_train_ltn] = rule_0_train[np.random.choice(len(rule_0_train), nr_random_bbs, replace=True)].astype(np.float32)
    feed_dict[rule_1_train_ltn] = rule_1_train[np.random.choice(len(rule_1_train), nr_random_bbs, replace=True)].astype(np.float32)
    

    feed_dict[non_rule_0_train_ltn] = non_rule_0_train_ltn[np.random.choice(len(rule_0_train),nr_random_bbs,replace=True)].astype(np.float32)
    return feed_dict_test


rule_0 = ltn.predicate("rule_0", pDimVar)
rule_1 = ltn.predicate("rule_1",pDimVar)
non_rule_0 = ltn.predicate("non_rule_0", pDimVar)

P = [rule_0]
inv_P = [rule_1]

pxy = [rule_0]
npxy = [non_rule_0]


P_basic = [rule_0]
inv_P_basic = [rule_1]

pxy_basic = [rule_0]
inv_P_npxy = [rule_1]


constraints = [Forall(pxy[i],P[i](pxy[i])) for i in range(10)]

constraints += [Forall(npxy[i],Not(P[i](npxy[i]))) for i in range(10)]

constraints += [Forall((x_nn,y_nn),Implies(P[i](x_nn,y_nn),inv_P[i](y_nn,x_nn)))
                for i in range(6)]

constraints += [Forall((x_nn,y_nn),Not(And(P[i](x_nn,y_nn),P[i](y_nn,x_nn))))
                for i in range(6)]



loss = -tf.reduce_min(tf.concat(constraints, axis=0))
loss_sum = tf.summary.scalar('loss', loss)
opt = tf.train.AdagradOptimizer(0.005).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    merged = tf.summary.merge_all()
    # training:
    
    feed_dict = get_feed_dict()
    feed_dict_test = get_feed_dict_test()
    
    saver = tf.train.import_meta_graph(model_meta_path)
    saver.restore(sess,tf.train.latest_checkpoint(model_path))

    def eval(test):
        result = sess.run([X(test) for X in P],feed_dict=feed_dict_test)
        return result
    def avg(lst):
        result = sum(lst)/len(lst)
        return result
    pred_rule_0 = []
    pred_rule_1 = []
    
    for i in range(50):

        test_rule_0 = eval(rule_0_nn)
        test_rule_1 = eval(rule_1_nn)


        pred_rule_0.append(test_rule_0[0][0][0])
        pred_rule_1.append(test_rule_1[1][0][0])
        

    print('pred_rule_0', avg(pred_rule_0))
    print('pred_rule_1', avg(pred_rule_1))
    