import tensorflow as tf
import LTN2_tb_3 as ltn
import numpy as np
from LTN2_tb_3 import Forall, Exists, Equiv, Implies, And, Or, Not
import os
import pandas as pd


# read data file from folder
data_path = ''
model_path = ''
log_Path = ''

# single object
single_voxel_data = np.load(data_path+'single.npy')
# positive basic relations
rule_0_data = np.load(data_path+'rule_0.npy')
rule_1_data = np.load(data_path+'rule_1.npy')

# negative basic relations
non_rule_0_data = np.load(data_path+'non_rule_0.npy')


nDimVar = 600
sDimVar = 300
pDimVar =  600


# single ltn data
single = single_voxel_data.astype('float32')



#positive basic relations ltn data
rule_0 = rule_0_data.astype("float32")
rule_1 = rule_1_data.astype("float32")

#negative basic relations ltn data
non_rule_0 = non_rule_0_data.astype('float32')


# ltn variables
x = ltn.variable('x',sDimVar)
y = ltn.variable('y',sDimVar)
z = ltn.variable('z',sDimVar)


rule_0 = ltn.variable("rule_0", pDimVar)
rule_1 = ltn.variable("rule_1", pDimVar)
non_rule_0 = ltn.variable("not_rule_0", pDimVar)

# nr_random_bbs = 50
nr_random_bbs = 1
def get_feed_dict():
    feed_dict = {}

    feed_dict[x] = single[np.random.choice(len(single), nr_random_bbs, replace=True)].astype(np.float32)
    feed_dict[y] = single[np.random.choice(len(single), nr_random_bbs, replace=True)].astype(np.float32)
    feed_dict[z] = single[np.random.choice(len(single), nr_random_bbs, replace=True)].astype(np.float32)
    return feed_dict


rule_0 = ltn.predicate("rule_0", pDimVar)
rule_1 = ltn.predicate("rule_1",pDimVar)






P = [rule_0]
inv_P = [rule_1]

pxy = [rule_0]
npxy = [non_rule_0]


P_basic = [rule_0]
inv_P_basic = [rule_1]

pxy_basic = [rule_0]
inv_P_npxy = [rule_1]



constraints =  [Forall(pxy[i],P[i](pxy[i])) for i in range(14)]

constraints += [Forall(npxy[i],Not(P[i](npxy[i]))) for i in range(14)]

constraints += [Forall((x,y),Implies(P_basic[i](x,y),inv_P_basic[i](y,x)))
                for i in range(6)]

constraints += [Forall((x,y),Not(And(P_basic[i](x,y),P_basic[i](y,x))))
                for i in range(6)]





loss = -tf.reduce_min(tf.concat(constraints, axis=0))
opt = tf.train.AdagradOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()






with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    merged = tf.summary.merge_all()
    # training:
    sess.run(init)
    feed_dict = get_feed_dict()
    test_writer = tf.summary.FileWriter(log_Path, graph_def=sess.graph_def)
    for i in range(100000):
        sess.run(opt,feed_dict=feed_dict)
        if i % 100 == 0:
            sat_level=sess.run(-loss, feed_dict=feed_dict)
            saver.save(sess,log_Path+'model.ckpt',global_step=1000)
            print(i, "sat level ----> ", sat_level)
            if sat_level > .99:
                break


