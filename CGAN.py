#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import argparse
import skimage


#load MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#devide train_set, test_set
#train_set = (mnist.train.images -0.5) / 0.5
#train_label = mnist.train.labels



def xavg_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


#training parameters
total_epoch = 100
batch_size = 100
learning_rate = 0.0002
# variables : input
n_hidden = 128
n_input = mnist.train.images.shape[1]
n_label = mnist.train.labels.shape[1]
n_noise = 100
print(n_label)
print(n_input,n_label)


#GAN does not use Y valiable beause it is unsupervised leraning
X = tf.placeholder(tf.float32, [None, n_input], name='X')
# but CGAN was using Y value this is put the condition information y
Y = tf.placeholder(tf.float32, [None, n_label], name='Y')
#noise Z is input
Z = tf.placeholder(tf.float32, [None, n_noise], name='Z')
is_train= tf.placeholder(dtype=tf.bool)


#Modify input to hidden weights for discriminator
D_W1 = tf.Variable(xavg_init([n_input + n_label , n_hidden]),name='D_W1')

#kind of variable for generator NN
#D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden],stddev=0.01),name = 'D_W1')
D_B1 = tf.Variable(tf.zeros([n_hidden]),name = 'D_B1')
#number 1 : how slimilerly to final result
D_W2 = tf.Variable(xavg_init([n_hidden, 1]), name = 'D_W2')
D_B2 = tf.Variable(tf.zeros([1]),name = 'D_B2')

D_var_list = [D_W1, D_B1, D_W2, D_B2]

print(D_var_list)



#Modifiy input to hidden weight for generator
G_W1 = tf.Variable(xavg_init([n_noise + n_label, n_hidden]),name='G_W1')

#Kind of varibable for desciminator NN
#G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden],stddev=0.01),name = 'G_W1')
G_B1 = tf.Variable(tf.zeros([n_hidden]),name = 'G_B1')
G_W2 = tf.Variable(xavg_init([n_hidden, n_input]),name = 'G_W2')
G_B2 = tf.Variable(tf.zeros([n_input]), name = 'G_B2')

G_var_list = [G_W1, G_B1, G_W2, G_B2]
print(G_var_list)




############################## G NN ################################
def generator(z,y):
 #   with tf.Variable_scope('generator',reuse=reuse):
#    with tf.VariableScope('generator'):
    inputs = tf.concat(axis=1, values=[z,y])
            #inputs = tf.contrib
            #hidden = tf.layers.dense(noisez,)
            #G_inputs = tf.concat(c_dim=1,values[z,y])

    hidden_G = tf.nn.relu(tf.matmul(inputs,G_W1) + G_B1)
            #hideen2 = tf.nn.relu(tf.matmul())
    log_G = tf.matmul(hidden_G,G_W2) + G_B2

    output_G = tf.nn.sigmoid(log_G)
    
    return output_G



############################## D NN ################################
def descriminator(x,y):
    #with tf.VariableScope('descriminator',reuse=reuse):
    #concatenate x and y
    inputs = tf.concat(axis=1, values=[x,y])

    hidden_D = tf.nn.relu(tf.matmul(inputs,D_W1) + D_B1)
    log_D = tf.matmul(hidden_D,D_W2) + D_B2
    output_D = tf.nn.sigmoid(log_D)
    
    return output_D, log_D



def get_noise(batch_size, n_noise):

    return np.random.normal(size=(batch_size, n_noise))



def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig



G = generator(Z, Y)
print(Z,Y,G)
real_D, log_real_D = descriminator(X, Y)
gene_D, log_gene_D = descriminator(G, Y)

loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                             (logits=log_real_D, labels=tf.ones_like(log_real_D)))
loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                             (logits=log_gene_D, labels=tf.zeros_like(log_gene_D)))
loss_D = loss_D_real + loss_D_gene
loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                        (logits=log_gene_D, labels=tf.ones_like(log_gene_D)))

train_D = tf.train.AdamOptimizer(learning_rate).minimize(loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(loss_G, var_list=G_var_list)



sess = tf.Session()
sess.run(tf.global_variables_initializer())



init = tf.global_variables_initializer()
if not os.path.exists('out/'):
    os.makedirs('out/')
    
i = 0

G_lossess = []
D_lossess = []
with tf.Session() as sess:
    sess.run(init)
    print('training start!')
    for epoch in range(total_epoch):
        #possible change the train.num_examles
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):

#             y_sample = np.zeros([16, n_label])
#             y_sample[:, 7] = 1

            #samples = sess.run(G, feed_dict={Z: noise, Y:y_sample})

#             fig = plot(samples)
#             plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
#             i += 1
#             plt.close(fig)
            
            
            #update generator, discriminator
            noise = get_noise(batch_size,n_noise)
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, loss_val_D = sess.run([train_D,loss_D],feed_dict={X:batch_xs,Z:noise,Y:batch_ys})
            _, loss_val_G = sess.run([train_G,loss_G],feed_dict={Z:noise, Y:batch_ys})

        print('Epoch:','%04d' % epoch, 
              'D loss: {:.4}'.format(loss_val_D),
              'G loss: {:.4}'.format(loss_val_G))

        
        ###### make image part###############3
        if epoch == 0 or (epoch + 1) % 10 == 0:
            
#             sample_size = 16
#             noise = get_noise(sample_size, n_noise)
#             y_sample = np.zeros(shape=[sample_size, n_label])
#             y_sample[:, 7] = 1

#             samples = sess.run(G, feed_dict={Z:noise, Y:y_sample})

#             fig = plot(samples)
#             plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
#             i += 1
#             plt.close(fig)
            
            ############save 10 array savefig ##########
            sample_size = 10
            noise = get_noise(sample_size, n_noise)
            y_sample = np.zeros([sample_size, n_label])
            samples = sess.run(G, feed_dict={Z:noise,Y:y_sample})
            
            fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

            for i in range(sample_size):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(samples[i], (28, 28)))

            plt.savefig('/home/cmpark/MRI/CmRefineGAN/result/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

print('finish!')  
          
