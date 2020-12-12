import tensorflow as tf
import numpy as np
import sys
sys.path.append('../utils')
sys.path.append('../vgg19')
from layer import *
from BasicConvLSTMCell import *
#from BasicConvLSTMCell1 import *
#from BasicConvLSTMCell1 import BasicConvLSTMCell1
from vgg19 import VGG19

class SRGAN:
    def __init__(self, x, x_com, is_training, batch_size):
        self.batch_size = batch_size
        self.vgg = VGG19(None, None, None)
        self.edge_hr = self.Laplacian(x)
        #self.detail_hr = x - self.frame_hr
        #self.downscaled = self.downscale(x)
        self.bic_ref = tf.image.resize_images(x_com, [self.image_size*8, self.image_size*8], method=2)
        self.att_edge, self.x_cnnmask, self.x_ganmask, self.CNN_sr, self.GAN_sr, self.RNN_sr, self.att_sr  = self.generator(x_com, is_training, False)
        #self.frame_sr, self.detail_sr, self.imitation_sr = self.generator_frame(self.downscaled, is_training, False)
        self.real_output = self.discriminator(x, is_training, False)
        self.fake_output = self.discriminator(self.GAN_sr, is_training, True)
        #self.real_output_att = self.discriminator_att(x, is_training, False)
        #self.fake_output_att = self.discriminator_att(self.att_sr, is_training, True)
        #seed_x = np.random.randint(0,32,size=1)
        #seed_y = np.random.randint(0,32,size=1)
        #self.real_base = self.discriminator(x, is_training, True)
        #self.fake_base = self.discriminator(self.base_sr, is_training, True)
        #self.real_frame = self.discriminator(self.frame_hr, is_training, True)
        #self.fake_frame = self.discriminator(self.frame_sr, is_training, True)self.CNN_edge, self.GAN_edge, self.real_output, self.fake_output, 
        self.g_loss, self.d_loss = self.inference_losses(self.edge_hr, self.att_edge, x, self.CNN_sr, self.GAN_sr, self.RNN_sr, self.att_sr, self.real_output, self.fake_output)
    image_size =  16#self.frame_hr, self.frame_sr,, self.g_frame_loss, self.d_frame_loss
    #image_size1 =  96self.real_base, self.fake_base,, self.real_output_att, self.fake_output_att
    #image_size0 =  16
    def generator(self, x_com, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):
			# CNN框架
            with tf.variable_scope('cnn1'):
                x_cnn = deconv_layer(
                    x_com, [3, 3, 64, 3], [self.batch_size, self.image_size, self.image_size, 64], 1)
                x_cnn = prelu(x_cnn)
            #shortcut = x
            for i in range(6):
                with tf.variable_scope('block{}cnn1'.format(i+1)):
                    x1=x2=x3=x_cnn
                    for j in range(3):
                        with tf.variable_scope('block{}_{}cnn1'.format(i+1,j+1)):
                            with tf.variable_scope('ud1'):
                                a1 = prelu(deconv_layer(x1, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #a1 = batch_normalize(a1, is_training)
                            with tf.variable_scope('ud2'):
                                b1 = prelu(deconv_layer(x2, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #b1 = batch_normalize(b1, is_training)
                            with tf.variable_scope('ud3'):
                                c1 = prelu(deconv_layer(x3, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #c1 = batch_normalize(c1, is_training)
                            sum = tf.concat([a1,b1,c1],3)
                            #sum = batch_normalize(sum, is_training)
                            with tf.variable_scope('ud4'):
                                x1 = prelu(deconv_layer(tf.concat([sum,x1],3), [1, 1, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #x1 = batch_normalize(x1, is_training)
                            with tf.variable_scope('ud5'):
                                x2 = prelu(deconv_layer(tf.concat([sum,x2],3), [1, 1, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #x2 = batch_normalize(x2, is_training)
                            with tf.variable_scope('ud6'):
                                x3 = prelu(deconv_layer(tf.concat([sum,x3],3), [1, 1, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #x3 = batch_normalize(x3, is_training)
                    with tf.variable_scope('ud7'):
                        block_out = prelu(deconv_layer(tf.concat([x1, x2, x3],3), [3, 3, 64, 192], [self.batch_size, self.image_size, self.image_size, 64], 1))
                    #with tf.variable_scope('ud8'):
                        #block_out_att = deconv_layer(block_out, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1)
                    #block_out_att = tf.nn.sigmoid(block_out_att)
                    #block_out = block_out_att*block_out + block_out
                    #x = x1+x2+x3+x
                    x_cnn+=block_out
                    #x = batch_normalize(x, is_training))
            with tf.variable_scope('cnn6'):
                x_cnn = deconv_layer(
                    x_cnn, [3, 3, 256, 64], [self.batch_size, self.image_size, self.image_size, 256], 1)#2
                x_cnn = pixel_shuffle_layerg(x_cnn, 2, 64) # n_split = 256 / 2 ** 2
                x_cnn = prelu(x_cnn)
            
            with tf.variable_scope('cnn7'):
                x_cnn = deconv_layer(
                    x_cnn, [3, 3, 256, 64], [self.batch_size, self.image_size*2, self.image_size*2, 256], 1)#2
                x_cnn = pixel_shuffle_layerg(x_cnn, 2, 64) # n_split = 256 / 2 ** 2
                x_cnn = prelu(x_cnn)
                
            with tf.variable_scope('cnn8'):
                x_cnn = deconv_layer(
                    x_cnn, [3, 3, 256, 64], [self.batch_size, self.image_size*4, self.image_size*4, 256], 1)#2
                x_cnn = pixel_shuffle_layerg(x_cnn, 2, 64) # n_split = 256 / 2 ** 2
                x_cnn = prelu(x_cnn)
                
            with tf.variable_scope('cnn9'):
                x_cnn = deconv_layer(
                    x_cnn, [3, 3, 3, 64], [self.batch_size, self.image_size*8, self.image_size*8, 3], 1)
            x_cnn_SR = x_cnn + self.bic_ref
            x_cnn_edge = self.Laplacian(x_cnn_SR)
            
			
			# GAN框架
            with tf.variable_scope('gan1'):
                x_gan = deconv_layer(
                    x_com, [3, 3, 64, 3], [self.batch_size, self.image_size, self.image_size, 64], 1)
                x_gan = prelu(x_gan)
            #shortcut = x
            for i in range(6):
                with tf.variable_scope('block{}gan1'.format(i+1)):
                    x1=x2=x3=x_gan
                    for j in range(3):
                        with tf.variable_scope('block{}_{}gan1'.format(i+1,j+1)):
                            with tf.variable_scope('ud1'):
                                a1 = prelu(deconv_layer(x1, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #a1 = batch_normalize(a1, is_training)
                            with tf.variable_scope('ud2'):
                                b1 = prelu(deconv_layer(x2, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #b1 = batch_normalize(b1, is_training)
                            with tf.variable_scope('ud3'):
                                c1 = prelu(deconv_layer(x3, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #c1 = batch_normalize(c1, is_training)
                            sum = tf.concat([a1,b1,c1],3)
                            #sum = batch_normalize(sum, is_training)
                            with tf.variable_scope('ud4'):
                                x1 = prelu(deconv_layer(tf.concat([sum,x1],3), [1, 1, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #x1 = batch_normalize(x1, is_training)
                            with tf.variable_scope('ud5'):
                                x2 = prelu(deconv_layer(tf.concat([sum,x2],3), [1, 1, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #x2 = batch_normalize(x2, is_training)
                            with tf.variable_scope('ud6'):
                                x3 = prelu(deconv_layer(tf.concat([sum,x3],3), [1, 1, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #x3 = batch_normalize(x3, is_training)
                    with tf.variable_scope('ud7'):
                        block_out = prelu(deconv_layer(tf.concat([x1, x2, x3],3), [3, 3, 64, 192], [self.batch_size, self.image_size, self.image_size, 64], 1))
                    #with tf.variable_scope('ud8'):
                        #block_out_att = deconv_layer(block_out, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1)
                    #block_out_att = tf.nn.sigmoid(block_out_att)
                    #block_out = block_out_att*block_out + block_out
                    #x = x1+x2+x3+x
                    x_gan+=block_out
            with tf.variable_scope('gan6'):
                x_gan = deconv_layer(
                    x_gan, [3, 3, 256, 64], [self.batch_size, self.image_size, self.image_size, 256], 1)#2
                x_gan = pixel_shuffle_layerg(x_gan, 2, 64) # n_split = 256 / 2 ** 2
                x_gan = prelu(x_gan)
            
            with tf.variable_scope('gan7'):
                x_gan = deconv_layer(
                    x_gan, [3, 3, 256, 64], [self.batch_size, self.image_size*2, self.image_size*2, 256], 1)#2
                x_gan = pixel_shuffle_layerg(x_gan, 2, 64) # n_split = 256 / 2 ** 2
                x_gan = prelu(x_gan)
                
            with tf.variable_scope('gan8'):
                x_gan = deconv_layer(
                    x_gan, [3, 3, 256, 64], [self.batch_size, self.image_size*4, self.image_size*4, 256], 1)#2
                x_gan = pixel_shuffle_layerg(x_gan, 2, 64) # n_split = 256 / 2 ** 2
                x_gan = prelu(x_gan)
                
            with tf.variable_scope('gan9'):
                x_gan = deconv_layer(
                    x_gan, [3, 3, 3, 64], [self.batch_size, self.image_size*8, self.image_size*8, 3], 1)
            x_gan_SR = x_gan + self.bic_ref
            x_gan_edge = self.Laplacian(x_gan_SR)
            
            # RNN框架
            with tf.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([self.image_size, self.image_size], [3, 3], 128)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            '''with tf.variable_scope('LSTM1'):
                cell_2 = BasicConvLSTMCell([self.image_size, self.image_size], [3, 3], 128)
                rnn_state_2 = cell_2.zero_state(batch_size=self.batch_size, dtype=tf.float32)'''
                
            with tf.variable_scope('rnn1'):
                x_rnn = deconv_layer(
                    x_com, [3, 3, 64, 3], [self.batch_size, self.image_size, self.image_size, 64], 1)
                x_rnn = prelu(x_rnn)

            lstm_input = lstm_in = x_rnn
            for n in range(6):
                with tf.variable_scope('lstm_1_{}'.format(n)):
                    x_rnn = deconv_layer(
                        lstm_in, [3, 3, 128, 64], [self.batch_size, self.image_size, self.image_size, 128], 1)
                    x_rnn = prelu(x_rnn)
                    y_1, rnn_state = cell(x_rnn, rnn_state)
                #with tf.variable_scope('lstm_2_{}'.format(n)):
                    #x_rnn = deconv_layer(
                        #y_1, [3, 3, 64, 128], [self.batch_size, self.image_size, self.image_size, 64], 1)
                    #x_rnn = prelu(x_rnn)
                #with tf.variable_scope('lstm_3_{}'.format(n)):
                    #x_rnn = deconv_layer(
                        #y_1, [3, 3, 128, 128], [self.batch_size, self.image_size, self.image_size, 128], 1)
                    #x_rnn = prelu(x_rnn)
                    #y_2, rnn_state = cell(x_rnn, rnn_state)
                with tf.variable_scope('lstm_4_{}'.format(n)):
                    x_rnn = deconv_layer(
                        y_1, [3, 3, 64, 128], [self.batch_size, self.image_size, self.image_size, 64], 1)
                    x_rnn = prelu(x_rnn)
                lstm_in += x_rnn 
            lstm_output = tf.concat([lstm_in, lstm_input],3)
            
            with tf.variable_scope('rnn4'):
                x_rnn = deconv_layer(
                    lstm_output, [3, 3, 64, 128], [self.batch_size, self.image_size, self.image_size, 64], 1)
                x_rnn = prelu(x_rnn)
                
            with tf.variable_scope('rnn5'):
                x_rnn = deconv_layer(
                    x_rnn, [3, 3, 256, 64], [self.batch_size, self.image_size, self.image_size, 256], 1)#2
                x_rnn = pixel_shuffle_layerg(x_rnn, 2, 64) # n_split = 256 / 2 ** 2
                x_rnn = prelu(x_rnn)
            
            with tf.variable_scope('rnn6'):
                x_rnn = deconv_layer(
                    x_rnn, [3, 3, 256, 64], [self.batch_size, self.image_size*2, self.image_size*2, 256], 1)#2
                x_rnn = pixel_shuffle_layerg(x_rnn, 2, 64) # n_split = 256 / 2 ** 2
                x_rnn = prelu(x_rnn)
                
            with tf.variable_scope('rnn7'):
                x_rnn = deconv_layer(
                    x_rnn, [3, 3, 256, 64], [self.batch_size, self.image_size*4, self.image_size*4, 256], 1)#2
                x_rnn = pixel_shuffle_layerg(x_rnn, 2, 64) # n_split = 256 / 2 ** 2
                x_rnn = prelu(x_rnn)
                
            with tf.variable_scope('rnn8'):
                x_rnn = deconv_layer(
                    x_rnn, [3, 3, 3, 64], [self.batch_size, self.image_size*8, self.image_size*8, 3], 1)
            x_rnn_SR = x_rnn + self.bic_ref
            #all_features = tf.concat([x_gan_SR, x_cnn_SR],3)
            # attention框架
			# cnn mask
            with tf.variable_scope('cnnmask1'):
                x_cnn_mask = conv_layer(x_cnn_SR, [3, 3, 3, 32], 1)#128
                x_cnn_mask = prelu(x_cnn_mask)
            with tf.variable_scope('cnnmask1_1'):
                x_cnn_mask = conv_layer(x_cnn_mask, [3, 3, 32, 64], 2)#64
                x_cnn_mask = prelu(x_cnn_mask)
            with tf.variable_scope('cnnmask1_2'):
                x_cnn_mask = conv_layer(x_cnn_mask, [3, 3, 64, 128], 2)#32
                x_cnn_mask = prelu(x_cnn_mask)
            with tf.variable_scope('cnnmask1_3'):
                x_cnn_mask = conv_layer(x_cnn_mask, [3, 3, 128, 64], 1)#128
                x_cnn_mask = prelu(x_cnn_mask)
            res_input_cnn = res_in_cnn = x_cnn_mask
            for j in range(3):
                with tf.variable_scope('cnnmask_1{}'.format(j)):
                    fuse = deconv_layer(
                        res_in_cnn, [3, 3, 64, 64], [self.batch_size, self.image_size*2, self.image_size*2, 64], 1)#2
                    fuse = prelu(fuse) 
                with tf.variable_scope('cnnmask_2{}'.format(j)):
                    fuse = deconv_layer(
                        fuse, [3, 3, 64, 64], [self.batch_size, self.image_size*2, self.image_size*2, 64], 1)#2
                    fuse = prelu(fuse)
                res_in_cnn +=fuse
            res_output_cnn = tf.concat([res_input_cnn,res_in_cnn],3)
            with tf.variable_scope('cnnmask1_4'):
                x_cnn_mask = deconv_layer(
                    res_output_cnn, [3, 3, 128, 128], [self.batch_size, self.image_size*4, self.image_size*4, 128], 2)#2
                #x_cnn_mask = pixel_shuffle_layerg(x_cnn_mask, 2, 32) # n_split = 256 / 2 ** 2
                x_cnn_mask = prelu(x_cnn_mask)
            with tf.variable_scope('cnnmask1_5'):
                x_cnn_mask = deconv_layer(
                    x_cnn_mask, [3, 3, 128, 128], [self.batch_size, self.image_size*8, self.image_size*8, 128], 2)#2
                #x_cnn_mask = pixel_shuffle_layerg(x_cnn_mask, 2, 32) # n_split = 256 / 2 ** 2
                x_cnn_mask = prelu(x_cnn_mask)
            with tf.variable_scope('cnnmask2'):
                x_cnn_mask = conv_layer(x_cnn_mask, [3, 3, 128, 3], 1)#128
                #x_cnn_mask = prelu(x_cnn_mask)
            x_cnn_mask = tf.nn.sigmoid(x_cnn_mask)
            #x_cnn_att = x_cnn_SR+x_cnnmask*x_cnn_SR
            x_cnn_att = x_cnn_mask*x_cnn_SR+x_cnn_SR
            #x_cnn_att_gan = (1-x_ganmask)*short1
			# gan mask
            with tf.variable_scope('ganmask1'):
                x_gan_mask = conv_layer(x_gan_SR, [3, 3, 3, 32], 1)#128
                x_gan_mask = prelu(x_gan_mask)
            with tf.variable_scope('ganmask1_1'):
                x_gan_mask = conv_layer(x_gan_mask, [3, 3, 32, 64], 2)#64
                x_gan_mask = prelu(x_gan_mask)
            with tf.variable_scope('ganmask1_2'):
                x_gan_mask = conv_layer(x_gan_mask, [3, 3, 64, 128], 2)#32
                x_gan_mask = prelu(x_gan_mask)
            with tf.variable_scope('ganmask1_3'):
                x_gan_mask = conv_layer(x_gan_mask, [3, 3, 128, 64], 1)#128
                x_gan_mask = prelu(x_gan_mask)
            res_input_gan = res_in_gan = x_gan_mask
            for j in range(3):
                with tf.variable_scope('ganmask_1{}'.format(j)):
                    fuse = deconv_layer(
                        res_in_gan, [3, 3, 64, 64], [self.batch_size, self.image_size*2, self.image_size*2, 64], 1)#2
                    fuse = prelu(fuse) 
                with tf.variable_scope('ganmask_2{}'.format(j)):
                    fuse = deconv_layer(
                        fuse, [3, 3, 64, 64], [self.batch_size, self.image_size*2, self.image_size*2, 64], 1)#2
                    fuse = prelu(fuse)
                res_in_gan +=fuse
            res_output_gan = tf.concat([res_input_gan,res_in_gan],3)
            with tf.variable_scope('ganmask1_4'):
                x_gan_mask = deconv_layer(
                    res_output_gan, [3, 3, 128, 128], [self.batch_size, self.image_size*4, self.image_size*4, 128], 2)#2
                #x_gan_mask = pixel_shuffle_layerg(x_gan_mask, 2, 32) # n_split = 256 / 2 ** 2
                x_gan_mask = prelu(x_gan_mask)
            with tf.variable_scope('ganmask1_5'):
                x_gan_mask = deconv_layer(
                    x_gan_mask, [3, 3, 128, 128], [self.batch_size, self.image_size*8, self.image_size*8, 128], 2)#2
                #x_gan_mask = pixel_shuffle_layerg(x_gan_mask, 2, 32) # n_split = 256 / 2 ** 2
                x_gan_mask = prelu(x_gan_mask)
            with tf.variable_scope('ganmask2'):
                x_gan_mask = conv_layer(x_gan_mask, [3, 3, 128, 3], 1)#128
                #x_gan_mask = prelu(x_gan_mask)
            x_gan_mask = tf.nn.sigmoid(x_gan_mask)
            #x_cnn_att = x_cnn_SR+x_cnnmask*x_cnn_SR
            x_gan_att = x_gan_mask*x_gan_SR+x_gan_SR
            # # RNN 
            # with tf.variable_scope('rnnmask1'):
                # x_rnnmask = conv_layer(x_rnn_SR, [3, 3, 3, 64], 1)#128
                # x_rnnmask = prelu(x_rnnmask)
            # with tf.variable_scope('rnnmask1_1'):
                # x_rnnmask = conv_layer(x_rnnmask, [3, 3, 64, 128], 2)#64
                # x_rnnmask = prelu(x_rnnmask)
            # with tf.variable_scope('rnnmask1_2'):
                # x_rnnmask = conv_layer(x_rnnmask, [3, 3, 128, 128], 2)#32
                # x_rnnmask = prelu(x_rnnmask)
            # with tf.variable_scope('rnnmask1_3'):
                # x_rnnmask = conv_layer(x_rnnmask, [3, 3, 128, 64], 1)#128
                # x_rnnmask = prelu(x_rnnmask)
            # res_input_rnn = res_in_rnn = x_rnnmask
            # for j in range(3):
                # with tf.variable_scope('rnnmask_1{}'.format(j)):
                    # fuse = deconv_layer(
                        # res_in_rnn, [3, 3, 64, 64], [self.batch_size, self.image_size*2, self.image_size*2, 64], 1)#2
                    # fuse = prelu(fuse) 
                # with tf.variable_scope('rnnmask_2{}'.format(j)):
                    # fuse = deconv_layer(
                        # fuse, [3, 3, 64, 64], [self.batch_size, self.image_size*2, self.image_size*2, 64], 1)#2
                    # fuse = prelu(fuse)
                # res_in_rnn +=fuse
            # res_output_gan = tf.concat([res_input_rnn,res_in_rnn],3)
            # with tf.variable_scope('rnnmask1_4'):
                # x_rnnmask = deconv_layer(
                    # res_output_gan, [3, 3, 128, 128], [self.batch_size, self.image_size*2, self.image_size*2, 128], 1)#2
                # x_rnnmask = pixel_shuffle_layerg(x_rnnmask, 2, 32) # n_split = 256 / 2 ** 2
                # x_rnnmask = prelu(x_rnnmask)
            # with tf.variable_scope('rnnmask1_5'):
                # x_rnnmask = deconv_layer(
                    # x_rnnmask, [3, 3, 128, 32], [self.batch_size, self.image_size*4, self.image_size*4, 128], 1)#2
                # x_rnnmask = pixel_shuffle_layerg(x_rnnmask, 2, 32) # n_split = 256 / 2 ** 2
                # x_rnnmask = prelu(x_rnnmask)
            # with tf.variable_scope('rnnmask2'):
                # x_rnnmask = conv_layer(x_rnnmask, [3, 3, 32, 3], 1)#128
                # x_rnnmask = prelu(x_rnnmask)
            # x_rnnmask = tf.nn.sigmoid(x_rnnmask)
            # x_rnn_att = x_rnnmask*x_rnn_SR+x_rnn_SR
            #att_SR = x_gan_att + x_cnn_att
			# fusion 融合x_cnn_SR, x_gan_SR, x_ganmask,x_cnnmask,x_rnnmask,
            att_feature = tf.concat([x_gan_att,x_cnn_att, x_gan_mask,x_cnn_mask],3)#x_gan_att + x_cnn_att
            with tf.variable_scope('fu1'):
                fuse = deconv_layer(
                    att_feature, [3, 3, 128, 12], [self.batch_size, self.image_size*8, self.image_size*8, 128], 1)#2
                fuse = prelu(fuse)
            #res_input = res_in = fuse
            # for j in range(3):
                # with tf.variable_scope('res1_{}'.format(j)):
                    # fuse = deconv_layer(
                        # res_in, [3, 3, 64, 128], [self.batch_size, self.image_size*8, self.image_size*8, 64], 1)#2
                    # fuse = prelu(fuse) 
                # with tf.variable_scope('res2_{}'.format(j)):
                    # fuse = deconv_layer(
                        # fuse, [3, 3, 128, 64], [self.batch_size, self.image_size*8, self.image_size*8, 128], 1)#2
                    # fuse = prelu(fuse)
                # res_in +=fuse
            # res_output = res_input + res_in
            with tf.variable_scope('fu5'):
                fuse = deconv_layer(
                    fuse, [3, 3, 3, 128], [self.batch_size, self.image_size*8, self.image_size*8, 3], 1)
            att_SR = fuse + x_rnn_SR#*(1-x_ganmask-x_cnnmask-x_rnnmask)
            x_att_edge = self.Laplacian(att_SR)
        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return x_att_edge, x_cnn_mask, x_gan_mask, x_cnn_SR, x_gan_SR, x_rnn_SR, att_SR
        
    def discriminator(self, x, is_training, reuse):
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [3, 3, 3, 64], 1)
                x = prelu(x)
                x = batch_normalize(x, is_training)
            #x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #x1_1 = max_pooling_layer(x, 2, 2)
            #x1_2 = avg_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 64], 1)
                x = prelu(x)
                x = batch_normalize(x, is_training)
            #x = tf.concat([x,x1_1],3)
            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 64, 128], 1)
                x = prelu(x)
                x = batch_normalize(x, is_training)
            #x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #x = avg_pooling_layer(x, 2, 2)
            #x2_1 = max_pooling_layer(x, 2, 2)
            #x2_2 = avg_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = prelu(x)
                x = batch_normalize(x, is_training)
            #x = tf.concat([x,x2_1],3)
            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 128, 256], 1)
                x = prelu(x)
                x = batch_normalize(x, is_training)
            #x3_1 = max_pooling_layer(x, 2, 2)
            #x3_2 = avg_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = prelu(x)
                x = batch_normalize(x, is_training)
            #x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            x = max_pooling_layer(x, 2, 2)
            #x = avg_pooling_layer(x, 2, 2)
            #x = tf.concat([x,x3_1],3)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 512], 1)
                x = prelu(x)
                x = batch_normalize(x, is_training)
            #x4_1 = max_pooling_layer(x, 2, 2)
            #x4_2 = avg_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = prelu(x)
                x = batch_normalize(x, is_training)
            #x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #x = tf.concat([x,x4_1],3)
            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv9'):
                x = conv_layer(x, [3, 3, 512, 1024], 1)
                x = prelu(x)
                x = batch_normalize(x, is_training)
            x = tf.reshape(x, (-1, 1, 1, 8*8*1024))
            x = flatten_layer(x)
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 1024)
                x = prelu(x)
            with tf.variable_scope('fc2'):
                x = full_connection_layer(x, 1)
                
        self.d_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return x
        
    '''def discriminator_att(self, x, is_training, reuse):
        with tf.variable_scope('discriminator_att', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [3, 3, 3, 64], 1)
                x = prelu(x)
                #x = batch_normalize(x, is_training)
            #x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #x1_1 = max_pooling_layer(x, 2, 2)
            #x1_2 = avg_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 64], 1)
                x = prelu(x)
                #x = batch_normalize(x, is_training)
            #x = tf.concat([x,x1_1],3)
            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 64, 128], 1)
                x = prelu(x)
                #x = batch_normalize(x, is_training)
            #x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #x = avg_pooling_layer(x, 2, 2)
            #x2_1 = max_pooling_layer(x, 2, 2)
            #x2_2 = avg_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = prelu(x)
                #x = batch_normalize(x, is_training)
            #x = tf.concat([x,x2_1],3)
            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 128, 256], 1)
                x = prelu(x)
            #x3_1 = max_pooling_layer(x, 2, 2)
            #x3_2 = avg_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = prelu(x)
                #x = batch_normalize(x, is_training)
            #x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            x = max_pooling_layer(x, 2, 2)
            #x = avg_pooling_layer(x, 2, 2)
            #x = tf.concat([x,x3_1],3)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 512], 1)
                x = prelu(x)
                #x = batch_normalize(x, is_training)
            #x4_1 = max_pooling_layer(x, 2, 2)
            #x4_2 = avg_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = prelu(x)
                #x = batch_normalize(x, is_training)
            #x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #x = tf.concat([x,x4_1],3)
            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv9'):
                x = conv_layer(x, [3, 3, 512, 1024], 1)
                x = prelu(x)
                #x = batch_normalize(x, is_training)
            x = tf.reshape(x, (-1, 1, 1, 8*8*1024))
            x = flatten_layer(x)
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 1024)
                x = prelu(x)
            with tf.variable_scope('fc2'):
                x = full_connection_layer(x, 1)
                
        self.d_variables_att = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_att')
        return x'''
        
    def downscale(self, x):
        K = 8
        arr = np.zeros([K, K, 3, 3])
        arr[:, :, 0, 0] = 1.0 / K ** 2
        arr[:, :, 1, 1] = 1.0 / K ** 2
        arr[:, :, 2, 2] = 1.0 / K ** 2
        weight = tf.constant(arr, dtype=tf.float32)
        downscaled = tf.nn.conv2d(
            x, weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled
    
    def sobel(self, x):
        weight=tf.constant([[[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
                                  [-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
                                  [-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0]],
                            [[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
                                  [-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
                                  [-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0]],
                            [[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
                                  [-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
                                  [-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0]]],
                                 shape=[3, 3, 3, 3])
        frame=tf.nn.conv2d(x,weight,[1,1,1,1],padding='SAME')
        frame = tf.cast(((frame - tf.reduce_min(frame)) / (tf.reduce_max(frame) - tf.reduce_min(frame)))*2.0-1, tf.float32)
        #frame = tf.cast(((frame - tf.reduce_min(frame)) / (tf.reduce_max(frame) - tf.reduce_min(frame))) * 255, tf.uint8)
        return frame
        
    def Laplacian(self, x):
        weight=tf.constant([
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]]
        ])
        frame=tf.nn.conv2d(x,weight,[1,1,1,1],padding='SAME')
        #frame = tf.cast(((frame - tf.reduce_min(frame)) / (tf.reduce_max(frame) - tf.reduce_min(frame))) * 255, tf.uint8)
        return frame
    
    def sobelg(self, x):
        weightx=tf.constant([[[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
                                  [-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
                                  [-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0]],
                            [[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
                                  [-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
                                  [-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0]],
                            [[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
                                  [-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
                                  [-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0]]],
                                 shape=[3, 3, 3, 3])
        framex=tf.nn.conv2d(x,weightx,[1,1,1,1],padding='SAME')
        weighty=tf.constant([[[-1.0, -2.0, -1.0], [-1.0, -2.0, -1.0], [-1.0, -2.0, -1.0],
                                  [0.0, 0.0, 0.0], [0, 0, 0], [0.0, 0.0, 0.0],
                                  [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0]],
                             [[-1.0, -2.0, -1.0], [-1.0, -2.0, -1.0], [-1.0, -2.0, -1.0],
                                  [0.0, 0.0, 0.0], [0, 0, 0], [0.0, 0.0, 0.0],
                                  [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0]],
                             [[-1.0, -2.0, -1.0], [-1.0, -2.0, -1.0], [-1.0, -2.0, -1.0],
                                  [0.0, 0.0, 0.0], [0, 0, 0], [0.0, 0.0, 0.0],
                                  [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0]]],
                                 shape=[3, 3, 3, 3])
        framey=tf.nn.conv2d(x,weighty,[1,1,1,1],padding='SAME')    
        frame = tf.sqrt(framex*framex + framey*framey)        
        #frame=lrelu(tf.nn.conv2d(x,weight,[1,1,1,1],padding='SAME'))
        #frame = tf.cast(((frame - tf.reduce_min(frame)) / (tf.reduce_max(frame) - tf.reduce_min(frame))) * 255, tf.uint8)
        return frame
        
    def inference_losses(self, edge_hr, att_edge, x, cnn_SR, gan_SR, rnn_SR, att_SR, true_output, fake_output):
        def inference_content_loss1(x, imitation):#true_output, fake_output,, true_output_att, fake_output_att
            _, x_phi = self.vgg.build_model(
                x, tf.constant(False), False) # First cnn_edge, gan_edge, 
            _, imitation_phi = self.vgg.build_model(
                imitation, tf.constant(False), True) # Second

            content_loss = None
            for i in range(len(x_phi)):
                #loss = tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
                loss = tf.reduce_mean(tf.sqrt((x_phi[i] - imitation_phi[i]) ** 2+(1e-3)**2))
                if content_loss is None:
                    content_loss = loss
                else:
                    content_loss = content_loss + loss
            return tf.reduce_mean(content_loss)
        
        def inference_content_loss2(x, imitation):#frame_hr, frame_sr,, real_frame, fake_frame,real_base, fake_base,
            _, x_phi = self.vgg.build_model(
                x, tf.constant(False), True) # First
            _, imitation_phi = self.vgg.build_model(
                imitation, tf.constant(False), True) # Second

            content_loss = None
            for i in range(len(x_phi)):
                #loss = tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
                loss = tf.reduce_mean(tf.sqrt((x_phi[i] - imitation_phi[i]) ** 2+(1e-3)**2))
                if content_loss is None:
                    content_loss = loss
                else:
                    content_loss = content_loss + loss
            return tf.reduce_mean(content_loss)
        def inference_content_loss3(x, imitation):#frame_hr, frame_sr,, real_frame, fake_frame,real_base, fake_base,
            _, x_phi = self.vgg.build_model(
                x, tf.constant(False), True) # First
            _, imitation_phi = self.vgg.build_model(
                imitation, tf.constant(False), True) # Second

            content_loss = None
            for i in range(len(x_phi)):
                #loss = tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
                loss = tf.reduce_mean(tf.sqrt((x_phi[i] - imitation_phi[i]) ** 2+(1e-3)**2))
                if content_loss is None:
                    content_loss = loss
                else:
                    content_loss = content_loss + loss
            return tf.reduce_mean(content_loss)
        def inference_content_loss4(x, imitation):#frame_hr, frame_sr,, real_frame, fake_frame,real_base, fake_base,
            _, x_phi = self.vgg.build_model(
                x, tf.constant(False), True) # First
            _, imitation_phi = self.vgg.build_model(
                imitation, tf.constant(False), True) # Second

            content_loss = None
            for i in range(len(x_phi)):
                #loss = tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
                loss = tf.reduce_mean(tf.sqrt((x_phi[i] - imitation_phi[i]) ** 2+(1e-3)**2))
                if content_loss is None:
                    content_loss = loss
                else:
                    content_loss = content_loss + loss
            return tf.reduce_mean(content_loss)
        # def inference_content_loss5(x, imitation):#frame_hr, frame_sr,, real_frame, fake_frame,real_base, fake_base,
            # _, x_phi = self.vgg.build_model(
                # x, tf.constant(False), True) # First
            # _, imitation_phi = self.vgg.build_model(
                # imitation, tf.constant(False), True) # Second

            # content_loss = None
            # for i in range(len(x_phi)):
                # #loss = tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
                # loss = tf.reduce_mean(tf.sqrt((x_phi[i] - imitation_phi[i]) ** 2+(1e-3)**2))
                # if content_loss is None:
                    # content_loss = loss
                # else:
                    # content_loss = content_loss + loss
            # return tf.reduce_mean(content_loss)
        def inference_content_loss_sr(frame_hr, frame_sr):
            content_base_loss = tf.reduce_mean(tf.sqrt((frame_hr - frame_sr) ** 2+(1e-3)**2))
            return tf.reduce_mean(content_base_loss)
            
        def wasserstein(y_true, y_pred):
            # return K.mean(y_true * y_pred) / K.mean(y_true)
            return K.mean(y_true * y_pred)
            
        def inference_adversarial_loss(true_output, fake_output):
            alpha = 1e-5#1e-5#1e-2,1e-5
            # g_loss = tf.reduce_mean(tf.nn.l2_loss(fake_output - tf.ones_like(fake_output)))
            g_loss = tf.reduce_mean(tf.sqrt((fake_output - tf.ones_like(fake_output)) ** 2+(1e-3)**2))
            # d_loss_real = tf.reduce_mean(tf.nn.l2_loss(real_output - tf.ones_like(real_output)))
            d_loss_real = tf.reduce_mean(tf.sqrt((true_output - tf.ones_like(true_output)) ** 2+(1e-3)**2))
            # d_loss_fake = tf.reduce_mean(tf.nn.l2_loss(fake_output + tf.zeros_like(fake_output)))
            d_loss_fake = tf.reduce_mean(tf.sqrt((fake_output + tf.ones_like(fake_output)) ** 2+(1e-3)**2))
            d_loss = d_loss_real + d_loss_fake
            return (g_loss * alpha, d_loss * alpha)

        def inference_adversarial_loss_with_sigmoid(real_frame, fake_frame):
            alpha = 1e-3#1e-3
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_frame),
                logits=fake_frame)
            d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_frame),
                logits=real_frame)
            d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_frame),
                logits=fake_frame)
            d_loss = d_loss_real + d_loss_fake
            return (g_loss * alpha, d_loss * alpha)
        
        #seed_x = np.random.randint(0,16)
        #seed_y = np.random.randint(0,16)
        #x_1 = tf.slice(x,[0, cor_x, cor_y, 0],[self.batch_size, 96, 96, 3])
        #gan_SR_1 = tf.slice(gan_SR,[0, cor_x, cor_y, 0],[self.batch_size, 96, 96, 3])
        x_2 = tf.slice(x,[0, 16, 16, 0],[self.batch_size, 96, 96, 3])
        gan_SR_2 = tf.slice(gan_SR,[0, 16, 16, 0],[self.batch_size, 96, 96, 3])
        #x_3 = tf.slice(x,[0, cor_x, cor_y, 0],[self.batch_size, 96, 96, 3])
        #att_SR_1 = tf.slice(att_SR,[0, 16, 16, 0],[self.batch_size, 96, 96, 3])
        #x_4 = tf.slice(x,[0, 16, 16, 0],[self.batch_size, 96, 96, 3])
        #att_SR_2 = tf.slice(att_SR,[0, 16, 16, 0],[self.batch_size, 96, 96, 3])
        
        content_loss1 = inference_content_loss1(x_2, gan_SR_2)
        #content_loss2 = inference_content_loss2(x_2, gan_SR_2)
        #content_loss3 = inference_content_loss3(x_4, att_SR_1)
        #content_loss4 = inference_content_loss4(x_4, att_SR_2)
        
        a = 1#100
        b = 10
        c = 3#弱化CNN
        d = 1#0.01# 
        content_loss = content_loss1#+ content_loss3# d*a*content_loss2 + d*a*b*content_loss4 + 
        #content_sr_loss2 = inference_content_loss_sr(edge_hr, att_edge)#c*a*d*content_sr_loss2 + 
        content_sr_loss3 = inference_content_loss_sr(x, rnn_SR)
        content_sr_loss4 = inference_content_loss_sr(x, cnn_SR)
        content_sr_loss5 = inference_content_loss_sr(x, att_SR)#  + 0.01*a*content_sr_loss5+ 
        #content_sr_loss6 = inference_content_loss_sr(x, gan_SR)
        #0.01
        content_sr_loss = 1*content_sr_loss3 + 1*content_sr_loss4 + 0.05*1*content_sr_loss5#+ 0.5*1*content_sr_loss2# + content_sr_loss6
        generator_loss, discriminator_loss = (inference_adversarial_loss(true_output, fake_output))
        #generator_loss_att, discriminator_loss_att = (inference_adversarial_loss(true_output_att, fake_output_att))
        g_loss = 1*content_loss  + 1*content_sr_loss + 1*generator_loss# + d*generator_loss_att
        d_loss = 1*discriminator_loss# + d*discriminator_loss_att 
        #d_frame_loss = discriminator_frame_loss
        return (g_loss, d_loss)

