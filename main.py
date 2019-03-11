import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


CONTENT_WEIGHT = 5.0
STYLE_WEIGHT = [10, 10, 10, 10, 10]
TV_WEIGHT = 0.5


def load_image(path):
    image = Image.open(path)
    image = np.asarray(image, dtype="float32")
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    return image


def save_image(output, name):
    image = np.reshape(output, (output.shape[1], output.shape[2], output.shape[3]))
    image = np.clip(image, 0, 255).astype('uint8')
    imsave(name, image)


def gram_matrix(x):
    features = tf.keras.backend.batch_flatten(tf.transpose(x, perm=[2,0,1]))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def content_loss(content, combination):
    return tf.reduce_sum(tf.square(content - combination))


def style_loss(style, combination):
    h,w,d = style.get_shape()
    M = h.value*w.value
    N = d.value
    S = gram_matrix(style)
    C = gram_matrix(combination)
    return tf.reduce_sum(tf.square(S-C)) / (4. * (N **2) * (M ** 2))

def tv_loss(output):
    horizontal_normal = tf.slice(output, [0, 0, 0, 0], [output.shape[0], output.shape[1], output.shape[2]-1,output.shape[3]])
    horizontal_one_right = tf.slice(output, [0, 0, 1, 0], [output.shape[0], output.shape[1], output.shape[2]-1,output.shape[3]])
    vertical_normal = tf.slice(output, [0, 0, 0, 0], [output.shape[0], output.shape[1]-1, output.shape[2],output.shape[3]])
    vertical_one_right = tf.slice(output, [0, 1, 0, 0], [output.shape[0], output.shape[1]-1, output.shape[2],output.shape[3]])
    tv_loss = tf.nn.l2_loss(horizontal_normal-horizontal_one_right)+tf.nn.l2_loss(vertical_normal-vertical_one_right)
    return tv_loss


def dilate_mask(mask):
    mask = np.reshape(mask, (mask.shape[1], mask.shape[2], mask.shape[3]))
    loose_mask = cv2.GaussianBlur(mask, (35,35), 35/3)
    loose_mask[loose_mask>=0.1] = 1
    loose_mask = loose_mask.reshape(1, mask.shape[0], mask.shape[1], mask.shape[2])
    return loose_mask

def first_pass(content_image, style_image, mask_image, dilated_mask):

    
    config = {"layer_content": "block2_conv2",
		"layers_style": ["block1_conv2","block2_conv2","block3_conv3",
        "block4_conv3","block5_conv3"]}

    sess = K.get_session()
    combination_im = tf.Variable(tf.random_uniform((1, content_image.shape[1],  content_image.shape[2],  content_image.shape[3])))
    input_tensor = tf.concat([content_image, style_image, combination_im, mask_image], 0)

    mask_smth = np.reshape(mask_image, (mask_image.shape[1], mask_image.shape[2], mask_image.shape[3]))
    mask_smth = cv2.GaussianBlur(mask_smth, (3,3) , 1)
    mask_smth = np.reshape(mask_smth, (1, mask_smth.shape[0], mask_smth.shape[1], mask_smth.shape[2]))
    
    model = VGG16(input_tensor=input_tensor, include_top=False, weights="imagenet")

    layers = dict([(layer.name, layer.output) for layer in model.layers])

    layer_content = layers[config["layer_content"]]
    layer_style = [layers[i] for i in config["layers_style"]]

    loss = tf.Variable(0.)
    init_new_vars_op = tf.initializers.variables([loss])
    sess.run(init_new_vars_op)

    loss = tf.add(loss, CONTENT_WEIGHT * content_loss(layer_content[3,:,:,:] * layer_content[0,:,:,:], layer_content[3,:,:,:] * layer_content[2,:,:,:]))

    for i in range(len(layer_style)):
        loss = tf.add(loss, STYLE_WEIGHT[i] * style_loss(layer_style[i][3,:,:,:] * layer_style[i][1,:,:,:], layer_style[i][3,:,:,:] * layer_style[i][2,:,:,:]))

    train_step = tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=[combination_im], options={'maxfun':20})

    print("Pass 1")
    for i in range(100):
        curr_loss = sess.run(loss)
        if (i+1) % 10 == 0:
            print("Iteration {0}, Loss: {1}".format(i, curr_loss))
            val_output = sess.run(combination_im)
            val_output = mask_smth/255 * val_output + (1-mask_smth/255) * style_image
            save_image(val_output, "val_output_"+str(i)+".jpg")
        train_step.minimize(session=sess)

    output = sess.run(combination_im)
    output = mask_smth/255 * output + (1-mask_smth/255) * style_image
    save_image(output, "pass1_output.jpg")
    return output

def second_pass(content_image, style_image, mask_image, dilated_mask, output_from_first_pass):
    
    config = {"layer_content": "block2_conv2",
		"layers_style": ["block1_conv2","block2_conv2","block3_conv3",
        "block4_conv3","block5_conv3"]}

    sess = K.get_session()
    combination_im = tf.Variable(output_from_first_pass)
    input_tensor = tf.concat([content_image, style_image, combination_im, mask_image], 0)

    mask_smth = np.reshape(mask_image, (mask_image.shape[1], mask_image.shape[2], mask_image.shape[3]))
    mask_smth = cv2.GaussianBlur(mask_smth, (3,3) , 1)
    mask_smth = np.reshape(mask_smth, (1, mask_smth.shape[0], mask_smth.shape[1], mask_smth.shape[2]))
    
    model = VGG16(input_tensor=input_tensor, include_top=False, weights="imagenet")

    layers = dict([(layer.name, layer.output) for layer in model.layers])

    layer_content = layers[config["layer_content"]]
    layer_style = [layers[i] for i in config["layers_style"]]

    loss = tf.Variable(0.)
    init_new_vars_op = tf.initializers.variables([loss])
    sess.run(init_new_vars_op)

    loss = tf.add(loss, CONTENT_WEIGHT/10 * content_loss(layer_content[3,:,:,:] * layer_content[0,:,:,:], layer_content[3,:,:,:] * layer_content[2,:,:,:]))

    for i in range(len(layer_style)):
        loss = tf.add(loss, STYLE_WEIGHT[i]/5 * style_loss(layer_style[i][3,:,:,:] * layer_style[i][1,:,:,:], layer_style[i][3,:,:,:] * layer_style[i][2,:,:,:]))

    loss = tf.add(loss, TV_WEIGHT * tv_loss(combination_im))

    train_step = tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=[combination_im], options={'maxfun':20})

    print("Pass 2")
    for i in range(20):
        curr_loss = sess.run(loss)
        if (i+1) % 10 == 0:
            print("Iteration {0}, Loss: {1}".format(i, curr_loss))
            val_output = sess.run(combination_im)
            val_output = mask_smth/255 * val_output + (1-mask_smth/255) * style_image
            save_image(val_output, "val_output_"+str(i+100)+".jpg")
        train_step.minimize(session=sess)

    output = sess.run(combination_im)
    output = mask_smth/255 * output + (1-mask_smth/255) * style_image
    save_image(output, "pass2_output.jpg")


content = load_image("trump_lastsupper/input.jpg")
style = load_image("trump_lastsupper/original.jpg")
mask = load_image("trump_lastsupper/mask.jpg")
dilated_mask = dilate_mask(mask)

output_pass1 = first_pass(content, style, mask, dilated_mask)
second_pass(content, style, mask, dilated_mask, output_pass1)
