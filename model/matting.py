from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import session_run_hook
import logging
import numpy as np


def unpool(pool, ind, ksize=[1, 2, 2, 1], name=None):
    logging.info('{} pool:{} ind:{}'.format(name,pool,ind))
    with tf.variable_scope(name) as scope:
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

        flat_input_size = tf.cumprod(input_shape)[-1]
        flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

        pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                 shape=tf.stack([input_shape[0], 1, 1, 1]))
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, tf.stack([flat_input_size, 1]))
        ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
        ind_ = ind_ - b * tf.cast(flat_output_shape[1], tf.int64)
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, tf.stack(output_shape))

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)
    return ret


def _matting_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    ground_truth = labels
    rgb_input = features['input']
    trimap_input = features['trimap']
    tf.summary.image('rgb_input', rgb_input, max_outputs=5)
    tf.summary.image('trimap_input', trimap_input, max_outputs=5)
    ground_truth.set_shape([params['batch_size'], params['image_height'], params['image_width'], 1])
    rgb_input.set_shape([params['batch_size'], params['image_height'], params['image_width'], 3])
    trimap_input.set_shape([params['batch_size'], params['image_height'], params['image_width'], 1])
    b_input = tf.concat([rgb_input, trimap_input], 3)
    print(b_input.shape)

    encoder_training = params['enc_dec']
    refinement_training = params['refinement']
    logging.info("Train encoder_decoder {}".format(encoder_training))
    logging.info("Train refinement {}".format(refinement_training))
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    def _layer_sum(name,l):
        #tf.summary.scalar(name+"_0",tf.reduce_sum(l[0]))
        ##tf.summary.scalar(name+"_1",tf.reduce_sum(l[1]))
        #tf.summary.scalar(name+"_2",tf.reduce_sum(l[2]))
        return

    _layer_sum("rgb_input",rgb_input)
    _layer_sum("rgb_input",trimap_input)
    # conv1_1
    en_parameters = []
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 4, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(b_input, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             name='biases', trainable=encoder_training)
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]
    _layer_sum("conv1_1",conv1_1)
    # conv1_2
    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    _layer_sum("conv1_2",conv1_2)
    # pool1
    pool1, arg1 = tf.nn.max_pool_with_argmax(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                             name='pool1')

    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope)

        en_parameters += [kernel, biases]

    # conv2_2
    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # pool2
    pool2, arg2 = tf.nn.max_pool_with_argmax(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                             name='pool2')

    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv3_3
    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # pool3
    pool3, arg3 = tf.nn.max_pool_with_argmax(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                             name='pool3')

    # conv4_1
    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv4_2
    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]


    # conv4_3
    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # pool4
    pool4, arg4 = tf.nn.max_pool_with_argmax(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                             name='pool4')

    # conv5_1
    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv5_2
    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv5_3
    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # pool5
    pool5, arg5 = tf.nn.max_pool_with_argmax(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                             name='pool5')
    # conv6_1
    with tf.name_scope('conv6_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([7, 7, 512, 4096], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv6_1 = tf.nn.relu(out, name='conv6_1')
        en_parameters += [kernel, biases]

    _layer_sum("conv6_1",conv6_1)
    # deconv6
    with tf.variable_scope('deconv6') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(conv6_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv6 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='deconv6')

    _layer_sum("deconv6",deconv6)
    # deconv5_1/unpooling
    deconv5_1 = unpool(deconv6, arg5,name='unpool5')
    _layer_sum("deconv5_1",deconv5_1)
    # deconv5_2
    with tf.variable_scope('deconv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(deconv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv5_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='deconv5_2')

    # deconv4_1/unpooling
    deconv4_1 = unpool(deconv5_2, arg4,name='unpool4')

    # deconv4_2
    with tf.variable_scope('deconv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(deconv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv4_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='deconv4_2')

    # deconv3_1/unpooling
    deconv3_1 = unpool(deconv4_2, arg3,name='unpool3')

    # deconv3_2
    with tf.variable_scope('deconv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 256, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(deconv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv3_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='deconv3_2')

    # deconv2_1/unpooling
    deconv2_1 = unpool(deconv3_2, arg2,name='unpool2')

    # deconv2_2
    with tf.variable_scope('deconv2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 128, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(deconv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv2_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='deconv2_2')

    # deconv1_1/unpooling
    deconv1_1 = unpool(deconv2_2, arg1,name='unpool1')

    # deconv1_2
    with tf.variable_scope('deconv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(deconv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv1_2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='deconv1_2')
    # pred_alpha_matte
    with tf.variable_scope('pred_alpha') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=encoder_training)
        conv = tf.nn.conv2d(deconv1_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=encoder_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        enc_dec_pred = tf.nn.sigmoid(out)

    _layer_sum("enc_dec_pred",enc_dec_pred)
    tf.summary.image('enc_dec_pred', enc_dec_pred, max_outputs=5)
    # refinement

    x = tf.concat([rgb_input, enc_dec_pred], 3)
    with tf.variable_scope('refinement_conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 4, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=refinement_training)
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=refinement_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        refinement1 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='refinement_conv1_2')
    with tf.variable_scope('refinement_conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=refinement_training)
        conv = tf.nn.conv2d(refinement1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=refinement_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        refinement2 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='refinement_conv2_2')
    with tf.variable_scope('refinement_conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=refinement_training)
        conv = tf.nn.conv2d(refinement2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=refinement_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        refinement3 = tf.nn.relu(tf.layers.batch_normalization(out, training=training), name='refinement_conv3_2')
    with tf.variable_scope('refinement_conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=refinement_training)
        conv = tf.nn.conv2d(refinement3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=refinement_training, name='biases')
        out = tf.nn.bias_add(conv, biases)
        refinement_pred = tf.nn.sigmoid(out)

    wl = tf.where(tf.equal(trimap_input, 128),
                  tf.fill([params['batch_size'], params['image_height'], params['image_width'], 1], 1.),
                  tf.fill([params['batch_size'], params['image_height'], params['image_width'], 1], 0.))
    unknown_region_size = tf.reduce_sum(wl)

    tf.summary.image('refinement_pred', refinement_pred, max_outputs=5)

    refinement_predictions = tf.where(tf.equal(trimap_input, 128), refinement_pred, trimap_input / 255.0)
    enc_dec_predictions = tf.where(tf.equal(trimap_input, 128), enc_dec_pred, trimap_input / 255.0)
    tf.summary.image('final_refinement', refinement_predictions, max_outputs=5)
    tf.summary.image('final_enc_dec', enc_dec_predictions, max_outputs=5)

    evaluation_hooks = None
    metrics = {}
    hooks = []
    if mode != tf.estimator.ModeKeys.PREDICT:
        original_background = features['original_background']
        foreground_input = features['foreground']
        raw_background = features['row_background']
        tf.summary.image('raw_background', raw_background, max_outputs=5)
        trimap_input.set_shape([params['batch_size'], params['image_height'], params['image_width'], 1])
        original_background.set_shape([params['batch_size'], params['image_height'], params['image_width'], 3])
        foreground_input.set_shape([params['batch_size'], params['image_height'], params['image_width'], 3])
        raw_background.set_shape([params['batch_size'], params['image_height'], params['image_width'], 3])
        bgs = tf.unstack(original_background)
        fgs = tf.unstack(foreground_input)

        def _loss(name, pred, final_pred):
            alpha_diff = tf.sqrt(tf.square(pred - ground_truth) + 1e-12)
            p_rgb = []
            l_matte = tf.unstack(final_pred)
            for i in range(params['batch_size']):
                p_rgb.append(l_matte[i] * fgs[i] + (tf.constant(1.) - l_matte[i]) * bgs[i])
            pred_rgb = tf.stack(p_rgb)
            tf.summary.image(name + '_pred_rgb', pred_rgb, max_outputs=5)
            c_diff = tf.sqrt(tf.square(pred_rgb - raw_background) + 1e-12) / 255.0
            alpha_loss = tf.reduce_sum(alpha_diff * wl) / unknown_region_size
            alpha_loss_eval = tf.metrics.mean(alpha_loss)
            metrics[name + '_alpha_loss'] = alpha_loss_eval
            comp_loss = tf.reduce_sum(c_diff * wl) / unknown_region_size
            comp_loss_eval = tf.metrics.mean(comp_loss)
            metrics[name + '_comp_loss'] = comp_loss_eval

            tf.summary.scalar(name + '_alpha_loss', alpha_loss)
            tf.summary.scalar(name + '_comp_loss', comp_loss)
            total_loss = (alpha_loss + comp_loss) * 0.5
            tf.summary.scalar(name + '_total_loss', total_loss)
            total_loss_eval = tf.metrics.mean(total_loss)
            metrics[name + '_total_loss'] = total_loss_eval
            return total_loss

        refinement_loss = _loss('refinement', refinement_pred, refinement_predictions)
        enc_dec_loss = _loss('enc_dec', enc_dec_pred, enc_dec_predictions)
        if refinement_training:
            total_loss = refinement_loss
        else:
            total_loss = enc_dec_loss
        opt = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(total_loss, global_step=tf.train.get_or_create_global_step())
        if params['vgg16'] is not None:
            hooks.append(InitVariablesHook(params['vgg16'],en_parameters))
    else:
        train_op = None
        total_loss = None
    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=metrics,
        predictions=refinement_predictions,
        loss=total_loss,
        training_hooks=hooks,
        evaluation_hooks=evaluation_hooks,
        # export_outputs=export_outputs,
        train_op=train_op)


class DeepMatting(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _matting_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config,
                model_dir=model_dir,
            )

        super(DeepMatting, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )



class InitVariablesHook(session_run_hook.SessionRunHook):
    def __init__(self, model_path,en_parameters):
        self._model_path = model_path
        self._en_parameters = en_parameters
        self._ops = []

    def begin(self):
        logging.info('Begin VGG16 Init')
        weights = np.load(self._model_path)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i == 28:
                break
            if k == 'conv1_1_W':
                self._ops.append(self._en_parameters[i].assign(np.concatenate([weights[k],np.zeros([3,3,1,64])],axis = 2)))
            else:
                if k=='fc6_W':
                    tmp = np.reshape(weights[k],(7,7,512,4096))
                    self._ops.append(self._en_parameters[i].assign(tmp))
                else:
                    self._ops.append(self._en_parameters[i].assign(weights[k]))

    def after_create_session(self, session, coord):
        logging.info('Do VGG16 Init')
        if len(self._ops)>0:
            session.run(self._ops)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return None

    def after_run(self, run_context, run_values):
        None