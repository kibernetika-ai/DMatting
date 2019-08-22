from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging
import numpy as np
import glob
import PIL.Image

i = 0

def original(features, data):
    global i

    def ReflectionPad2d(p):
        k = i

        def _f(x):
            r = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode="REFLECT")
            print('ReflectionPad2d {}: {}'.format(k, r.shape))
            return r

        return _f

    def Pad(p):
        k = i

        def _f(x):
            r = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
            print('Pad {}: {}'.format(k, r.shape))
            return r

        return _f

    def Conv2d(strides=[1, 1, 1, 1]):
        global i
        if i == 0:
            kernel = 'Const'
        else:
            kernel = 'Const_{}'.format(i)
        bias = 'Const_{}'.format(i + 1)
        k = i

        def _f(x, pad=0):
            if pad > 0:
                c0 = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
            else:
                c0 = x
            c1 = tf.nn.conv2d(c0, data[kernel], strides, padding="VALID", use_cudnn_on_gpu=True, data_format='NHWC',
                              dilations=[1, 1, 1, 1])
            r = tf.add(c1, data[bias])
            print('Conv2d {}: kernel {}'.format(k, data[kernel].shape))
            print('Conv2d {}: bias {}'.format(k, data[bias].shape))
            print('Conv2d {}: {}'.format(k, r.shape))
            return r

        i += 2
        return _f

    def InstanceNormalization():
        global i
        name1 = 'Const_{}'.format(i)
        name2 = 'Const_{}'.format(i + 1)
        k = i

        def _f(x):
            mean = tf.reduce_mean(x, [1, 2], keepdims=True)
            var = x - mean
            var = var * var
            var = tf.reduce_mean(var, [1, 2], keepdims=True)
            c3 = (x - mean) / tf.sqrt(var + 1e-9)
            c4 = c3 * data[name1] + data[name2]
            print('InstanceNormalization {}: {}'.format(k, c4.shape))
            return c4

        i += 2
        return _f

    from tensorflow.python.keras.utils import conv_utils
    def ConvTranspose2d(sp):
        global i
        kernel = 'Const_{}'.format(i)
        bias = 'Const_{}'.format(i + 1)
        k = i

        def _f(x):
            # c1 = conv2d_backprop_input(x,data[kernel],out_backprop,[1, 2, 2, 1],"VALID")
            inputs_shape = tf.shape(x)
            batch_size = inputs_shape[0]
            height, width = inputs_shape[1], inputs_shape[2]
            print('ConvTranspose2d {}: input {}'.format(k, x.shape))
            print('ConvTranspose2d {}: kernel {}'.format(k, data[kernel].shape))
            print('ConvTranspose2d {}: bias {}'.format(k, data[bias].shape))
            inputs_shape = tf.stack(
                [batch_size, tf.cast(height * 2, dtype=tf.int32), tf.cast(width * 2, dtype=tf.int32), sp])
            c1 = tf.nn.conv2d_backprop_input(inputs_shape, data[kernel], x, [1, 2, 2, 1], 'SAME', use_cudnn_on_gpu=True)
            r = tf.add(c1, data[bias])
            print('ConvTranspose2d3 {}: {}'.format(k, r.shape))
            return r

        i += 2
        return _f

    i = 0
    refpad01_1 = ReflectionPad2d(3)
    conv01_1 = Conv2d()
    in01_1 = InstanceNormalization()

    # relu
    conv02_1 = Conv2d(strides=[1, 2, 2, 1])
    conv02_2 = Conv2d()
    in02_1 = InstanceNormalization()

    conv03_1 = Conv2d(strides=[1, 2, 2, 1])
    conv03_2 = Conv2d()
    in03_1 = InstanceNormalization()

    # relu
    ## res block 1
    refpad04_1 = ReflectionPad2d(1)
    conv04_1 = Conv2d()
    in04_1 = InstanceNormalization()

    # relu
    refpad04_2 = ReflectionPad2d(1)
    conv04_2 = Conv2d()
    in04_2 = InstanceNormalization()

    # + input
    ## res block 2
    refpad05_1 = ReflectionPad2d(1)
    conv05_1 = Conv2d()
    in05_1 = InstanceNormalization()
    # relu
    refpad05_2 = ReflectionPad2d(1)
    conv05_2 = Conv2d()
    in05_2 = InstanceNormalization()
    # + input

    ## res block 3
    refpad06_1 = ReflectionPad2d(1)
    conv06_1 = Conv2d()
    in06_1 = InstanceNormalization()
    # relu
    refpad06_2 = ReflectionPad2d(1)
    conv06_2 = Conv2d()
    in06_2 = InstanceNormalization()
    # + input

    ## res block 4
    refpad07_1 = ReflectionPad2d(1)
    conv07_1 = Conv2d()
    in07_1 = InstanceNormalization()
    # relu
    refpad07_2 = ReflectionPad2d(1)
    conv07_2 = Conv2d()
    in07_2 = InstanceNormalization()
    # + input

    ## res block 5
    refpad08_1 = ReflectionPad2d(1)
    conv08_1 = Conv2d()
    in08_1 = InstanceNormalization()
    # relu
    refpad08_2 = ReflectionPad2d(1)
    conv08_2 = Conv2d()
    in08_2 = InstanceNormalization()
    # + input

    ## res block 6
    refpad09_1 = ReflectionPad2d(1)
    conv09_1 = Conv2d()
    in09_1 = InstanceNormalization()
    # relu
    refpad09_2 = ReflectionPad2d(1)
    conv09_2 = Conv2d()
    in09_2 = InstanceNormalization()
    # + input

    ## res block 7
    refpad10_1 = ReflectionPad2d(1)
    conv10_1 = Conv2d()
    in10_1 = InstanceNormalization()
    # relu
    refpad10_2 = ReflectionPad2d(1)
    conv10_2 = Conv2d()
    in10_2 = InstanceNormalization()
    # + input

    ## res block 8
    refpad11_1 = ReflectionPad2d(1)
    conv11_1 = Conv2d()
    in11_1 = InstanceNormalization()
    # relu
    refpad11_2 = ReflectionPad2d(1)
    conv11_2 = Conv2d()
    in11_2 = InstanceNormalization()
    # + input

    ##------------------------------------##
    deconv01_1 = ConvTranspose2d(128)
    deconv01_2 = Conv2d()
    in12_1 = InstanceNormalization()

    # relu
    deconv02_1 = ConvTranspose2d(64)
    deconv02_2 = Conv2d()
    in13_1 = InstanceNormalization()

    # relu
    refpad12_1 = ReflectionPad2d(3)
    deconv03_1 = Conv2d()

    print(0)
    y = tf.nn.relu(in01_1(conv01_1(refpad01_1(features))))
    y = tf.nn.relu(in02_1(conv02_2(conv02_1(y, pad=1), pad=1)))
    t04 = tf.nn.relu(in03_1(conv03_2(conv03_1(y, pad=1), pad=1)))
    print(1)
    ##
    y = tf.nn.relu(in04_1(conv04_1(refpad04_1(t04))))
    t05 = in04_2(conv04_2(refpad04_2(y))) + t04
    print(2)
    y = tf.nn.relu(in05_1(conv05_1(refpad05_1(t05))))
    t06 = in05_2(conv05_2(refpad05_2(y))) + t05
    print(3)
    y = tf.nn.relu(in06_1(conv06_1(refpad06_1(t06))))
    t07 = in06_2(conv06_2(refpad06_2(y))) + t06
    print(4)
    y = tf.nn.relu(in07_1(conv07_1(refpad07_1(t07))))
    t08 = in07_2(conv07_2(refpad07_2(y))) + t07
    print(5)
    y = tf.nn.relu(in08_1(conv08_1(refpad08_1(t08))))
    t09 = in08_2(conv08_2(refpad08_2(y))) + t08
    print(6)
    y = tf.nn.relu(in09_1(conv09_1(refpad09_1(t09))))
    t10 = in09_2(conv09_2(refpad09_2(y))) + t09
    print(7)
    y = tf.nn.relu(in10_1(conv10_1(refpad10_1(t10))))
    t11 = in10_2(conv10_2(refpad10_2(y))) + t10
    print(8)
    y = tf.nn.relu(in11_1(conv11_1(refpad11_1(t11))))
    y = in11_2(conv11_2(refpad11_2(y))) + t11
    ##
    print(9)
    y = tf.nn.relu(in12_1(deconv01_2(deconv01_1(y), pad=1)))
    y = tf.nn.relu(in13_1(deconv02_2(deconv02_1(y), pad=1)))
    y = tf.nn.tanh(deconv03_1(refpad12_1(y)))
    return y

def _fake_conv2d_transpose(t,deps):
    l = t.get_shape().as_list()
    t = tf.image.resize_bilinear(t,(l[1]*2,l[2]*2))
    return tf.layers.conv2d(t,deps,2, strides=(1, 1), padding='same')

def _anime_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    if not training:
        features = features['input']
    else:
        features.set_shape([params['batch_size'],256,256,3])
    #features = tf.cast(features,tf.float32)
    #features = -1 + 2 * features / 255.0
    deps = params['deps']
    # 32
    y = tf.layers.conv2d(features, deps, 7, strides=(1, 1), padding='same')
    y = tf.layers.batch_normalization(y, axis=-1, training=training)
    y = tf.nn.relu(y)
    deps *= 2
    y = tf.layers.conv2d(y, deps, 3, strides=(2, 2), padding='same')
    y = tf.layers.conv2d(y, deps, 3, strides=(1, 1), padding='same')
    y = tf.layers.batch_normalization(y, axis=-1, training=training)
    y = tf.nn.relu(y)
    deps *= 2
    y = tf.layers.conv2d(y, deps, 3, strides=(2, 2), padding='same')
    y = tf.layers.conv2d(y, deps, 3, strides=(1, 1), padding='same')
    y = tf.layers.batch_normalization(y, axis=-1, training=training)
    t = tf.nn.relu(y)

    for _ in range(params['layers']):
        y = tf.layers.conv2d(t, deps, 3, strides=(1, 1), padding='same')
        y = tf.layers.batch_normalization(y, axis=-1, training=training)
        y = tf.nn.relu(y)
        if params['dropout']>0 and training:
            y = tf.layers.dropout(y,params['dropout'],training=training)
        y = tf.layers.conv2d(y, deps, 3, strides=(1, 1), padding='same')
        y = tf.layers.batch_normalization(y, axis=-1, training=training)
        t = y + t

    deps = int(deps / 2)
    y = _fake_conv2d_transpose(t, deps)
    y = tf.layers.conv2d(y, deps, 3, strides=(1, 1), padding='same')
    y = tf.layers.batch_normalization(y, axis=-1, training=training)
    y = tf.nn.relu(y)
    deps = int(deps / 2)
    y = _fake_conv2d_transpose(y, deps)
    y = tf.layers.conv2d(y, deps, 3, strides=(1, 1), padding='same')
    y = tf.layers.batch_normalization(y, axis=-1, training=training)
    y = tf.nn.relu(y)
    y = tf.layers.conv2d(y, 3, 7, strides=(1, 1), padding='same')
    y = tf.nn.tanh(y)
    #pred = (y * 0.5 + 0.5) * 255
    #pred = tf.cast(pred,tf.uint8)
    if training:
        export_outputs = None
        print('init original')
        npzfile = np.load(params['original_weights'])
        data = {}
        for n in npzfile.files:
            data[n] = npzfile[n]
        o = original(features, data)
        exmp = tf.concat([o,y],1)
        exmp = (exmp * 0.5 + 0.5) * 255
        #exmp  = exmp[:, :,[2, 1, 0]]
        tf.summary.image("example",exmp)
        loss = tf.losses.absolute_difference(o, y, reduction=tf.losses.Reduction.MEAN)
        g = tf.get_default_graph()
        if training:
            tf.contrib.quantize.create_training_graph(input_graph=g)
        else:
            tf.contrib.quantize.create_eval_graph(input_graph=g)

        opt = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step=tf.train.get_or_create_global_step())
    else:
        loss = None
        train_op = None
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(y)}
    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops={},
        predictions=pred,
        loss=loss,
        training_hooks=[],
        evaluation_hooks=[],
        export_outputs=export_outputs,
        train_op=train_op)


class Anime(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _anime_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config,
                model_dir=model_dir,
            )

        super(Anime, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )


def input_fn(params):
    files = glob.glob(params['data_set'])
    batch_size = params['batch_size']

    def _input_fn():
        ds = tf.data.Dataset.from_tensor_slices(files)

        def _image(f):
            f = str(f, encoding='UTF-8')
            img = PIL.Image.open(f).convert("RGB")
            img = img.resize((256, 256), PIL.Image.BICUBIC)
            img = np.asarray(img, np.float32)
            img = img[:, :, [2, 1, 0]]
            #features = tf.cast(features,tf.float32)
            imt = -1 + 2 * img / 255.0
            return img, np.array([1], dtype=np.int32)

        ds = ds.map(lambda f: tuple(tf.py_func(_image, [f], [tf.float32, tf.int32])), num_parallel_calls=1)
        ds = ds.shuffle(batch_size * 4).repeat(params['epoch']).batch(batch_size)
        return ds

    return _input_fn
