import tensorflow as tf
import argparse
import os
import logging
import configparser
from model.anime import Anime
from model.anime import input_fn
import json


def args_str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    conf_parser = argparse.ArgumentParser(
        add_help=False
    )
    conf_parser.add_argument(
        '--checkpoint_dir',
        default=os.environ.get('TRAINING_DIR', 'training') + '/' + os.environ.get('BUILD_ID', '1'),
        help='Directory to save checkpoints and logs',
    )
    args, remaining_argv = conf_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    checkpoint_dir = args.checkpoint_dir
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size.',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=-1.0,
        help='Drouput',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-2,
        help='Recommended learning_rate is 2e-4',
    )

    parser.add_argument(
        '--epoch',
        type=int,
        default=1,
        help='Epoch to trian',
    )
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        default=100,
        help="Log summary every 'save_summary_steps' steps",
    )
    parser.add_argument(
        '--save_checkpoints_secs',
        type=int,
        default=600,
        help="Save checkpoints every 'save_checkpoints_secs' secs.",
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        type=int,
        default=1000,
        help="Save checkpoints every 'save_checkpoints_steps' steps",
    )
    parser.add_argument(
        '--keep_checkpoint_max',
        type=int,
        default=5,
        help='The maximum number of recent checkpoint files to keep.',
    )
    parser.add_argument(
        '--log_step_count_steps',
        type=int,
        default=100,
        help='The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.',
    )
    parser.add_argument(
        '--deps',
        type=int,
        default=16,
        help='Deps',
    )
    parser.add_argument(
        '--layers',
        type=int,
        default=4,
        help='Deps',
    )
    parser.add_argument(
        '--data_set',
        default='',
        help='Location of training files or evaluation files',
    )
    parser.add_argument(
        '--original_weights',
        default='./constants.npz',
        help='Weights of original',
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(worker=False)
    group.set_defaults(evaluator=False)
    group.set_defaults(test=False)
    group.add_argument('--worker', dest='worker', action='store_true',
                       help='Start in Worker(training) mode.')
    group.add_argument('--evaluator', dest='evaluator', action='store_true',
                       help='Start in evaluation mode')
    group.add_argument('--test', dest='test', action='store_true',
                       help='Test mode')
    group.add_argument('--export', dest='export', action='store_true',
                       help='Export model')
    p_file = os.path.join(checkpoint_dir, 'parameters.ini')
    if tf.gfile.Exists(p_file):
        parameters = configparser.ConfigParser(allow_no_value=True)
        parameters.read(p_file)
        parser.set_defaults(**dict(parameters.items("PARAMETERS", raw=True)))
    args = parser.parse_args(remaining_argv)
    print('\n*************************\n')
    print(args)
    print('\n*************************\n')
    return checkpoint_dir, args


def export(checkpoint_dir, params):
    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
    )
    feature_placeholders = {
        'input': tf.placeholder(tf.float32, [None, None, None, 3], name='input'),

    }
    receiver = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_placeholders)
    net = Anime(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
    )
    export_path = net.export_savedmodel(
        checkpoint_dir,
        receiver,
    )
    export_path = export_path.decode("utf-8")
    logging.info('Export path: {}'.format(export_path))


def train(mode, checkpoint_dir, params):
    logging.info("start build  model")

    save_summary_steps = params['save_summary_steps']
    save_checkpoints_secs = params['save_checkpoints_secs'] if params['save_checkpoints_steps'] is None else None
    save_checkpoints_steps = params['save_checkpoints_steps']

    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        save_summary_steps=save_summary_steps,
        save_checkpoints_secs=save_checkpoints_secs,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=params['keep_checkpoint_max'],
        log_step_count_steps=params['log_step_count_steps'],
    )

    net = Anime(
        params=params,
        model_dir=checkpoint_dir,
        config=conf
    )
    logging.info("Start %s mode", mode)
    if mode == 'train':
        fn = input_fn(params)
        net.train(input_fn=fn)
    else:
        logging.info("Not implemented")


def main():
    checkpoint_dir, args = parse_args()
    logging.info('------------------')
    logging.info('TF VERSION: {}'.format(tf.__version__))
    logging.info('ARGS: {}'.format(args))
    logging.info('------------------')
    if args.worker:
        mode = 'train'
    elif args.test:
        mode = 'test'
    else:
        mode = 'eval'
        cluster = {'chief': ['fake_worker1:2222'],
                   'ps': ['fake_ps:2222'],
                   'worker': ['fake_worker2:2222']}
        os.environ['TF_CONFIG'] = json.dumps(
            {
                'cluster': cluster,
                'task': {'type': 'evaluator', 'index': 0}
            })

    params = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'save_summary_steps': args.save_summary_steps,
        'save_checkpoints_steps': args.save_checkpoints_steps,
        'save_checkpoints_secs': args.save_checkpoints_secs,
        'keep_checkpoint_max': args.keep_checkpoint_max,
        'log_step_count_steps': args.log_step_count_steps,
        'data_set': args.data_set,
        'epoch': args.epoch,
        'original_weights': args.original_weights,
        'deps': args.deps,
        'layers': args.layers,
        'dropout':args.dropout,

    }
    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)

    if args.export:
        export(checkpoint_dir, params)
        return
    train(mode, checkpoint_dir, params)


if __name__ == '__main__':
    main()
