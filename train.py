#!/usr/bin/env python

import argparse
from pathlib import Path

from os import environ as env
env['TF_CPP_MIN_LOG_LEVEL'] = '2'               # hide info & warnings
env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'       # grow GPU memory as needed

import tensorflow as tf
import tensorflow_datasets as tfds
import nstesia as nst


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Ghiasi (2017) style transfer model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--bottleneck_dim', type=int, default=100,
                        help='bottleneck dimension')
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='content weight')
    parser.add_argument('--style_weight', type=float, default=1e-3,
                        help='style weight')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                        help='learning rate of the adam optimizer')

    parser.add_argument('--epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')

    parser.add_argument('--style_dataset', default='dtd',
                        choices=['dtd', 'pbn'],
                        help='training style dataset')
    parser.add_argument('--data_dir', default='/tmp',
                        help='dataset directory - requires ~120gb')

    parser.add_argument('--load_model',
                        help='load a model to resume training')
    parser.add_argument('--save_model', default='saved/model',
                        help='where to save the trained model')

    return parser.parse_args()


def get_coco_ds(data_dir, batch_size):
    ds = tfds.load('coco/2014', split='train', data_dir=data_dir)
    ds = ds.map( lambda data: tf.cast(data['image'], dtype=tf.float32) )
    ds = ds.map( lambda image: tf.image.resize(image, [256,256]) )
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE)


def style_image_preprocess(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.5 * 255.0)
    image = tf.image.random_saturation(image, 0.5, 1.5)
    image = tf.image.random_hue(image, 0.2)
    return tf.clip_by_value(image, 0.0, 255.0)


def get_dtd_ds(data_dir, batch_size):
    ds = tfds.load('dtd', split='all', data_dir=data_dir)
    ds = ds.map( lambda data: tf.cast(data['image'], dtype=tf.float32) )
    ds = ds.map( lambda image: tf.image.random_crop(image, [256,256,3]) )
    ds = ds.shuffle(1000).batch(batch_size, drop_remainder=True)
    ds = ds.map( style_image_preprocess )
    ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


def get_pbn_ds(data_dir, batch_size):
    ds = tf.keras.utils.image_dataset_from_directory(
        Path(data_dir) / 'pbn/train',
        label_mode=None,
        batch_size=None,
        shuffle=False,
        image_size=(512,512),
    )
    ds = ds.map( lambda image: tf.image.random_crop(image, [256,256,3]) )
    ds = ds.shuffle(1000).batch(batch_size, drop_remainder=True)
    ds = ds.map( style_image_preprocess )
    return ds.prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':

    args = parse_args()

    content_ds = get_coco_ds(args.data_dir, args.batch_size)

    if args.style_dataset == 'dtd':
        style_ds = get_dtd_ds(args.data_dir, args.batch_size)
    else:
        style_ds = get_pbn_ds(args.data_dir, args.batch_size)

    train_ds = tf.data.Dataset.zip((content_ds,style_ds))

    if args.load_model:
        model = nst.ghiasi_2017.StyleTransferModel.from_saved(args.load_model)
    else:
        model = nst.ghiasi_2017.StyleTransferModel(
            bottleneck_dim=args.bottleneck_dim,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
        )
        model.compile( tf.keras.optimizers.Adam(args.learning_rate) )

    model.fit(train_ds, epochs=args.epochs)

    model.save(args.save_model, save_traces=False)
