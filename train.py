#!/usr/bin/env python

import argparse

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
    parser.add_argument('--style_weight', type=float, default=1e-4,
                        help='style weight')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='training batch size')

    parser.add_argument('--data_dir', default='/tmp',
                        help='dataset directory - requires ~120gb')
    parser.add_argument('--saved_model', default='saved/model',
                        help='where to save the trained model.')

    return parser.parse_args()


def get_coco_ds(data_dir, batch_size):
    ds = tfds.load('coco/2014', split='train', data_dir=data_dir)
    ds = ds.map( lambda data: tf.cast(data['image'], dtype=tf.float32) )
    ds = ds.map( lambda image: tf.image.resize(image, [256,256]) )
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE)


def style_image_preprocess(image):
    image = tf.image.resize(image, [512,512])
    image = tf.image.random_crop(image, [256,256,3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.5)
    image = tf.image.random_contrast(image, 0.5, 1.5)
    return tf.clip_by_value(image, 0.0, 255.0)


def get_dtd_ds(data_dir, batch_size):
    ds = tfds.load('dtd', split='test+train+validation', data_dir=data_dir)
    ds = ds.map( lambda data: tf.cast(data['image'], dtype=tf.float32) )
    ds = ds.map( style_image_preprocess )
    ds = ds.shuffle(1000).repeat()
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE)


def train_model(
    bottleneck_dim,
    content_weight, style_weight,
    content_ds, style_ds, epochs,
):
    train_ds = tf.data.Dataset.zip((content_ds,style_ds))

    model = nst.ghiasi_2017.StyleTransferModel(
        bottleneck_dim=bottleneck_dim,
        content_weight=content_weight,
        style_weight=style_weight,
    )
    model.compile(optimizer='adam')
    model.fit(train_ds, epochs=epochs)

    return model


if __name__ == '__main__':

    args = parse_args()

    content_ds = get_coco_ds(args.data_dir, args.batch_size)
    style_ds = get_dtd_ds(args.data_dir, args.batch_size)

    model = train_model(
        args.bottleneck_dim,
        args.content_weight, args.style_weight,
        content_ds, style_ds, args.epochs,
    )

    model.save(args.saved_model, save_traces=False)
