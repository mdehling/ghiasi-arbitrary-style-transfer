Arbitrary Style Transfer (Ghiasi et al, 2017)
=============================================
In this repository I aim to give a brief description and demonstration of
Ghiasi's approach to neural style transfer.  The implementation used is part
of my python package [`nstesia`](https://github.com/mdehling/nstesia/) and can
be found in its module [`ghiasi_2017`](
https://github.com/mdehling/nstesia/blob/main/src/nstesia/ghiasi_2017.py).

> [!NOTE]
> There are two simple ways to run the demo notebook yourself without installing
> any software on your local machine:
>
> 1. View the notebook on GitHub and click the _Open in Colab_ button (requires
>    a Google account).
> 2. Create a GitHub Codespace for this repository and run the notebook in
>    VSCode (requires a GitHub account).

Introduction
------------
In 2016, Johnson, Alahi, and Fei-Fei published their article _Perceptual
Losses for Real-Time Style Transfer and Super-Resolution_ in which they
introduced a style transfer network which, once trained for a given style
image, allows for fast neural stylization of arbitrary input images using a
single forward pass.  Soon after, Ulyanov, Vedaldi, and Lempitsky submitted
their note _Instance Normalization: The Missing Ingredient for Fast
Stylization_, in which they showed that replacing the batch normalization
layers of the model by instance normalization lead to a significant
improvement in visual results.

In their 2017 article _A Learned Representation for Artistic Style_, Dumoulin,
Kudlur, and Shlens presented a way to perform neural style transfer for
multiple styles—or a mix of styles—using the forward pass of a single style
transfer network trained on a collection of style images simultaneously.
Their key realization was that the learned filters of the style transfer
network could be shared between different styles, and it is sufficient to
learn a different set of normalization parameters for each style.  To allow
for this, they proposed to replace each batch or instance normalization layer
by their new conditional instance normalization layer which learns $N$ sets of
instance normalization parameters and takes an $N$-dimensional style vector as
a second input to select which (linear combination of) parameters to use.

Still in 2017, Ghiasi, Lee, Kudlur, Dumoulin, and Shlens published their
article _Exploring the Structure of a Real-Time, Arbitrary Neural Artistic
Stylization Network_.  Their main contribution was the introduction of a
style prediction network:  Instead of learning a set of normalization
parameters for each of $N$ style images, they chose a bottleneck dimension $N$
and employed a style prediction network to map style images to $N$-dimensional
style vectors.  The final stylization model takes a pair consisting of a
content and a style image as input and produces a pastiche image.

The Style Transfer Network
--------------------------
The style transfer network has three main parts: the encoder, the bottleneck,
and the decoder.  In addition, some pre- and post-processing is performed.
Below I give a description of the various parts of the network.  The given
output dimensions are based on input image size of 256x256, but note they are
provided for illustration only; the network is fully convolutional and
handles input images of any size.

The style transfer network takes two inputs: a content image and a style
vector.  The style vector is produced from a style image using the style
prediction network which I describe further down.

### Pre-Processing
Input images are assumed to take RGB values in `0.0..255.0`. The preprocessing
layer simply subtracts the ImageNet means.

```text
Layer                   Description                        Output Size
----------------------------------------------------------------------
preprocess              Pre-Processing                     256x256x3
```

### The Encoder
The encoder is composed of three convolutional blocks, each consisting of a
reflection padding layer, a convolutional layer, a normalization layer, and a
`relu` activation layer.  All padding layers of the transfer model use `same`
amount of reflection padding, i.e., the amount of padding is calculated by the
same formulas as it is for `same` padding, but the type of padding applied is
reflection padding instead of constant.  The second and third block uses
strided convolutions to reduce the spatial dimensions by a total factor of 4.

```text
Block / Layer           Description                        Output Size
----------------------------------------------------------------------
down_block_1 / rpad     ReflectionPadding
             / conv     Convolution (32, 9x9, stride 1)
             / norm     BatchNormalization
             / act      Activation (ReLU)                  256x256x32

down_block_2 / rpad     ReflectionPadding
             / conv     Convolution (64, 3x3, stride 2)
             / norm     BatchNormalization
             / act      Activation (ReLU)                  128x128x64

down_block_3 / rpad     ReflectionPadding
             / conv     Convolution (128, 3x3, stride 2)
             / norm     BatchNormalization
             / act      Activation (ReLU)                  64x64x128
```

### The Bottleneck
The bottleneck comprises five residual blocks, each of which consists of seven
layers and a residual connection.  In order, the layers are: padding,
convolution, normalization, and `relu` activation, followed by another
padding, convolution, and normalization layer.

```text
Block / Layer           Description                        Output Size
----------------------------------------------------------------------
res_block_1..5 / rpad1  ReflectionPadding
               / conv1  Convolution (128, 3x3, stride 1)
               / norm1  ConditionalInstanceNormalization
               / relu1  Activation (ReLU)
               / rpad2  ReflectionPadding
               / conv2  Convolution (128, 3x3, stride 1)
               / norm2  ConditionalInstanceNormalization
               +        Residual                           64x64x128
```

### The Decoder
The decoder roughly mirrors the encoder:  It is again composed of three
convolutional blocks, each of which consist of a reflection padding, a
convolutional layer, a normalization layer, and an activation layer.  The
first two of these layers start with an upsampling layer performing 2x nearest
neighbor upsampling before the other layers and use 'relu' activations, while
the final layer ends with a 'sigmoid' activation.

```text
Block / Layer           Description                        Output Size
----------------------------------------------------------------------
up_block_1 / up         UpSampling (2x, nearest)
           / rpad       ReflectionPadding
           / conv       Convolution (64, 3x3, stride 1)
           / norm       ConditionalInstanceNormalization
           / act        Activation (ReLU)                  128x128x64

up_block_2 / up         UpSampling (2x, nearest)
           / rpad       ReflectionPadding
           / conv       Convolution (32, 3x3, stride 1)
           / norm       ConditionalInstanceNormalization
           / act        Activation (ReLU)                  256x256x32

up_block_3 / rpad       ReflectionPadding
           / conv       Convolution (3, 9x9, stride 1)
           / norm       ConditionalInstanceNormalization
           / act        Activation (Sigmoid)               256x256x3
```

### Post-Processing
The final `sigmoid` activation layer of the decoder gives an output image with
rgb values in `0.0..1.0`.  These values are multiplied by a factor `255.0` to
obtain an output image with values in the desired range.

```text
Layer                   Description                        Output Size
----------------------------------------------------------------------
rescale                 Rescaling (factor 255.0)           256x256x3
```

The Style Prediction Network
----------------------------
The style prediction network takes a style image as input and produces an
$N$-dimensional style vector as output.  It consists of a preprocessing layer
and an Inception-V3 model up to layer 'mixed7' (the keras name for 'Mixed6e'),
followed by spatial averaging and a fully connected layer with $N$ outputs.

```text
Layer                   Description                        Output Size
----------------------------------------------------------------------
prep                    PreProcess                         256x256x3
inception_v3            InceptionV3 ('mixed7')             14x14x768
avg                     GlobalAveragePooling               1x1x768
bneck                   Conv2D, squeeze                    N
```

Training Method
---------------
Let $x_c$ be a content image and $x_s$ be a style image.  Denote by $T$ the
style transfer network and by $P$ the style prediction network.  The style
vector is $v = P(x_s)$ and is used to compute the pastiche image
$x_p = T(x_c, v) = T(x_c, P(x_s))$.

The style transfer and prediction networks are trained jointly to minimize the
weighted sum

$$
\mathcal{L}(x_c,x_s,x_p) =
w_C\cdot\mathcal{L}_C(x_c,x_p) + w_S\cdot\mathcal{L}_S(x_s,x_p) \quad,
$$

where $\mathcal{L}_C$ and $\mathcal{L}_S$ denote the content and style loss as
introduced by Gatys et al.  For my implementation of these losses, see the
module [`gatys_2015`](
https://github.com/mdehling/nstesia/blob/main/src/nstesia/gatys_2015.py).

Training is performed for 50 epochs using an `adam` optimizer with a learning
rate of `1e-3`.  The images of the Microsoft COCO/2014 dataset are used as
content images and as style images one of two different datasets are used: the
describable textures dataset or the painter by numbers dataset.  All content
images are resized to 256x256 without any other processing; image augmentation
is used for the style images (random cropping, flipping, and hue/contrast
adjustments) before finally resizing them to 256x256.

Results
-------
This repository contains a python script [`train.py`](train.py) which lets you
adjust some training parameters, downloads the training dataset, performs
joint training of the style transfer and prediction networks, and finally
saves the trained combined model to disk.

> **Note**
> Results are coming soon...

References
----------
* Johnson, Alahi, Fei-Fei - _Perceptual Losses for Real-Time Style Transfer
  and Super-Resolution_, 2016.
  [[pdf]](https://link.springer.com/content/pdf/10.1007/978-3-319-46475-6_43.pdf)
  [[suppl]](https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-319-46475-6_43/MediaObjects/419974_1_En_43_MOESM1_ESM.pdf)
  [[code]](https://github.com/jcjohnson/fast-neural-style)
* Ulyanov, Vedaldi, Lempitsky - _Instance Normalization: The Missing
  Ingredient for Fast Stylization_, 2016.
  [[arxiv]](https://arxiv.org/abs/1607.08022)
* Dumoulin, Kudlur, Shlens - _A Learned Representation for Artistic Style_,
  2017.
  [[arxiv]](https://arxiv.org/abs/1610.07629)
  [[code]](https://github.com/magenta/magenta/tree/main/magenta/models/image_stylization)
* Ghiasi, Lee, Kudlur, Dumoulin, Shlens - _Exploring the Structure of a
  Real-Time, Arbitrary Neural Stylization Network_, 2017.
  [[pdf]](http://www.bmva.org/bmvc/2017/papers/paper114/paper114.pdf)
  [[suppl]](http://www.bmva.org/bmvc/2017/papers/paper114/supplementary114.pdf)
* Gatys, Ecker, Bethge - _A Neural Algorithm of Artistic Style_, 2015.
  [[pdf]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
* Lin et al - _Microsoft COCO: Common Objects in Context_, 2014.
  [[arxiv]](https://arxiv.org/abs/1405.0312)
  [[www]](https://cocodataset.org/)
* Cimpoi et al - _Describing Textures in the Wild_, 2014.
  [[pdf]](https://www.robots.ox.ac.uk/~vgg/publications/2014/Cimpoi14/cimpoi14.pdf)
  [[www]](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
* Kiri Nichol - _Painter by Numbers_, 2016.
  [[www]](https://www.kaggle.com/c/painter-by-numbers)
