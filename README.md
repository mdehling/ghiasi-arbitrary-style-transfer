Arbitrary Style Transfer (Ghiasi et al, 2017)
=============================================
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
a second input to select which (linear combination of) parameters to use.  For
more details, see the 'Network Architecture' section below.

In this repository I aim to give a brief description and demonstration of
Dumoulin's style transfer model.  The implementation used is part of my python
package [`nstesia`](https://github.com/mdehling/nstesia/) and can be found in
its module [`dumoulin_2017`](
https://github.com/mdehling/nstesia/blob/main/src/nstesia/dumoulin_2017.py).

Network Architecture
--------------------
The style transfer network has three main parts: the encoder, the bottleneck,
and the decoder.  In addition, some pre- and post-processing is performed.
Below I give a description of the various parts of the network.  The given
output dimensions are based on input image size of 256x256, but note they are
provided for illustration only; the network is fully convolutional and
handles input images of any size.

### Pre-Processing
Input images are assumed to take RGB values in `0.0..255.0`. The preprocessing
layer centers the RGB values around their ImageNet means.  Note that the
padding layer of Johnson's model was removed: Padding is applied to each
convolution layer individually instead of in one chunk here.

```text
Layer                   Description                        Output Size
----------------------------------------------------------------------
preprocess              Pre-Processing                     256x256x3
```

### The Encoder
The encoder is composed of three convolutional blocks, each consisting of a
reflection padding layer, a convolutional layer, a normalization layer, and a
`relu` activation layer.  All reflection layers of the transfer model use
`same` amount of reflection padding, i.e., the amount of padding is calculated
by the same formulas as it is for `same` padding, but the type of padding
applied is reflection padding instead of constant.  The second and third block
uses strided convolutions to reduce the spatial dimensions by a total factor
of 4.

```text
Block / Layer           Description                        Output Size
----------------------------------------------------------------------
down_block_1 / rpad     ReflectionPadding
             / conv     Convolution (32, 9x9, stride 1)
             / norm     ConditionalInstanceNormalization
             / act      Activation (ReLU)                  256x256x32

down_block_2 / rpad     ReflectionPadding
             / conv     Convolution (64, 3x3, stride 2)
             / norm     ConditionalInstanceNormalization
             / act      Activation (ReLU)                  128x128x64

down_block_3 / rpad     ReflectionPadding
             / conv     Convolution (128, 3x3, stride 2)
             / norm     Batch/InstanceNormalization
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
               + res    Residual                           64x64x128
```

### The Decoder
The decoder roughly mirrors the encoder:  It is again composed of three
convolutional blocks, each of which consist of a reflection padding, a
convolutional layer, a normalization layer, and an activation layer.  The
first two of these layers start with an upsampling layer performing 2x nearest
neighbor upsampling before the other layers and use 'relu' activations, while
the final layer end with a 'sigmoid' activation.

```text
Block / Layer           Description                        Output Size
----------------------------------------------------------------------
up_block_1 / up       UpSampling (2x, nearest)
           / rpad     ReflectionPadding
           / conv     Convolution (64, 3x3, stride 1)
           / norm     ConditionalInstanceNormalization
           / act      Activation (ReLU)                  128x128x64

up_block_2 / up       UpSampling (2x, nearest)
           / rpad     ReflectionPadding
           / conv     Convolution (32, 3x3, stride 1)
           / norm     ConditionalInstanceNormalization
           / act      Activation (ReLU)                  256x256x32

up_block_3 / rpad     ReflectionPadding
           / conv     Convolution (3, 9x9, stride 1)
           / norm     ConditionalInstanceNormalization
           / act      Activation (Sigmoid)               256x256x3
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

Training Method
---------------
Let $x_s$ be a chosen style image and denote by $T_s$ the _image 
transformation network_.  The goal is to produce, for any content image
$x_c$, a pastiche image $x_p = T_s(x_c)$ using a single forward pass of the
network.  The objective formulated to achieve this goal is to minimize a
weighted sum

$$
\mathcal{L}(x_c,x_s,x_p) =
w_C\cdot\mathcal{L}_C(x_c,x_p) + w_S\cdot\mathcal{L}_S(x_s,x_p) \quad,
$$

where $\mathcal{L}_C$ and $\mathcal{L}_S$ denote the content and style loss as
introduced by Gatys et al.  For my implementation of these losses, see the
module [`gatys_2015`](
https://github.com/mdehling/nstesia/blob/main/src/nstesia/gatys_2015.py).
Note the absence of a variation loss term: the use of upsampling instead of
transposed (or fractionally strided) convolutions makes its use unnecessary.

Training is performed for 8 epochs over the images of the Microsoft COCO/2014
dataset using an `adam` optimizer with a learning rate of `1e-3`.  All images
are resized to 256x256 and served in batches of 16.

Results
-------
This repository contains a python script [`train.py`](train.py) which takes a
collection of style images as well as some training parameters as input,
downloads the training dataset, performs training of the style transfer model,
and finally saves the trained model to disk.  The directory `saved/model`
contains a model trained in this way for the 32 images in `img/style`.  To try
the model out yourself, have a look at the notebook
[`multi-style-transfer.ipynb`](multi-style-transfer.ipynb).
All images below were produced using it.

> **Note**
> The images included here are lower quality jpeg files.  I have linked them
> to their lossless png versions.

The following are two sets of stylizations of the same content images in the
same styles as used to demonstrate Johnson et al's style transfer network.
Note that all of these pastiches were produced using a single style transfer
network.  The quality of the results here is comparable to that of pastiches
produced by Johnson's networks trained for individual styles—see my repository
[`johnson-fast-style-transfer`](
https://github.com/mdehling/johnson-fast-style-transfer).

[![](img/results/content-style-matrix-1.jpg)
](img/results/content-style-matrix-1.png)

Note that the use of upsampling layers instead of transposed or fractionally
strided convolutions can lead to improved results by eliminating checkerboard
artifacts.  This is particularly clear when comparing the first of the
following stylizations to the the corresponding one created using Johnson's
network.

[![](img/results/content-style-matrix-2.jpg)
](img/results/content-style-matrix-2.png)

The following demonstrates the ability of Dumoulin et al's network to produce
pastiches in mixed styles.

[![](img/results/style-mix-matrix.jpg)
](img/results/style-mix-matrix.png)

References
----------
* Dumoulin, Kudlur, Shlens - _A Learned Representation for Artistic Style_,
  2017.
  [[arxiv]](https://arxiv.org/abs/1610.07629)
  [[code]](https://github.com/magenta/magenta/tree/main/magenta/models/image_stylization)
* Johnson, Alahi, Fei-Fei - _Perceptual Losses for Real-Time Style Transfer
  and Super-Resolution_, 2016.
  [[pdf]](https://link.springer.com/content/pdf/10.1007/978-3-319-46475-6_43.pdf)
  [[suppl]](https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-319-46475-6_43/MediaObjects/419974_1_En_43_MOESM1_ESM.pdf)
  [[code]](https://github.com/jcjohnson/fast-neural-style)
* Ulyanov, Vedaldi, Lempitsky - _Instance Normalization: The Missing
  Ingredient for Fast Stylization_, 2016.
  [[arxiv]](https://arxiv.org/abs/1607.08022)
* Gatys, Ecker, Bethge - _A Neural Algorithm of Artistic Style_, 2015.
  [[pdf]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
* Lin et al - _Microsoft COCO: Common Objects in Context_, 2014.
  [[www]](https://cocodataset.org/)
  [[arxiv]](https://arxiv.org/abs/1405.0312)
