# pytorch-pwc
This is a personal reimplementation of PWC-Net [1] using PyTorch. Should you be making use of this work, please cite the paper accordingly. Also, make sure to adhere to the <a href="https://github.com/NVlabs/PWC-Net#license">licensing terms</a> of the authors. Should you be making use of this particular implementation, please acknowledge it appropriately.

<a href="https://arxiv.org/abs/1709.02371" rel="Paper"><img src="http://www.arxiv-sanity.com/static/thumbs/1709.02371v1.pdf.jpg" alt="Paper" width="100%"></a>

For the original version of this work, please see: https://github.com/NVlabs/PWC-Net
<br />
Another optical flow implementation from me: https://github.com/sniklaus/pytorch-spynet

## background
The authors of PWC-Net are thankfully already providing a reference implementation in PyTorch. However, it performs worse than their original Caffe version. To address this issue, the implementation in this repository replicates the official Caffe version and does thus not have the shortcomings of the official PyTorch implementation.

## setup
The correlation layer is implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided binary packages as outlined in the CuPy repository.

## usage
To run it on your own pair of images, use the following command. You can choose between two models, please make sure to see their paper / the code for more details.

```
python run.py --model sintel --first ./images/first.png --second ./images/second.png --out ./out.flo
```

I am afraid that I cannot guarantee that this reimplementation is correct. However, it produced results identical to the Caffe implementation of the original authors in the examples that I tried. Please feel free to contribute to this repository by submitting issues and pull requests.

## comparison
<p align="center"><img src="comparison/comparison.gif?raw=true" alt="Comparison"></p>

## license
As stated in the <a href="https://github.com/NVlabs/PWC-Net#license">licensing terms</a> of the authors of the paper, the models are free for non-commercial share-alike purpose. Please make sure to further consult their licensing terms.

## references
```
[1]  @inproceedings{Sun_CVPR_2018,
         author = {Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz},
         title = {{PWC-Net}: {CNNs} for Optical Flow Using Pyramid, Warping, and Cost Volume},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2018}
     }
```