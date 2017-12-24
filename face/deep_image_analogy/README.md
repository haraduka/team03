# Deep Image Analogy in PyTorch

## Requirements
- python 3.5
- pytorch 0.2.0
- torchvision 0.1.9

## Usage
`python deep_image_analogy.py [GPU id] [Image_A_path] [Image_Bp_path] [Out_path] [--simple_mode(optional)]`

For example,  
CPU:  
`python deep_image_analogy.py -1 demo/content.png demo/style.png demo/out.png --simple_mode`  
GPU:   
`python deep_image_analogy.py [GPU id] demo/content.png demo/style.png demo/out.png`

## Acknowledgements
Our codes acknowledge
- [Deep-Image-Analogy-PyTorch](https://github.com/harveyslash/Deep-Image-Analogy-PyTorch)
  - PatchMatch code
  - We used other Deep-Image-Analogy code as reference

Thanks!
