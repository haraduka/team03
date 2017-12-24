import argparse
import numpy as np
import cv2
import copy
import torch

import image_loader
from VGG import VGG
from third_party.PatchMatch.PatchMatchOrig import PatchMatch

def blend_features(F_A, R_Bp, alpha, tau=0.05, kappa=300.):
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    F_A_normed = F_A**2
    F_A_normed = F_A_normed/np.max(F_A_normed)
    F_A_thresh = sigmoid(kappa*(F_A_normed - tau))

    W_A = alpha * F_A_thresh
    F_Ap = F_A * W_A + R_Bp * (1. - W_A)
    return F_Ap

def normalize(feat_map):
    return feat_map/np.linalg.norm(feat_map,ord=2,axis=(2),keepdims=True)

def main(A_path, Bp_path, out, use_gpu, simple_mode):

    patch_sizes = [3, 3, 3, 5, 5] # PM patch sizes
    rand_radii = [14, 6, 6, 4, 4] # PM random search radii

    alphas = [0.8, 0.7, 0.6, 0.1]
    if simple_mode:
        deconv_iters = [5, 5, 5, 5]
        pm_iters = [5, 1, 1, 1, 1]
    else:
        deconv_iters = [1700, 1700, 1700, 1700]
        pm_iters = [5, 5, 5, 5, 5]

    model = VGG(use_gpu=use_gpu)

    imaga_A_tensor  = image_loader.load(A_path)
    image_Bp_tensor = image_loader.load(Bp_path)

    F_A_list  = model.get_features(imaga_A_tensor)
    F_Bp_list = model.get_features(image_Bp_tensor)

    for L in [5, 4, 3, 2, 1]:
        index = 5 - L
        print('L:', L)
        F_A  = F_A_list[L-1]
        F_Bp = F_Bp_list[L-1]

        if L < 5:
            R_Bp = model.deconv(prev_F_Bp, deconv_iters[index-1], L, 'Adam')
            R_A  = model.deconv(prev_F_A, deconv_iters[index-1], L, 'Adam')
            F_Ap = blend_features(F_A, R_Bp, alphas[index-1])
            F_B  =  blend_features(F_Bp, R_A, alphas[index-1])
        else:
            F_Ap = F_A
            F_B = F_Bp

        F_A_normed  = normalize(F_A)
        F_Bp_normed = normalize(F_Bp)
        F_Ap_normed = normalize(F_Ap)
        F_B_normed  = normalize(F_B)

        pm_AB = PatchMatch(F_A_normed, F_Ap_normed, F_B_normed, F_Bp_normed, patch_sizes[index])
        pm_BA = PatchMatch(F_Bp_normed, F_B_normed, F_Ap_normed, F_A_normed, patch_sizes[index])

        if L < 5:
            pm_AB.nnf = pmab_prev.upsample_nnf(size=pm_AB.nnf.shape[0])
            pm_BA.nnf = pmba_prev.upsample_nnf(size=pm_BA.nnf.shape[0])

        pm_AB.propagate(iters=pm_iters[index], rand_search_radius=rand_radii[index])
        if L == 1:
            break
        pm_BA.propagate(iters=pm_iters[index], rand_search_radius=rand_radii[index])

        F_Bp_warped = pm_AB.reconstruct_image(F_Bp)
        F_A_warped  = pm_BA.reconstruct_image(F_A)

        prev_F_Bp = copy.deepcopy(F_Bp_warped)
        prev_F_A  = copy.deepcopy(F_A_warped)
        pmba_prev = copy.deepcopy(pm_BA)
        pmab_prev = copy.deepcopy(pm_AB)

    image_Bp = cv2.imread(Bp_path)
    image_Bp = cv2.resize(image_Bp, (224, 224))
    reconstructed_B = pm_AB.reconstruct_avg(image_Bp, patch_size=2)
    cv2.imwrite(out, reconstructed_B)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_id',
                        type=int,
                        help='GPU id. Please set -1 to use CPU')
    parser.add_argument('content',
                        help='Path to content image.')
    parser.add_argument('style',
                        help='Path to style image.')
    parser.add_argument('out',
                        help='Path to output image.')
    parser.add_argument('--simple_mode',
                        dest='simple_mode',
                        default=False,
                        action='store_true',
                        help='Run in simple(fast) mode')
    args = parser.parse_args()

    if args.gpu_id == -1:
        use_gpu = False
        main(args.content, args.style, args.out, use_gpu, args.simple_mode)
    else:
        use_gpu = True
        with torch.cuda.device(args.gpu_id):
            main(args.content, args.style, args.out, use_gpu, args.simple_mode)
