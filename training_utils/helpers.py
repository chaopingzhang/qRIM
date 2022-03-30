import numpy as np
import torch
from torch.nn import functional as F

from fastMRI.data import transforms
from training_utils import ssim

def isnan(x):
    return np.isin(True, (x != x).cpu().numpy())


def real_to_complex(x):
    return torch.stack(torch.chunk(x, 2, 1), -1)


def complex_to_real(x):
    return torch.cat((x[..., 0], x[..., 1]), 1)


def img_to_ksp_multicoil(img, sense):
    return transforms.fft2(img_to_multicoil(img, sense))

    
def img_to_multicoil(img, sense):
    re_sense, im_sense = sense.chunk(2, -1)
    re_eta, im_eta = torch.unsqueeze(img, -4).chunk(2, -1)

    re_se = re_eta * re_sense - im_eta * im_sense
    im_se = re_eta * im_sense + im_eta * re_sense

    sensed_imgs = torch.cat((re_se, im_se), -1)
    if torch.isnan(sensed_imgs).any():
        sensed_imgs[sensed_imgs != sensed_imgs] = 0
    return sensed_imgs


def combine_imgs_from_multicoil(image, sense, coil_sum_method=None):
    re_sense, im_sense = sense.chunk(2, -1)
    re_image, im_image = image.chunk(2, -1)

    if coil_sum_method == 'rss':
        re_out = transforms.root_sum_of_squares(re_image * re_sense + im_image * im_sense, 2)
        im_out = transforms.root_sum_of_squares(im_image * re_sense - re_image * im_sense, 2)
    else:
        re_out = torch.sum(re_image * re_sense + im_image * im_sense, 2)
        im_out = torch.sum(im_image * re_sense - re_image * im_sense, 2)

    return torch.cat((re_out, im_out), -1)

def loss_est_analytic(grad_fun, TEs, sense_k, eta_y, y_ksp, mask_subsampling, mask_brain):
    
    scaling = 1.0e-3
    nr_TEs = len(TEs)
    eta_img = grad_fun(eta_y[0].unsqueeze(0), eta_y[1].unsqueeze(0), eta_y[2].unsqueeze(0), eta_y[3].unsqueeze(0), mask_brain)

    R2star_map = eta_y[0].unsqueeze(0)
    S0_map = eta_y[1].unsqueeze(0)
    B0_map = eta_y[2].unsqueeze(0)
    phi_map = eta_y[3].unsqueeze(0)
    S0_map_real = S0_map
    S0_map_imag = phi_map

    eta_ksp = img_to_ksp_multicoil(eta_img, sense_k)
    diff_data = (eta_ksp - y_ksp) * mask_subsampling.unsqueeze(1).unsqueeze(-1)
    diff_data_inverse = combine_imgs_from_multicoil(transforms.ifft2(diff_data), sense_k.unsqueeze(0))

    S0_part_der = torch.stack([torch.stack((torch.exp(-TEs[i] * scaling * R2star_map) * torch.cos(B0_map * scaling * -TEs[i]),
                          -torch.exp(-TEs[i] * scaling * R2star_map) * torch.sin(B0_map * scaling * -TEs[i])), -1)
            for i in range(nr_TEs)], 1)

    R2str_part_der = torch.stack([torch.stack((-TEs[i] * scaling * torch.exp(-TEs[i] * scaling * R2star_map) * (S0_map_real * torch.cos(B0_map * scaling * -TEs[i]) - S0_map_imag * torch.sin(B0_map * scaling * -TEs[i])),
                          -TEs[i] * scaling * torch.exp(-TEs[i] * scaling * R2star_map) * (-S0_map_real * torch.sin(B0_map * scaling * -TEs[i]) - S0_map_imag * torch.cos(B0_map * scaling * -TEs[i]))), -1)
            for i in range(nr_TEs)], 1)

    S0_map_real_grad = diff_data_inverse[..., 0] * S0_part_der[..., 0] - diff_data_inverse[..., 1] * S0_part_der[..., 1]
    S0_map_imag_grad = diff_data_inverse[..., 0] * S0_part_der[..., 1] + diff_data_inverse[..., 1] * S0_part_der[..., 0]
    R2star_map_real_grad = diff_data_inverse[..., 0] * R2str_part_der[..., 0] - diff_data_inverse[..., 1] * R2str_part_der[..., 1]
    R2star_map_imag_grad = diff_data_inverse[..., 0] * R2str_part_der[..., 1] + diff_data_inverse[..., 1] * R2str_part_der[..., 0]

    S0_map_grad = torch.stack([S0_map_real_grad, S0_map_imag_grad], -1).squeeze()
    S0_map_grad = torch.mean(S0_map_grad, 0)
    R2star_map_grad = torch.stack([R2star_map_real_grad, R2star_map_imag_grad], -1).squeeze()
    R2star_map_grad = torch.mean(R2star_map_grad, 0)

    return torch.stack([R2star_map_grad[..., 0], S0_map_grad[..., 0], R2star_map_grad[..., 1], S0_map_grad[..., 1]], 0)


def img_reshape(image):
    return image.reshape(-1, 1, image.size(-2), image.size(-1))


def reweight(image):
    return -torch.atan(0.1 * (image-120)) + 2


def image_loss(image, target, mask_brain, args):
    loss_selector = {'l1': lambda x, t: F.l1_loss(x, t, reduction='none'),
                     'mse': lambda x, t, m: F.mse_loss(x*m, t*m, reduction='none'),
                     'ssim': lambda x, t, m, img_reshape: -ssim.ssim_uniform(img_reshape(x*m), img_reshape(t*m), window_size=7, reduction='none'),
                     }
    loss_fun = loss_selector[args.loss]

    mask = torch.ones_like(target)
    if args.loss == 'ssim':
        normalizer = target.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        target = target / normalizer
        image = image / normalizer

        lossR2star = loss_fun(image[:,0,:,:], target[:,0,:,:], mask_brain, img_reshape) * 3
        loss_S0 = loss_fun(image[:,1,:,:], target[:,1,:,:], mask_brain, img_reshape) * 1
        loss_B0 = loss_fun(image[:,2,:,:], target[:,2,:,:], mask_brain, img_reshape) * 1
        loss_phi = loss_fun(image[:,3,:,:], target[:,3,:,:], mask_brain, img_reshape) * 1
        loss = torch.cat((lossR2star, loss_S0, loss_B0, loss_phi), dim=1)
    else:
        lossR2star = loss_fun(image[:,0,:,:], target[:,0,:,:], mask_brain)/300.0
        loss_S0 = loss_fun(image[:,1,:,:], target[:,1,:,:], mask_brain)/500.0
        loss_B0= loss_fun(image[:,2,:,:], target[:,2,:,:], mask_brain)/20000.0
        loss_phi = loss_fun(image[:,3,:,:], target[:,3,:,:], mask_brain)/500.0
        loss = torch.stack((lossR2star, loss_S0, loss_B0, loss_phi), dim=1)

    return loss.sum((-2, -1, 0)) / mask.sum((-2, -1, 0)) #, lossR2star, loss_S0, loss_B0, loss_phi
