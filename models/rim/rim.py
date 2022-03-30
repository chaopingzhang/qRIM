import torch
from torch import nn

from training_utils.helpers import loss_est_analytic

class RIM(nn.Module):

    def __init__(self, rnn, grad_fun):
        super(RIM, self).__init__()
        self.rnn = rnn
        self.grad_fun = grad_fun

    def forward(self, eta, scaling, data, hx=None, n_steps=1, accumulate_eta=False, coil_sum_method='rss'):
        """
        :param eta: Starting value for eta [n_batch,features,height,width]
        :param grad_fun: The gradient function, takes as input eta and outputs gradient of same dimensionality
        :param hx: Hidden state of the RNN
        :param n_steps: Number of time steps, that the RIM should perform. Default: 1
        :param accumulate_eta: Bool, if True will save all intermediate etas in a list, else outputs only the last eta.
                               Default: False
        :return: etas, hx
        """
        etas = []

        for i in range(n_steps):
            y, y_ksp, mask_subsampling, mask_brain, TEs, sense = data[0], data[1], data[2], data[3], data[4], data[5]
            grad_eta = torch.zeros_like(eta) 
            batchsize = eta.size(0)

            # analytic grad
            for k in range(batchsize):
                sense_k = sense[k,...].squeeze()
                mask_brain_k = mask_brain[k,...].squeeze()
                mask_subsampling_k = mask_subsampling[k,...]
                eta_k = eta[k,...]*scaling[0,...]

                ksp = y_ksp[k,...].squeeze()
                grad_eta_k = loss_est_analytic(self.grad_fun, TEs, sense_k, [eta_k[i] for i in range(eta_k.size(0))], ksp, mask_subsampling_k, mask_brain_k)
                grad_eta[k, ...] = grad_eta_k / 100
                grad_eta[grad_eta != grad_eta] = 0

            x_in = torch.cat((eta, grad_eta), 1)
            delta, hx = self.rnn.forward(x_in, hx)

            eta = eta + delta
            eta_tmp = eta[:,0,:,:]
            eta_tmp[eta_tmp < 0] = 0
            eta[:,0,:,:] = eta_tmp

            if accumulate_eta:
                etas.append(eta)

        if not accumulate_eta:
            etas = eta

        return etas, hx