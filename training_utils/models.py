import torch
from models.rim import RIM


class RescaleByStd(object):
    def __init__(self, slack=1e-6):
        self.slack = slack

    def forward(self, data):
        gamma = data.std(dim=list(range(1, data.dim())), keepdim=True) + self.slack
        data = data / gamma
        return data, gamma

    def reverse(self, data, gamma):
        data = data * gamma
        return data


class RescaleByStd_qMRI(object):
    def __init__(self, slack=1e-6):
        self.slack = slack

    def forward(self, data):
        gamma = data.std(dim=list(range(2, data.dim())), keepdim=True) + self.slack
        data = data / gamma
        return data, gamma

    def reverse(self, data, gamma):
        data = data * gamma
        return data
    

class RescaleByMax(object):
    def __init__(self, slack=1e-6):
        self.slack = slack

    def forward(self, data):
        gamma = torch.max(torch.max(torch.abs(data), 3, keepdim=True)[0], 2, keepdim=True)[0] + self.slack
        data = data / gamma
        return data, gamma

    def reverse(self, data, gamma):
        data = data * gamma
        return data
    

class RIMfastMRI(torch.nn.Module):
    def __init__(self, model, preprocessor=RescaleByMax(), n_steps=8, coil_sum_method='rss'):
        """
        An RIM model wrapper for the fastMRI challenge.
        :param model: RIM model
        :param preprocessor: a function that rescales each sample
        :param n_steps: Number of RIM steps [int]
        """
        super().__init__()
        assert isinstance(model, RIM)
        self.model = model
        self.preprocessor = preprocessor
        self.n_steps = n_steps
        self.coil_sum_method = coil_sum_method

    def forward(self, y, y_ksp, mask_subsampling, mask_brain, TEs, sense=None, metadata=None):
        """
        :param y: Zero-filled kspace reconstruction [Tensor]
        :param mask: Sub-sampling mask
        :param metadata: will be ignored
        :return: complex valued image estimate
        """
        accumulate_eta = self.training

        gamma =  torch.Tensor([[[[ 150.0]],
                
                                [[ 150.0]],
                
                                [[ 1000.0]],
                
                                [[ 150.0]]]])
        
        gamma = gamma.to(y.device)
        y = y/gamma

        eta = y
        eta, hx = self.model.forward(eta, gamma, [y, y_ksp, mask_subsampling, mask_brain, TEs, sense],
                                     n_steps=self.n_steps, accumulate_eta=accumulate_eta,
                                     coil_sum_method=self.coil_sum_method)

        if accumulate_eta:
            eta = [self.preprocessor.reverse(e, gamma) for e in eta]
        else:
            eta = self.preprocessor.reverse(eta, gamma)

        return eta
