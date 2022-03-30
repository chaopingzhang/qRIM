__author__ = 'Kai Lønning, Dimitrios Karkalousos'

import functools as ft
import itertools as it
import math

import numpy as np
import torch


def gaussian_kernel(shape, fwhms):
    """
    Creates the gaussian kernel used in generate_mask. mask_shape should be the
    same as the shape of the MR-image, fwhms are the Full-width-half-maximum
    """
    kernels = []

    for fwhm, kern_len in zip(fwhms, shape):
        sigma = fwhm / np.sqrt(8 * np.log(2))
        x = np.linspace(-1., 1., kern_len)
        g = np.exp(- x ** 2 / (2 * sigma ** 2))
        kernels.append(g)

    if len(kernels) == 1:
        kernel = kernels[0]

    if len(kernels) == 2:
        kernel = np.sqrt(np.einsum('i,j->ij', *kernels))

    if len(kernels) == 3:
        kernel = np.cbrt(np.einsum('i,j,k->ijk', *kernels))

    return kernel / kernel.sum()


def line_rectangle_intersection(shape, theta):
    if abs(np.cos(theta)) < 1e-6:
        xmin, xmax = shape[2] / 2, shape[2] / 2
        ymin, ymax = 0, shape[1] - .1
    else:
        x0, y0 = shape[0] / 2, shape[1] / 2

        if 0 < theta < np.pi:
            y1 = y0
        else:
            y1 = -y0

        if -np.pi / 2 < theta < np.pi / 2:
            x1 = x0
        else:
            x1 = -x0

        y = x1 * np.tan(theta)
        ymax = y0 + y

        if ymax >= shape[1] or ymax < 0:
            x = y1 / np.tan(theta)
            xmax = x0 + x
            xmin = x0 - x
            ymax = shape[1] - .1
            ymin = 0
        else:
            ymin = y0 - y
            xmax = shape[0] - .1
            xmin = 0

    return xmin, ymin, xmax, ymax


class Masker():
    def __init__(self, subsampling_dist, center_fractions, accelerations, fwhms):
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.fwhms = fwhms
        self.rng = np.random.RandomState()
        self.make_mask = getattr(self, subsampling_dist)

    def __call__(self, shape, seed=None):
        """
        Generates masks used for data corruption purposes in such a way that
        samples are picked from k-space in accordance with a centered gaussian
        kernel pdf stored in self.gaussian_kernel. Sample_size says how many
        samples should be picked, n_mask says how many masks to generate for
        each image.
        """
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)

        choice = self.rng.randint(0, len(self.accelerations))
        self.center_fraction = self.center_fractions[choice]
        self.acceleration = self.accelerations[choice]

        shape = np.array((shape[-3], shape[-2]))
        mask = self.make_mask(shape)
        mask = mask.unsqueeze(0).unsqueeze(-1)

        return mask

    def gaussian1d(self, shape):
        shape = shape[-1]
        scale = int(shape * np.random.uniform(*self.center_fractions))

        top = (shape - scale) // 2
        btm = (shape - scale - top)

        mask = np.concatenate((np.zeros(top), np.ones(scale), np.zeros(btm)))
        n_sample = int(shape / np.random.uniform(*self.accelerations))
        kernel = gaussian_kernel((shape,), (np.random.uniform(*self.fwhms),))
        samples = np.random.choice(range(shape), size=n_sample, replace=False, p=kernel)

        mask[samples] = 1.

        shape = [1, shape]

        return torch.from_numpy(mask.reshape(*shape).astype(np.float32))

    def gaussian2d(self, shape):
        scale = np.random.uniform(*self.center_fractions)

        a, b = scale * shape[0], scale * shape[1]
        afocal, bfocal = shape[0] / 2, shape[1] / 2
        xx, yy = np.mgrid[:shape[0], :shape[1]]
        ellipse = np.power((xx - afocal) / a, 2) + np.power((yy - bfocal) / b, 2)

        mask = (ellipse < 1).astype(float)
        n_sample = int(shape[0] * shape[1] / np.random.uniform(*self.accelerations))
        cartesian_prod = [e for e in np.ndindex(*shape)]
        kernel = gaussian_kernel(shape, (np.random.uniform(*self.fwhms), np.random.uniform(*self.fwhms)))

        idxs = np.random.choice(range(len(cartesian_prod)), size=n_sample, replace=False, p=kernel.flatten())
        mask[tuple(zip(*list(map(cartesian_prod.__getitem__, idxs))))] = 1.

        return torch.from_numpy(mask.reshape(*shape).astype(np.float32))

    def gaussian3d(self, shape):
        scale = np.random.uniform(*self.center_fractions)
        shape = np.array((1, shape[-2], shape[-1]))

        d, a, b = map(lambda x: x * scale, shape)
        dfocal, afocal, bfocal = map(lambda x: x / 2, shape)
        zz, xx, yy = np.mgrid[:shape[0], :shape[1], :shape[2]]
        ellipsoid = np.power((zz - dfocal) / d, 2) + np.power((xx - afocal) / a, 2) + np.power((yy - bfocal) / b, 2)

        mask = (ellipsoid < 1).astype(float)
        n_sample = int(ft.reduce(lambda x, y: x * y, shape) / np.random.uniform(*self.accelerations))
        cartesian_prod = [e for e in np.ndindex(*shape)]
        kernel = gaussian_kernel(shape, [np.random.uniform(*self.fwhms) for _ in range(len(shape))])

        idxs = np.random.choice(range(len(cartesian_prod)), size=n_sample, replace=False, p=kernel.flatten())
        mask[tuple(zip(*list(map(cartesian_prod.__getitem__, idxs))))] = 1.

        return torch.from_numpy(mask.reshape(*shape).astype(np.float32)).squeeze(0)

    def radial2d(self, shape):
        mask = np.zeros(shape)
        n_sample = int(shape[0] * shape[1] / np.random.uniform(*self.accelerations))

        current_samples = set()
        samples = set()
        angle = np.random.uniform(-np.pi, np.pi)

        while len(current_samples) < n_sample:
            samples.update(current_samples)

            xmin, ymin, xmax, ymax = line_rectangle_intersection(shape, angle)

            length = int(np.hypot(xmax - xmin, ymax - ymin))
            x = np.linspace(xmin, xmax, length).astype(np.int)
            y = np.linspace(ymin, ymax, length).astype(np.int)

            current_samples.update(zip(x, y))

            """ 'golden angles', first according to MR papers, second according to wikipedia """
            # angle += 1.9416
            angle += 2.39996322972865332

        if abs(len(samples) - n_sample) > len(current_samples) - n_sample:
            samples.update(current_samples)

        mask[tuple(zip(*samples))] = 1.

        return torch.from_numpy(mask.reshape(*shape).astype(np.float32))

    def radial3d(self, shape):
        shape = np.array((1, shape[-2], shape[-1]))
        mask = np.zeros(shape)

        c, a, b = map(lambda x: .5 * x - .05, shape)
        r_max = 2 * np.sqrt(c ** 2 + a ** 2 + b ** 2)
        n_sample = int(ft.reduce(lambda x, y: x * y, shape) / np.random.uniform(*self.accelerations))
        n_spoke = int(n_sample / (.095 * r_max))  # (.9 * r_max))
        hs = [2 * k / (n_spoke - 1) - 1 for k in range(n_spoke)]
        thetas = np.arccos(hs)
        phis = [0]

        for h in hs[1:-1]:
            phis.append(phis[-1] + 3.6 / np.sqrt(n_spoke * (1 - h ** 2)) % 2 * np.pi)
        phis.append(0)

        hs = hs[len(hs) // 2:]
        thetas = thetas[len(thetas) // 2:]
        phis = phis[len(phis) // 2:]

        zs = c * np.cos(thetas)
        xs = a * np.sin(thetas) * np.cos(phis)
        ys = b * np.sin(thetas) * np.sin(phis)
        rs = 2 * np.sqrt(zs ** 2 + xs ** 2 + ys ** 2).astype(np.int)

        zlines = np.concatenate([np.linspace(c - z, c + z, r).astype(np.int) for z, r in zip(zs, rs)][::2])
        xlines = np.concatenate([np.linspace(a - x, a + x, r).astype(np.int) for x, r in zip(xs, rs)][::2])
        ylines = np.concatenate([np.linspace(b - y, b + y, r).astype(np.int) for y, r in zip(ys, rs)][::2])

        samples = zip(zlines, xlines, ylines)

        mask[tuple(zip(*samples))] = 1.

        return torch.from_numpy(mask.reshape(*shape).astype(np.float32)).squeeze(0)

    def periodic1d(self, shape):
        shape = shape[-1]
        scale = int(np.random.uniform(*self.center_fractions) * shape)

        start = (shape - scale) // 2
        end = (shape + scale) // 2

        mask = np.zeros(shape)
        mask[start:end] = 1.
        mask[np.arange(0, shape, int(np.random.uniform(*self.accelerations)))] = 1.

        shape = [1, shape]

        return torch.from_numpy(mask.reshape(*shape).astype(np.float32))

    def periodic2d(self, shape):
        _scale = np.random.uniform(*self.center_fractions)
        scale = int(_scale * shape[0]), int(_scale * shape[1])

        start = (shape[0] - scale[0]) // 2, (shape[1] - scale[1]) // 2
        end = (shape[0] + scale[0]) // 2, (shape[1] + scale[1]) // 2

        mask = np.zeros(shape)
        mask[start[0]:end[0], start[1]:end[1]] = 1.

        acc = int(np.random.uniform(*self.accelerations))

        mask[tuple(zip(*list(it.product(np.arange(0, shape[0], acc), np.arange(0, shape[1], acc)))))] = 1.

        return torch.from_numpy(mask.reshape(*shape).astype(np.float32))

    def poisson1d(self, shape, k=10):
        shape = shape[-1]
        scale = int(shape * np.random.uniform(*self.center_fractions))

        top = (shape - scale) // 2
        btm = (shape - scale - top)

        mask = np.concatenate((np.zeros(top), np.ones(scale), np.zeros(btm)))

        accelerations = np.random.uniform(*self.accelerations)

        n_sample = math.ceil(shape / accelerations)

        def getbin1(candidate):
            return int(candidate // accelerations)

        def getbin2(candidate):
            kernel = gaussian_kernel((n_sample,), (1,))
            distance = np.array([k * shape for k in kernel])

        getbin2(3)

        background_grid = n_sample * [-1]
        activelist = [np.random.choice(shape)]

        background_grid[getbin1(activelist[0])] = activelist[0]

        n_sampled = 1

        while activelist and n_sampled < n_sample:
            centerpoint_idx = np.random.randint(len(activelist))
            centerpoint = activelist[centerpoint_idx]

            issampled = False
            for _ in range(k):
                candidate = np.random.choice([centerpoint - np.random.randint(accelerations, 2 * accelerations),
                                              centerpoint + np.random.randint(accelerations, 2 * accelerations)])

                if candidate >= 0 and candidate < shape and background_grid[getbin1(candidate)] == -1 and not \
                        any(map(lambda x: abs(x - candidate) < accelerations if x != -1 else False, \
                                background_grid[
                                max(0, getbin1(candidate) - 1):int(candidate // accelerations)] + background_grid[min(
                                    len(background_grid) - 1, \
                                        getbin1(candidate) + 1):min(len(background_grid) - 1,
                                                                    getbin1(candidate) + 2)])):
                    background_grid[getbin1(candidate)] = candidate
                    activelist.append(candidate)
                    issampled = True
                    n_sampled += 1

                    break

            if not issampled:
                del activelist[centerpoint_idx]

        samples = np.array([b for b in background_grid if b >= 0], dtype=int)

        mask[samples] = 1

        shape = [1, shape]

        return torch.from_numpy(mask.reshape(*shape).astype(np.float32))
