"""
This code is based on the training code found at
https://github.com/facebookresearch/fastMRI/blob/master/models/unet/train_unet.py
"""

import logging
import os
import pathlib
import random
import shutil
import sys
import time

import numpy as np
import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from fastMRI.common.args import Args
from fastMRI.data import transforms
from models.rim import RIM, ConvRNN
from models.unet.unet_2d import UnetModel2d
from training_utils.helpers import image_loss
from training_utils.linear_mapping import signal_model_forward
from training_utils.models import RIMfastMRI
from training_utils.load.data_qMRI_loaders import create_training_sense_loaders

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.current_device()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train_epoch(args, epoch, model, train_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(train_loader)
    torch.autograd.set_detect_anomaly(True)

    memory_allocated = []
    for i, data in enumerate(train_loader):
        R2star_map_init, S0_map_init, B0_map_init, phi_map_init, R2star_map_target, S0_map_target, \
        B0_map_target, phi_map_target, y_ksp, mask_brain, sampling_mask, TEs, sensitivity_map, fname, slice = data

        TEs = TEs[0].to(args.device)

        R2star_map_init = R2star_map_init.to(args.device)
        S0_map_init = S0_map_init.to(args.device)
        B0_map_init = B0_map_init.to(args.device)
        phi_map_init = phi_map_init.to(args.device)

        R2star_map_target = R2star_map_target.to(args.device)
        S0_map_target = S0_map_target.to(args.device)
        B0_map_target = B0_map_target.to(args.device)
        phi_map_target = phi_map_target.to(args.device)

        sensitivity_map = sensitivity_map.to(args.device)
        sampling_mask = sampling_mask.to(args.device)
        mask_brain = mask_brain.to(args.device)
        y_ksp = y_ksp.to(args.device)

        optimizer.zero_grad()
        model.zero_grad()

        if args.use_rim:
            estimate = model.forward(y=torch.stack([R2star_map_init, S0_map_init, B0_map_init, phi_map_init], 1), y_ksp=y_ksp, mask_subsampling=sampling_mask, mask_brain=torch.ones_like(mask_brain), TEs=TEs, sense=sensitivity_map, metadata=[])
        else:
            args.n_steps=1
            estimate = model.forward(torch.stack([R2star_map_init, S0_map_init, B0_map_init, phi_map_init], 1))
            estimate = [estimate,]

        target = torch.stack([R2star_map_target, S0_map_target, B0_map_target, phi_map_target], 1)
        if isinstance(estimate, list):
            loss = [image_loss(e, target, mask_brain, args) for e in estimate]
            loss = sum(loss) / len(loss)
            writer.add_scalar('Loss-R2star', loss[0].item(), global_step + i)
            writer.add_scalar('Loss-S0', loss[1].item(), global_step + i)
            writer.add_scalar('Loss-B0', loss[2].item(), global_step + i)
            writer.add_scalar('Loss-phi', loss[3].item(), global_step + i)
            loss = loss.mean() /2
        else:
            loss = image_loss(estimate, target, args)

        loss.backward()
        optimizer.step()

        loss.detach_()
        avg_loss = avg_loss + loss.item() if i > 0 else loss.item()
        writer.add_scalar('Loss', loss.item(), global_step + i)

        if args.device == 'cuda':
            memory_allocated.append(torch.cuda.max_memory_allocated() * 1e-6)
            torch.cuda.reset_max_memory_allocated()

        if i % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{i:4d}/{len(train_loader):4d}] '
                f'Loss = {loss.detach().item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s '
            )
            memory_allocated = []
        start_iter = time.perf_counter()

    optimizer.zero_grad()

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    mse_losses = []
    memory_allocated = []

    start = time.perf_counter()
    for i, data in enumerate(data_loader):
        R2star_map_init, S0_map_init, B0_map_init, phi_map_init, R2star_map_target, S0_map_target, \
        B0_map_target, phi_map_target, y_ksp, mask_brain, sampling_mask, TEs, sensitivity_map, fname, slice = data

        TEs = TEs[0].to(args.device)

        R2star_map_init = R2star_map_init.to(args.device)
        S0_map_init = S0_map_init.to(args.device)
        B0_map_init = B0_map_init.to(args.device)
        phi_map_init = phi_map_init.to(args.device)
        
        R2star_map_target = R2star_map_target.to(args.device)
        S0_map_target = S0_map_target.to(args.device)
        B0_map_target = B0_map_target.to(args.device)
        phi_map_target = phi_map_target.to(args.device)
        
        sensitivity_map = sensitivity_map.to(args.device)
        sampling_mask = sampling_mask.to(args.device)
        mask_brain = mask_brain.to(args.device)
        y_ksp = y_ksp.to(args.device)
        
        if args.use_rim:
            output = model.forward(y=torch.stack([R2star_map_init, S0_map_init, B0_map_init, phi_map_init], 1), y_ksp=y_ksp, mask_subsampling=sampling_mask, mask_brain=torch.ones_like(mask_brain), TEs=TEs, sense=sensitivity_map, metadata=[])
            output_np = output[args.n_steps-1].to('cpu')
        else:
            args.n_steps=1
            output = model.forward(torch.stack([R2star_map_init, S0_map_init, B0_map_target, phi_map_target], 1))
            output_np = output.to('cpu')

        del output

        map_gt = torch.stack([R2star_map_target, S0_map_target, B0_map_target, phi_map_target], 1).to('cpu')
        mse_losses.append(image_loss(output_np, map_gt, mask_brain.to('cpu'), args).detach().numpy())

        if args.device == 'cuda':
            memory_allocated.append(torch.cuda.max_memory_allocated() * 1e-6)
            torch.cuda.reset_max_memory_allocated()

        del R2star_map_init, S0_map_init, B0_map_init, phi_map_init, R2star_map_target, S0_map_target, \
        B0_map_target, phi_map_target, mask_brain, sampling_mask, TEs, sensitivity_map, fname, slice
        torch.cuda.empty_cache()

    writer.add_scalar('Val_MSE', np.mean(mse_losses), epoch)
    writer.add_scalar('Val_memory', np.max(memory_allocated), epoch)

    return np.mean(np.mean(mse_losses)), time.perf_counter() - start, np.max(memory_allocated)


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    args.use_rim = False
    if args.use_rim:
        if args.data_type == 'memmap':
            im_channels = args.n_coils
        else:
            im_channels = 4 * 2  # four channels: R2* map, S0 map, B0 map, phi map, and gradients for each of them.

        conv_nd = 2
        rnn = ConvRNN(input_size=im_channels, recurrent_layer=args.recurrent_layer, conv_dim=conv_nd)
        model = RIM(rnn, grad_fun=signal_model_forward())
        model = RIMfastMRI(model, n_steps=args.n_steps, coil_sum_method=args.coil_sum_method)
    else:
        # use u-net
        in_channels: int = 4
        out_channels: int = 4
        num_filters: int = 64 # 16
        num_pool_layers: int = 2
        dropout_probability: float = 0.0
        model = UnetModel2d(in_channels, out_channels, num_filters, num_pool_layers, dropout_probability)
        
    return model.to(args.device)


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    if args.optimizer.upper() == 'RMSPROP':
        optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    if args.optimizer.upper() == 'ADAM':
        optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    if args.optimizer.upper() == 'SGD':
        optimizer = torch.optim.SGD(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    checkpoint_pretrained = os.path.join(args.exp_dir, 'pretrained.pt')
    if args.checkpoint is None:
        checkpoint_path = os.path.join(args.exp_dir, 'model.pt')
    else:
        checkpoint_path = args.checkpoint
    
    num_epochs_new = args.num_epochs
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint, model, optimizer = load_model(checkpoint_path)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        if os.path.exists(checkpoint_pretrained):
            _, model, optimizer = load_model(checkpoint_pretrained)
            optimizer.lr = args.lr
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    train_loader, val_loader, display_loader = create_training_sense_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, num_epochs_new):
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        mse_loss, dev_time, dev_mem = evaluate(args, epoch, model, val_loader, writer)

        writer.add_scalar('Loss-epoch', train_loss, epoch)

        is_new_best = mse_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, mse_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'VAL_MSE = {mse_loss:.4g} '
            f'TrainTime = {train_time:.4f}s ValTime = {dev_time:.4f}s ValMemory = {dev_mem:.2f}',
        )
        if args.exit_after_checkpoint:
            writer.close()
            sys.exit(0)
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--batch_size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--use_rim', action='store_true',
                        help='If set, RIM with fixed parameters')
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer to use choose between"
                                                                      "['Adam', 'SGD', 'RMSProp']")
    parser.add_argument('--loss', choices=['l1', 'mse', 'ssim'], default='mse', help='Training loss')
    parser.add_argument('--loss_subsample', type=float, default=1., help='Sampling rate for loss mask')
    parser.add_argument('--use_rss', action='store_true',
                        help='If set, will train singlecoil model with RSS targets')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--n_slices', type=int, default=1, help='Number of slices in an observation. Default=1, if'
                                                                'n_slices > 1, we will use 3d convolutions')
    parser.add_argument('--lr_step_size', type=int, default=10, help='Period of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--n_steps', type=int, default=8, help='Number of RIM steps')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report_interval', type=int, default=1, help='Period of loss reporting')
    parser.add_argument('--data_parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp_dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--multiplicity', type=int, default=1,
                        help='Number of eta estimates at every time step. The higher multiplicity, the lower the '
                             'number of necessary time steps would be expected.')
    parser.add_argument('--shared_weights', action='store_true',
                        help='If set, weights will be shared over time steps. (only relevant for IRIM)')
    parser.add_argument('--n_hidden', type=int, nargs='+', help='Number of hidden features in each layer. Can'
                                                                'be either Int or List of Ints')
    parser.add_argument('--n_network_hidden', type=int, nargs='+', help='Number of hidden features in each layer. Can'
                                                                        'be either Int or List of Ints')
    parser.add_argument('--dilations', type=int, nargs='+', help='Kernel dilations in each in each layer. Can'
                                                                 'be either Int or List of Ints')
    parser.add_argument('--depth', type=int, help='Number of RNN layers.')
    parser.add_argument('--train_resolution', type=int, nargs=2, default=None, help='Image resolution during training')
    parser.add_argument('--parametric_output', action='store_true', help='Use a parametric function for map the'
                                                                         'last layer of the iRIM to an image estimate')
    parser.add_argument('--exit_after_checkpoint', action='store_true')
    parser.add_argument('--recurrent_layer', type=str, choices=['gru', 'indrnn'], default='gru',
                        help='Type of recurrent input')
    parser.add_argument('--n_coils', type=int, default=1, help="Number of MR-images' coils.")
    parser.add_argument('--coil_sum_method', type=str, choices=['rss', 'sum'], default='rss',
                        help="Choose to sum coils over rss or torch.sum")
    parser.add_argument('--sequence', type=str, choices=['MEGRE', 'FUTURE_SEQUENCES'], default='MEGRE',
                        help="Choose for which sequence to compute the parameter maps")
    parser.add_argument('--TEs', type=tuple, default=(3.0, 11.5, 20.0, 28.5), help="Echo times (/ms) in the ME_GRE sequence.")

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.data_type == 'pickle':
        args.coil_sum_method = 'sum'
    main(args)
