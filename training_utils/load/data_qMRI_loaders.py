from torch.utils.data import DataLoader

from utils.subsample import Masker
from training_utils.load.data_qMRI_transformers import Transform
from training_utils.load.data_qMRI import SliceData
from pdb import set_trace as bp


def create_training_sense_datasets(args):
    train_mask = Masker("gaussian2d", args.center_fractions, args.accelerations, (.7, .7))
    val_mask = Masker("gaussian2d", args.center_fractions, args.accelerations, (.7, .7))

    train_data = SliceData(
        root=args.data_path/ f'training_others/',
        transform=Transform(train_mask, args.accelerations[0], args.sequence, args.TEs, args.resolution, use_seed=True),
        sample_rate=args.sample_rate,
        sequence=args.sequence,
        TEs=args.TEs,
        n_slices=args.n_slices,
        use_rss=args.use_rss,
        use_seed=True
    )

    val_data = SliceData(
        root=args.data_path / f'validation/',
        transform=Transform(train_mask, args.accelerations[0], args.sequence, args.TEs, args.resolution, use_seed=True),
        sample_rate=1.,
        sequence=args.sequence,
        TEs=args.TEs,
        # n_slices=-1  # args.n_slices
        n_slices= args.n_slices,
        use_seed=True
        )

    return val_data, train_data


def create_training_sense_loaders(args):
    val_data, train_data = create_training_sense_datasets(args)
    display_limit = 2 if len(val_data) > 1 else 1
    display_data=[]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size= args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    display_loader = DataLoader(
        dataset=display_data,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, display_loader


def create_testing_sense_loaders(args):
    mask_func = Masker("gaussian2d", args.center_fractions, args.accelerations, (.7, .7))

    data = SliceData(
        root=args.data_path / f'testing_all',
        transform=Transform(mask_func, args.accelerations[0], args.sequence, args.TEs, args.resolution, use_seed=True),
        sample_rate=1.,
        sequence=args.sequence,
        TEs=args.TEs,
        n_slices=1,  # args.n_slices
        use_seed=True
    )

    data_loader = DataLoader(
        dataset=data,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return data_loader
