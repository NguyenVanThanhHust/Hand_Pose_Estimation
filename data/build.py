from torch.utils import data
from .datasets import CMU_Hand_Dataset
from .transforms import build_transforms

def build_datasets(cfg, data_folder, transform, split="train"):
    assert cfg.INPUT.TYPE in ["synthetic", "manual"]
    datasets = CMU_Hand_Dataset(data_folder=data_folder, data_type=cfg.INPUT.TYPE, split=split, transform=transform)
    return datasets

def make_data_loader(cfg, split="train"):
    if split=="train":
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False
    transform = build_transforms(cfg, split=split)
    datasets = build_datasets(cfg, cfg.INPUT.FOLDER, transform, split=split)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader