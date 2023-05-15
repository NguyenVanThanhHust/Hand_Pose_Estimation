from .unet import PoseUnet
from .loss import bce_loss

def build_model(cfg):
    model = PoseUnet(num_class=cfg.MODEL.NUM_CLASSES)
    return model

def build_loss(cfg):
    return bce_loss