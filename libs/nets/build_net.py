from libs.nets.cos_net import SplitCosNet


def build_net(cfg):
    if cfg.net.net_type == "SplitCosNet":
        return SplitCosNet(
            cfg.net.backbone_type,
            for_insubset=cfg.net.for_insubset,
        )
    else:
        raise NotImplementedError()
