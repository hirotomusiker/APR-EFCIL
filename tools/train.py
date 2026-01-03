import argparse
import time

from omegaconf import OmegaConf

import libs.learners as learners
from libs.datasets.build import build_dataset
from libs.nets.build_net import build_net
from libs.utils.logger import build_logger
from libs.utils.logger import log_results
from libs.utils.misc import seed_everything
from libs.utils.recorder import Recorder


def arg_parser():
    parser = argparse.ArgumentParser(description="Adversarial Pseudo Replay")
    parser.add_argument("config", default="config/cifar100/apr.yaml", type=str)
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        help="Dot-dict style overrides"
        "(e.g. incremental.init_cls=50 adc.do_adc=False)",
    )
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--clsseed", default=1993, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--logfile", default="log.log", type=str)
    parser.add_argument(
        "--resultcsv", default="experiment_results.csv", type=str
    )
    return parser.parse_args()


def main():
    args = arg_parser()
    logger = build_logger(name="CL", logfilename=args.logfile)
    device = args.device
    default_cfg = OmegaConf.load("configs/default.yaml")
    child_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(default_cfg, child_cfg)
    if args.cfg_options:
        OmegaConf.set_struct(cfg, True)
        cfg.merge_with_dotlist(args.cfg_options)
    dataset = build_dataset(
        cfg,
        logger=logger,
        clsseed=args.clsseed,
    )
    start_time = time.time()
    logger.info(f"Starting experiment with seed = {args.seed}")
    logger.info(f"cfg = {cfg}")
    seed_everything(args.seed)
    net = build_net(cfg)
    learner = getattr(learners, cfg.learner.learner_type)(
        cfg, net, dataset, logger, args.ckpt, device
    )
    acc_list, forget_list, acc_matrix, custom_metrics = learner.run()
    elapsed_time = time.time() - start_time
    recorder = Recorder(
        logger, csv_path=args.resultcsv, metrics=list(acc_list.keys())
    )
    recorder.add_header(
        OmegaConf.to_container(cfg, resolve=True), args, args.seed
    )
    recorder.add_results(
        acc_list, forget_list, acc_matrix, custom_metrics, elapsed_time
    )
    recorder.save_csv()
    log_results(acc_list, forget_list, acc_matrix, elapsed_time, logger)


if __name__ == "__main__":
    main()
