import sys
from pprint import pformat
from typing import Any

import colored_traceback
import hydra
import ignite.distributed as idist
import torch
import torchvision.utils as vutils
from data import setup_data
from ignite.engine import Events
from ignite.utils import manual_seed
from models import ConditionalDiscriminator, ConditionalGenerator
from omegaconf import DictConfig
from torch import nn, optim
from trainers_cDCGAN import setup_evaluator, setup_trainer
from utils import (
    log_metrics,
    save_config,
    setup_config,
    setup_exp_logging,
    setup_handlers,
    setup_logging,
    setup_output_dir,
)

colored_traceback.add_hook(always=True)

FAKE_IMG_FNAME = "fake_sample_epoch_{:04d}.png"
REAL_IMG_FNAME = "real_sample_epoch_{:04d}.png"


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def run(local_rank: int, config: Any):
    # make a certain seed
    rank = idist.get_rank()
    manual_seed(config.seed + rank)

    # create output folder and copy config file to output dir
    output_dir = setup_output_dir(config, rank)
    if rank == 0:
        save_config(config, output_dir)

    config.output_dir = output_dir

    # donwload datasets and create dataloaders
    dataloader_train, dataloader_eval, num_channels = setup_data(config)

    # model, optimizer, loss function, device
    device = idist.device()

    fixed_noise = torch.randn(
        100, # config.batch_size // idist.get_world_size(),
        config.nz,
        device=device,
    )

    fixed_labels = torch.arange(0, 10).expand(size=(10, 10)).flatten().to(device)
    fixed_y = torch.nn.functional.one_hot(fixed_labels).float().to(device)

    # networks
    
    model_g = ConditionalGenerator(nz=config.nz, nc=config.nc, ngf=config.ngf)
    model_g.apply(weights_init)
    model_g = idist.auto_model(model_g)
    model_d = ConditionalDiscriminator(ndf=config.ndf, nc=config.nc, nchannels=num_channels)
    model_d.apply(weights_init)
    model_d = idist.auto_model(model_d)

    # loss
    loss_fn = nn.BCELoss().to(device=device)

    # optimizers
    optimizer_d = idist.auto_optim(
        optim.Adam(model_d.parameters(), lr=config.lr_d, betas=(config.beta1, 0.999))
    )
    optimizer_g = idist.auto_optim(
        optim.Adam(model_g.parameters(), lr=config.lr_g, betas=(config.beta1, 0.999))
    )

    # trainer and evaluator
    trainer = setup_trainer(
        config=config,
        netG=model_g,
        netD=model_d,
        d_opt=optimizer_d,
        g_opt=optimizer_g,
        criterion=loss_fn,
        device=device,
        train_sampler=dataloader_train.sampler,
    )
    evaluator = setup_evaluator(
        config=config,
        netG=model_g,
        netD=model_d,
        criterion=loss_fn,
        device=device,
    )

    # setup engines logger with python logging
    # print training configurations
    logger = setup_logging(config)
    logger.info("Configuration: \n%s", pformat(config))
    trainer.logger = evaluator.logger = logger
    setup_handlers(trainer, evaluator, config)
    # experiment tracking
    if rank == 0:
        exp_logger = setup_exp_logging(
            config,
            trainer,
            {"optimizer_d": optimizer_d, "optimizer_g": optimizer_g},
            evaluator,
            project="AMAL-GAN",
        )

    # print metrics to the stderr
    # with `add_event_handler` API
    # for training stats
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.log_every_iters),
        log_metrics,
        tag="train",
    )

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_fake_example(engine):
        fake = model_g(fixed_noise, fixed_y)
        path = config.output_dir / FAKE_IMG_FNAME.format(engine.state.epoch)
        vutils.save_image(fake.detach(), path, normalize=True, nrow=10)
        img = exp_logger._wandb.Image(str(path), caption=FAKE_IMG_FNAME.format(engine.state.epoch))
        exp_logger.log({"example": img})

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_real_example(engine):
        img, y = engine.state.batch
        path = config.output_dir / REAL_IMG_FNAME.format(engine.state.epoch)
        vutils.save_image(img, path, normalize=True)

    # run evaluation at every training epoch end
    # with shortcut `on` decorator API and
    # print metrics to the stderr
    # again with `add_event_handler` API
    # for evaluation stats
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def _():
        evaluator.run(dataloader_eval, epoch_length=config.eval_epoch_length)
        log_metrics(evaluator, "eval")

    # let's try run evaluation first as a sanity check
    @trainer.on(Events.STARTED)
    def _():
        evaluator.run(dataloader_eval, epoch_length=config.eval_epoch_length)

    # setup if done. let's run the training
    trainer.run(
        dataloader_train,
        max_epochs=config.max_epochs,
        epoch_length=config.train_epoch_length,
    )

    # close logger
    if rank == 0:
        exp_logger.close()


# main entrypoint
@hydra.main(version_base=None, config_path="./exp_configs", config_name="cDCGAN")
def main(cfg: DictConfig):
    config = setup_config(cfg)
    with idist.Parallel(config.backend) as p:
        p.run(run, config=config)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=stdout")
    main()
