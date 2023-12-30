from typing import Any, Union

import ignite.distributed as idist
import torch
from ignite.engine import DeterministicEngine, Engine, Events
from torch.cuda.amp import autocast
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler, Sampler


def setup_trainer(
    config: Any,
    netG: Module,
    netD: Module,
    d_opt: Optimizer,
    g_opt: Optimizer,
    criterion: Module,
    device: Union[str, torch.device],
    train_sampler: Sampler,
) -> Union[Engine, DeterministicEngine]:
    ws = idist.get_world_size()

    def get_labels_one(batch_size):
        r = torch.ones(batch_size // ws, 1)
        return r.to(device)

    def sample_z(batch_size, nz):
        return torch.randn(batch_size // ws, nz, device=device)

    # this is for the generated ground-truth label
    def get_labels_zero(batch_size):
        r = torch.zeros(batch_size // ws, 1)
        return r.to(device)

    def train_function(engine: Union[Engine, DeterministicEngine], batch: Any):
        netG.train()
        netD.train()

        im, _ = batch  # we don't care about the label for unconditional generation
        im = im.to(device)

        cur_batch_size = im.shape[0]
        # 1. sample a z vector
        x = sample_z(cur_batch_size, config.nz)
        # 2. Generate a fake image
        x = netG(x)
        # 3. Classify real image with D
        yhat_real = netD(im)
        loss_D_real = criterion(yhat_real, get_labels_one(cur_batch_size))
        # 4. Classify fake image with D
        yhat_fake = netD(x)
        loss_D_fake = criterion(yhat_fake, get_labels_zero(cur_batch_size))

        ###
        ### Discriminator
        ###
        d_loss = loss_D_real + loss_D_fake
        d_opt.zero_grad()
        d_loss.backward(
            retain_graph=True
        )  # we need to retain graph=True to be able to calculate the gradient in the g backprop
        d_opt.step()

        ###
        ### Generator
        ###
        yhat_fake_generator = netD(x)
        g_loss = criterion(yhat_fake_generator, get_labels_one(cur_batch_size))
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        # Save Metrics

        avg_real_score = yhat_real.mean().item()
        avg_fake_score = yhat_fake.mean().item()
        avg_fake_score_generator = yhat_fake_generator.mean().item()

        metrics = {
            "epoch": engine.state.epoch,
            "errD": d_loss.item(),
            "errG": g_loss.item(),
            "D_x": avg_real_score,
            "D_G_z1": avg_fake_score,
            "D_G_z2": avg_fake_score_generator,
        }
        engine.state.metrics = metrics

        return metrics

    trainer = Engine(train_function)

    # set epoch for distributed sampler
    @trainer.on(Events.EPOCH_STARTED)
    def set_epoch():
        if idist.get_world_size() > 1 and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(trainer.state.epoch - 1)

    return trainer


def setup_evaluator(
    config: Any,
    netG: Module,
    netD: Module,
    criterion: Module,
    device: Union[str, torch.device],
) -> Engine:
    ws = idist.get_world_size()

    def get_labels_one(batch_size):
        r = torch.ones(batch_size // ws, 1)
        return r.to(device)

    def sample_z(batch_size, nz):
        return torch.randn(batch_size // ws, nz, device=device)

    # this is for the generated ground-truth label
    def get_labels_zero(batch_size):
        r = torch.zeros(batch_size // ws, 1)
        return r.to(device)

    @torch.no_grad()
    def eval_function(engine: Engine, batch: Any):
        netG.eval()
        netD.eval()

        im, _ = batch  # we don't care about the label for unconditional generation
        im = im.to(device)

        cur_batch_size = im.shape[0]
        # 1. sample a z vector
        x = sample_z(cur_batch_size, config.nz)
        # 2. Generate a fake image
        x = netG(x)
        # 3. Classify real image with D
        yhat_real = netD(im)
        loss_D_real = criterion(yhat_real, get_labels_one(cur_batch_size))
        # 4. Classify fake image with D
        yhat_fake = netD(x)
        loss_D_fake = criterion(yhat_fake, get_labels_zero(cur_batch_size))

        ###
        ### Discriminator
        ###
        d_loss = loss_D_real + loss_D_fake

        ###
        ### Generator
        ###
        yhat_fake_generator = netD(x)
        g_loss = criterion(yhat_fake_generator, get_labels_one(cur_batch_size))

        # Save Metrics

        avg_real_score = yhat_real.mean().item()
        avg_fake_score = yhat_fake.mean().item()
        avg_fake_score_generator = yhat_fake_generator.mean().item()

        metrics = {
            "epoch": engine.state.epoch,
            "errD": d_loss.item(),
            "errG": g_loss.item(),
            "D_x": avg_real_score,
            "D_G_z1": avg_fake_score,
            "D_G_z2": avg_fake_score_generator,
        }
        engine.state.metrics = metrics

        return metrics

    return Engine(eval_function)
