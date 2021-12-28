import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discriminator_loss(logits_real, logits_fake):
    loss_real = bce_loss(logits_real,torch.ones_like(logits_real, dtype=torch.float))
    loss_fake = bce_loss(logits_fake,torch.zeros_like(logits_fake, dtype=torch.float))
    loss = loss_real + loss_fake
    return loss

def generator_loss(logits_fake):
    loss_fake = bce_loss(logits_fake, torch.ones_like(logits_fake, dtype=torch.float))
    return loss_fake


def ls_discriminator_loss(scores_real, scores_fake):
    loss = None
    loss_real = torch.mean(0.5 * ((scores_real - torch.ones_like(scores_real)) ** 2))
    loss_fake = torch.mean(0.5 * (scores_fake ** 2))
    loss = loss_real + loss_fake
    return loss

def ls_generator_loss(scores_fake):
    loss_fake = None
    loss_fake = torch.mean(0.5 * ((scores_fake - torch.ones_like(scores_fake)) ** 2))
    return loss_fake
