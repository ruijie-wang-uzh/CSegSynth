"""
Pre-train the VAE model
"""

import os
import torch
import random
from vae_models import BaseAE
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils import read_obj, mri_collate_func, get_time, write_obj, MRIData


class Main:
    def __init__(self):
        self.subj2mri = read_obj(os.path.join(args.data_path, 'subj2mri_d{}.pickle'.format(args.down_factor)))

        train_subjs, valid_subjs = [list(_) for _ in torch.utils.data.random_split(list(self.subj2mri.keys()), args.split)]

        write_obj(obj={'train_subjs': train_subjs, 'valid_subjs': valid_subjs},
                  file_path=os.path.join(args.model_path, '{}(split).pickle'.format(args.timestamp)))

        self.train_dataset = MRIData(subjs=train_subjs, subj2mri=self.subj2mri)
        self.valid_dataset = MRIData(subjs=valid_subjs, subj2mri=self.subj2mri)

        self.check_subjs = [random.choice(self.valid_dataset.subjs) for _ in range(args.num_gen_samples)]

    def train(self):
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=mri_collate_func,
                                  shuffle=True)
        valid_loader = DataLoader(dataset=self.valid_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=mri_collate_func,
                                  shuffle=True)

        model = BaseAE(channel_base=args.channel_base)

        if args.num_devices > 1:
            model = torch.nn.DataParallel(model)

        if args.pre_timestamp:
            model_state_dict, optimizer_state_dict = read_obj(file_path=os.path.join(args.model_path, '{}(model_{}).pickle'.format(args.pre_timestamp, args.pre_epoch)))
            model.load_state_dict(model_state_dict)

        model.cuda()

        optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

        if args.pre_timestamp:
            optim.load_state_dict(optimizer_state_dict)

        loss_cri = torch.nn.BCELoss(reduction='mean')

        best_valid_loss = 9e16

        for epoch in range(args.num_epochs):
            epoch_loss = 0.
            epoch_kl_loss = 0.
            epoch_recon_loss = 0.

            num_batches = 0
            epoch_log = {}

            model.train()

            for batch, batch_data in enumerate(train_loader):
                optim.zero_grad()

                real_images = batch_data

                kl_loss, y_hat = model(real_images)
                kl_loss = torch.sum(kl_loss, dim=1).mean()

                recon_loss = loss_cri(y_hat, real_images)
                batch_loss = args.kl_weight * kl_loss + (1 - args.kl_weight) * recon_loss

                batch_loss.backward()
                torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=args.grad_clip)

                optim.step()

                epoch_loss += batch_loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_recon_loss += recon_loss.item()
                num_batches += 1

            epoch_log['train_loss'] = epoch_loss / num_batches
            epoch_log['kld loss'] = epoch_kl_loss / num_batches
            epoch_log['recon loss'] = epoch_recon_loss / num_batches

            print('* epoch {}, time: {}, loss: {}'.format(epoch, get_time(), epoch_log))

            valid_loss = 0.
            valid_kl_loss = 0.
            valid_recon_loss = 0.

            num_batches = 0

            model.eval()
            with torch.no_grad():
                for batch, batch_data in enumerate(valid_loader):
                    real_images = batch_data

                    kl_loss, y_hat = model(real_images)

                    kl_loss = torch.sum(kl_loss, dim=1).mean()

                    recon_loss = loss_cri(y_hat, real_images)

                    batch_loss = args.kl_weight * kl_loss + (1 - args.kl_weight) * recon_loss

                    valid_loss += batch_loss.item()
                    valid_kl_loss += kl_loss.item()
                    valid_recon_loss += recon_loss.item()
                    num_batches += 1

                epoch_log['valid_loss'] = valid_loss / num_batches
                epoch_log['valid kld loss'] = valid_kl_loss / num_batches
                epoch_log['valid recon loss'] = valid_recon_loss / num_batches

                print('\t* valid loss {}'.format(valid_loss / num_batches))

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss

                    tmp_real_images = torch.stack([self.valid_dataset.subj2mri[_].cuda() for _ in self.check_subjs], dim=0)

                    recon_images = model(tmp_real_images)[1].cpu()

                    fake_images = model.module.decoder(torch.randn((args.num_gen_samples, args.channel_base * 32)).cuda()).cpu()

                    write_obj(obj={'subjs': self.check_subjs, 'real': tmp_real_images.cpu(), 'recon': recon_images, 'fake': fake_images},
                              file_path=os.path.join(args.model_path, '{}(check_{}).pickle'.format(args.timestamp, epoch + args.pre_epoch)))

                    write_obj(obj=[model.state_dict(), optim.state_dict()],
                              file_path=os.path.join(args.model_path, '{}(model_{}).pickle'.format(args.timestamp, epoch + args.pre_epoch)))


if __name__ == '__main__':

    args = ArgumentParser()
    args.add_argument('--dataset', default='Aomic', type=str)

    args.add_argument('--data_path', default='datasets/{}/output', type=str)

    args.add_argument('--down_factor', default=2, type=int)

    args.add_argument('--split', type=float, default=[0.9, 0.1], nargs='+')

    args.add_argument('--batch_size', default=7, type=int)

    args.add_argument('--save_freq', default=25, type=int)

    args.add_argument('--num_gen_samples', type=int, default=3)

    args.add_argument('--num_epochs', default=5000, type=int)

    args.add_argument('--channel_base', default=32, type=int)

    args.add_argument('--pre_timestamp', default='', type=str)

    args.add_argument('--pre_epoch', default=0, type=int)

    args.add_argument('--lr', default=1e-5, type=float)

    args.add_argument('--kl_weight', default=5e-4, type=float)

    args.add_argument('--random_seed', type=int, default=0)

    args.add_argument('--model_path', default='datasets/{}/models', type=str)

    args.add_argument('--grad_clip', default=1, type=float)

    args.add_argument('--num_devices', type=int, default=torch.cuda.device_count())

    args.add_argument('--timestamp', type=str, default=get_time())

    args = args.parse_args()
    args.data_path = args.data_path.format(args.dataset)
    args.model_path = args.model_path.format(args.dataset)

    print('## VAE {} Pretraining - {}'.format(args.dataset, args.timestamp))
    for k, v in vars(args).items():
        print('* {}: {}'.format(k, v))

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    main = Main()
    main.train()
