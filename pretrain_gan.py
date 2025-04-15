"""
Pre-train the GAN model
"""

import os
import torch
import random
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from models import Generator, Discriminator, calc_gradient_penalty
from utils import read_obj, mri_collate_func, get_time, write_obj, MRIData


class Main:
    def __init__(self):
        self.subj2mri = read_obj(os.path.join(args.data_path, 'subj2mri_d{}.pickle'.format(args.down_factor)))

        self.train_dataset = MRIData(subjs=list(self.subj2mri.keys()), subj2mri=self.subj2mri)

        self.check_subjs = [random.choice(self.train_dataset.subjs) for _ in range(args.num_gen_samples)]

    def train(self):
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=args.batch_size,
                                  collate_fn=mri_collate_func, shuffle=True)

        d_model = Discriminator(channel_base=args.channel_base)
        g_model = Generator(channel_base=args.channel_base)

        if args.num_devices > 1:
            d_model = torch.nn.DataParallel(d_model)
            g_model = torch.nn.DataParallel(g_model)

        if args.pre_timestamp:
            (d_model_state_dict, g_model_state_dict, d_optimizer_state_dict, g_optimizer_state_dict) = read_obj(
                file_path=os.path.join(args.model_path, '{}(model_{}).pickle'.format(args.pre_timestamp, args.pre_epoch)))

            d_model.load_state_dict(d_model_state_dict)
            g_model.load_state_dict(g_model_state_dict)

        d_model.cuda()
        g_model.cuda()

        d_optimizer = torch.optim.Adam(d_model.parameters(), lr=args.lr)
        g_optimizer = torch.optim.Adam(g_model.parameters(), lr=args.lr)

        dis_penalty = torch.linspace(0, 1, args.dis_warmup)
        warm_up_count = 0

        for epoch in range(args.num_epochs):

            train_g_loss = 0.

            train_d_real_score = 0.
            train_d_fake_score = 0.
            train_d_loss = 0.
            train_fake_gp = 0.

            num_batches = 0.

            epoch_log = {}

            d_model.train()
            g_model.train()

            for batch, batch_data in enumerate(train_loader):

                real_images = batch_data

                # train the generator
                for p in d_model.parameters():
                    p.requires_grad = False

                for p in g_model.parameters():
                    p.requires_grad = True

                g_optimizer.zero_grad()

                _batch_size = real_images.size(0)

                z_rand = torch.randn((_batch_size, args.channel_base * 32)).cuda()
                x_rand = g_model(z_rand)

                d_fake_score = d_model(x_rand).mean()

                g_loss = - d_fake_score
                g_loss = dis_penalty[warm_up_count] * g_loss
                if warm_up_count < args.dis_warmup - 1:
                    warm_up_count += 1

                g_loss.backward()
                torch.nn.utils.clip_grad_value_(parameters=g_model.parameters(), clip_value=args.g_grad_clip)

                g_optimizer.step()

                train_g_loss += g_loss.item()

                num_batches += 1

                if num_batches % args.num_g_iters == 0:
                    # train the discriminator
                    for p in g_model.parameters():
                        p.requires_grad = False

                    for p in d_model.parameters():
                        p.requires_grad = True

                    d_optimizer.zero_grad()

                    z_rand = torch.randn((_batch_size, args.channel_base * 32)).cuda()
                    x_rand = g_model(z_rand)

                    d_fake_score = d_model(x_rand).mean()
                    d_real_score = d_model(real_images).mean()

                    real_images.requires_grad = True
                    gp_fake = calc_gradient_penalty(d_model, real_images, x_rand)
                    real_images.requires_grad = False

                    d_loss = d_fake_score - d_real_score + args.gp_lambda * gp_fake

                    d_loss.backward()
                    torch.nn.utils.clip_grad_value_(parameters=d_model.parameters(), clip_value=args.d_grad_clip)

                    d_optimizer.step()

                    train_d_fake_score += d_fake_score.item()
                    train_d_real_score += d_real_score.item()
                    train_fake_gp += gp_fake.item()
                    train_d_loss += d_loss.item()

            epoch_log['g_loss'] = train_g_loss / num_batches / args.batch_size

            epoch_log['d_fake_score'] = train_d_fake_score / num_batches / args.batch_size * args.num_g_iters
            epoch_log['d_real_score'] = train_d_real_score / num_batches / args.batch_size * args.num_g_iters
            epoch_log['gp_loss'] = train_fake_gp / num_batches / args.batch_size * args.num_g_iters
            epoch_log['d_loss'] = train_d_loss / num_batches / args.batch_size * args.num_g_iters

            epoch_log['dis_warmup'] = dis_penalty[warm_up_count]

            print('* epoch {}, time: {}'.format(epoch, get_time()))

            if epoch % args.save_freq == 0:
                with torch.no_grad():
                    g_model.eval()

                    tmp_real_images = torch.stack([self.train_dataset.subj2mri[_].cuda() for _ in self.check_subjs], dim=0)

                    fake_images = g_model(torch.randn((args.num_gen_samples, args.channel_base * 32)).cuda()).cpu()

                    write_obj(obj={'subjs': self.check_subjs, 'real': tmp_real_images.cpu(), 'fake': fake_images},
                              file_path=os.path.join(args.model_path, '{}(check_{}).pickle'.format(args.timestamp, epoch + args.pre_epoch)))

                write_obj(obj=[d_model.state_dict(), g_model.state_dict(), d_optimizer.state_dict(), g_optimizer.state_dict()],
                          file_path=os.path.join(args.model_path, '{}(model_{}).pickle'.format(args.timestamp, epoch + args.pre_epoch)))


if __name__ == '__main__':

    args = ArgumentParser()
    args.add_argument('--dataset', default='Aomic', type=str)

    args.add_argument('--data_path', default='datasets/{}/output', type=str)

    args.add_argument('--down_factor', default=2, type=int)

    args.add_argument('--batch_size', default=32, type=int)

    args.add_argument('--save_freq', default=50, type=int)

    args.add_argument('--num_gen_samples', type=int, default=3)

    args.add_argument('--num_epochs', default=5000, type=int)

    args.add_argument('--channel_base', default=32, type=int)

    args.add_argument('--pre_timestamp', default='', type=str)

    args.add_argument('--pre_epoch', default=0, type=int)

    args.add_argument('--gp_lambda', default=10, type=float)

    args.add_argument('--dis_warmup', default=200, type=int)

    args.add_argument('--num_g_iters', default=4, type=int)

    args.add_argument('--lr', default=2e-6, type=float)

    args.add_argument('--random_seed', type=int, default=0)

    args.add_argument('--model_path', default='datasets/{}/models', type=str)

    args.add_argument('--d_grad_clip', default=1, type=float)

    args.add_argument('--g_grad_clip', default=1, type=float)

    args.add_argument('--num_devices', type=int, default=torch.cuda.device_count())

    args.add_argument('--timestamp', type=str, default=get_time())

    args = args.parse_args()
    args.data_path = args.data_path.format(args.dataset)
    args.model_path = args.model_path.format(args.dataset)

    print('## GAN {} Pretraining - {}'.format(args.dataset, args.timestamp))
    for k, v in vars(args).items():
        print('* {}: {}'.format(k, v))

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    main = Main()
    main.train()
