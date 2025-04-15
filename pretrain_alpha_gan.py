"""
Pre-train the alpha-GAN model
Please note that the alpha-GAN model implementation and training are partially based on: https://github.com/cyclomon/3dbraingen
"""

import os
import torch
import random
from models import Discriminator, Generator, calc_gradient_penalty, AlphaEncoder, CodeDiscriminator
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils import read_obj, MRIData, mri_collate_func, get_time, write_obj


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
        e_model = AlphaEncoder(channel_base=args.channel_base)
        cd_model = CodeDiscriminator(channel_base=args.channel_base)

        if args.num_devices > 1:
            d_model = torch.nn.DataParallel(d_model)
            g_model = torch.nn.DataParallel(g_model)
            e_model = torch.nn.DataParallel(e_model)
            cd_model = torch.nn.DataParallel(cd_model)

        if args.pre_timestamp:
            (d_model_state_dict, g_model_state_dict, e_model_state_dict,
             cd_model_state_dict, d_optimizer_state_dict, g_optimizer_state_dict,
             e_optimizer_state_dict, cd_optimizer_state_dict) = read_obj(
                file_path=os.path.join(args.model_path, '{}(model_{}).pickle'.format(args.pre_timestamp, args.pre_epoch)))

            d_model.load_state_dict(d_model_state_dict)
            g_model.load_state_dict(g_model_state_dict)
            e_model.load_state_dict(e_model_state_dict)
            cd_model.load_state_dict(cd_model_state_dict)

        d_model.cuda()
        g_model.cuda()
        e_model.cuda()
        cd_model.cuda()

        d_optimizer = torch.optim.Adam(d_model.parameters(), lr=args.d_lr)
        g_optimizer = torch.optim.Adam(g_model.parameters(), lr=args.g_lr)
        e_optimizer = torch.optim.Adam(e_model.parameters(), lr=args.e_lr)
        cd_optimizer = torch.optim.Adam(cd_model.parameters(), lr=args.cd_lr)

        if args.pre_timestamp:
            d_optimizer.load_state_dict(d_optimizer_state_dict)
            g_optimizer.load_state_dict(g_optimizer_state_dict)
            e_optimizer.load_state_dict(e_optimizer_state_dict)
            cd_optimizer.load_state_dict(cd_optimizer_state_dict)

        recon_cri = torch.nn.BCELoss(reduction='mean')

        dis_penalty = torch.linspace(0, 1, args.dis_warmup)
        warm_up_count = 0

        for epoch in range(args.num_epochs):

            train_g_fake_score = 0.
            train_g_recon_score = 0.
            train_e_score = 0.
            train_eg_loss = 0.
            train_recon_loss = 0.

            train_d_real_score = 0.
            train_d_fake_score = 0.
            train_d_recon_score = 0.
            train_fake_gp = 0.
            train_recon_gp = 0.
            train_d_loss = 0.

            train_c_real_score = 0.
            train_c_fake_score = 0.
            train_c_gp = 0.
            train_c_loss = 0.

            num_batches = 0.

            epoch_log = {}

            d_model.train()
            g_model.train()
            e_model.train()
            cd_model.train()

            for batch, batch_data in enumerate(train_loader):

                real_images = batch_data

                # train the encoder-generator
                for p in d_model.parameters():
                    p.requires_grad = False
                for p in cd_model.parameters():
                    p.requires_grad = False

                for p in e_model.parameters():
                    p.requires_grad = True
                for p in g_model.parameters():
                    p.requires_grad = True

                e_optimizer.zero_grad()
                g_optimizer.zero_grad()

                _batch_size = real_images.size(0)

                z_rand = torch.randn((_batch_size, args.channel_base * 32)).cuda()
                x_rand = g_model(z_rand)

                z_hat = e_model(real_images)
                x_hat = g_model(z_hat)

                c_loss = - cd_model(z_hat).mean()

                d_fake_score = d_model(x_rand).mean()
                d_recon_score = d_model(x_hat).mean()
                eg_loss = - d_fake_score - d_recon_score

                recon_loss = args.recon_w * recon_cri(x_hat, real_images)

                loss1 = dis_penalty[warm_up_count] * (args.cd_w * c_loss + eg_loss) + recon_loss
                if warm_up_count < args.dis_warmup - 1:
                    warm_up_count += 1

                loss1.backward()
                torch.nn.utils.clip_grad_value_(parameters=e_model.parameters(), clip_value=args.e_grad_clip)
                torch.nn.utils.clip_grad_value_(parameters=g_model.parameters(), clip_value=args.g_grad_clip)

                e_optimizer.step()
                g_optimizer.step()

                train_g_fake_score += d_fake_score.item()
                train_g_recon_score += d_recon_score.item()
                train_e_score += - c_loss.item()
                train_eg_loss += loss1.item()
                train_recon_loss += recon_loss.item()

                num_batches += 1

                if num_batches % args.num_eg_iters:

                    # train the discriminator
                    for p in d_model.parameters():
                        p.requires_grad = True

                    for p in cd_model.parameters():
                        p.requires_grad = False
                    for p in e_model.parameters():
                        p.requires_grad = False
                    for p in g_model.parameters():
                        p.requires_grad = False

                    d_optimizer.zero_grad()

                    z_rand = torch.randn((_batch_size, args.channel_base * 32)).cuda()
                    x_rand = g_model(z_rand)

                    z_hat = e_model(real_images)
                    x_hat = g_model(z_hat)

                    d_fake_score = d_model(x_rand).mean()
                    d_recon_score = d_model(x_hat).mean()
                    d_real_score = d_model(real_images).mean()

                    d_loss = d_fake_score + d_recon_score - 2 * d_real_score

                    real_images.requires_grad = True
                    gp_fake = calc_gradient_penalty(d_model, real_images, x_rand)
                    gp_recon = calc_gradient_penalty(d_model, real_images, x_hat)
                    real_images.requires_grad = False

                    loss2 = d_loss + args.gp_lambda * (gp_fake + gp_recon)

                    loss2.backward()
                    torch.nn.utils.clip_grad_value_(parameters=d_model.parameters(), clip_value=args.d_grad_clip)

                    d_optimizer.step()

                    train_d_fake_score += d_fake_score.item()
                    train_d_recon_score += d_recon_score.item()
                    train_d_real_score += d_real_score.item()
                    train_fake_gp += gp_fake.item()
                    train_recon_gp += gp_recon.item()
                    train_d_loss += loss2.item()

                    # train the code discriminator
                    for p in d_model.parameters():
                        p.requires_grad = False

                    for p in cd_model.parameters():
                        p.requires_grad = True

                    for p in e_model.parameters():
                        p.requires_grad = False
                    for p in g_model.parameters():
                        p.requires_grad = False

                    cd_optimizer.zero_grad()

                    z_rand = torch.randn((_batch_size, args.channel_base * 32)).cuda()
                    cd_real_score = cd_model(z_rand).mean()

                    z_hat = e_model(real_images)
                    cd_fake_score = cd_model(z_hat).mean()

                    z_rand.requires_grad = True
                    gp_cd = calc_gradient_penalty(cd_model, z_rand, z_hat)

                    loss3 = cd_fake_score - cd_real_score + args.gp_lambda * gp_cd

                    loss3.backward()
                    torch.nn.utils.clip_grad_value_(parameters=cd_model.parameters(), clip_value=args.cd_grad_clip)

                    cd_optimizer.step()

                    train_c_real_score += cd_real_score.item()
                    train_c_fake_score += cd_fake_score.item()
                    train_c_gp += gp_cd.item()
                    train_c_loss += loss3.item()

            epoch_log['1_gen_fake_dis_score'] = train_g_fake_score / num_batches / args.batch_size
            epoch_log['1_gen_recon_dis_score'] = train_g_recon_score / num_batches / args.batch_size

            epoch_log['1_enc_code_score'] = train_e_score / num_batches / args.batch_size
            epoch_log['1_recon_img_loss'] = train_recon_loss / num_batches / args.batch_size

            epoch_log['loss1'] = train_eg_loss / num_batches / args.batch_size

            epoch_log['2_real_score'] = train_d_real_score / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['2_randn_fake_score'] = train_d_fake_score / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['2_recon_score'] = train_d_recon_score / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['2_randn_fake_gp'] = train_fake_gp / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['2_recon_gp'] = train_recon_gp / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['loss2'] = train_d_loss / num_batches / args.batch_size * args.num_eg_iters

            epoch_log['3_real_randn_score'] = train_c_real_score / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['3_encoder_output_score'] = train_c_fake_score / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['3_real_randn-encoder_output_gp'] = train_c_gp / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['loss3'] = train_c_loss / num_batches / args.batch_size * args.num_eg_iters

            epoch_log['dis_warmup'] = dis_penalty[warm_up_count]

            print('* epoch {}, time: {}'.format(epoch, get_time()))

            if epoch % args.save_freq == 0:
                with torch.no_grad():
                    e_model.eval()
                    g_model.eval()

                    tmp_real_images = torch.stack([self.train_dataset.subj2mri[_].cuda() for _ in self.check_subjs], dim=0)

                    fake_images = g_model(torch.randn((args.num_gen_samples, args.channel_base * 32)).cuda()).cpu()

                    recon_images = g_model(e_model(tmp_real_images)).cpu()

                    write_obj(obj={'subjs': self.check_subjs, 'real': tmp_real_images.cpu(), 'fake': fake_images, 'recon': recon_images},
                              file_path=os.path.join(args.model_path, '{}(check_{}).pickle'.format(args.timestamp, epoch + args.pre_epoch)))

                write_obj(obj=[d_model.state_dict(), g_model.state_dict(), e_model.state_dict(), cd_model.state_dict(),
                               d_optimizer.state_dict(), g_optimizer.state_dict(), e_optimizer.state_dict(), cd_optimizer.state_dict()],
                          file_path=os.path.join(args.model_path, '{}(model_{}).pickle'.format(args.timestamp, epoch + args.pre_epoch)))

    def gen_imgs(self):
        g_model = Generator(channel_base=args.channel_base)
        e_model = AlphaEncoder(channel_base=args.channel_base)

        if args.num_devices > 1:
            g_model = torch.nn.DataParallel(g_model)
            e_model = torch.nn.DataParallel(e_model)

        (_, g_model_state_dict, e_model_state_dict, _, _, _, _, _) = read_obj(
            file_path=os.path.join(args.model_path,
                                   '{}(model_{}).pickle'.format(args.pre_timestamp, args.pre_epoch)))

        g_model.load_state_dict(g_model_state_dict)
        e_model.load_state_dict(e_model_state_dict)

        g_model.cuda()
        e_model.cuda()

        with torch.no_grad():
            e_model.eval()
            g_model.eval()

            tmp_real_images = torch.stack([self.train_dataset.subj2mri[_].cuda() for _ in self.check_subjs], dim=0)

            fake_images = g_model(torch.randn((args.num_gen_samples, args.channel_base * 32)).cuda()).cpu()

            recon_images = g_model(e_model(tmp_real_images)).cpu()

            write_obj(obj={'subjs': self.check_subjs, 'real': tmp_real_images.cpu(), 'fake': fake_images, 'recon': recon_images},
                      file_path=os.path.join(args.model_path, '{}(check_{}).pickle'.format(args.pre_timestamp, args.pre_epoch)))


if __name__ == '__main__':

    args = ArgumentParser()
    args.add_argument('--dataset', default='Aomic', type=str)

    args.add_argument('--data_path', default='datasets/{}/output', type=str)

    args.add_argument('--down_factor', default=2, type=int)

    args.add_argument('--batch_size', default=16, type=int)

    args.add_argument('--save_freq', default=25, type=int)

    args.add_argument('--num_gen_samples', default=3, type=int)

    args.add_argument('--num_epochs', default=3000, type=int)

    args.add_argument('--channel_base', default=32, type=int)

    args.add_argument('--pre_timestamp', default='', type=str)

    args.add_argument('--pre_epoch', default=0, type=int)

    args.add_argument('--gp_lambda', default=10, type=float)

    args.add_argument('--recon_w', default=10.0, type=float)

    args.add_argument('--cd_w', default=1., type=float)

    args.add_argument('--dis_warmup', default=1000, type=int)

    args.add_argument('--num_eg_iters', default=10, type=int)

    args.add_argument('--d_lr', default=2e-5, type=float)
    args.add_argument('--g_lr', default=2e-5, type=float)
    args.add_argument('--e_lr', default=2e-5, type=float)
    args.add_argument('--cd_lr', default=2e-5, type=float)

    args.add_argument('--random_seed', default=0, type=int)

    args.add_argument('--model_path', default='datasets/{}/models', type=str)

    args.add_argument('--d_grad_clip', default=1, type=float)
    args.add_argument('--g_grad_clip', default=1, type=float)
    args.add_argument('--e_grad_clip', default=1, type=float)
    args.add_argument('--cd_grad_clip', default=1, type=float)

    args.add_argument('--num_devices', type=int, default=torch.cuda.device_count())

    args.add_argument('--timestamp', type=str, default=get_time())

    args.add_argument('--gen_img', action='store_true')

    args = args.parse_args()
    args.data_path = args.data_path.format(args.dataset)
    args.model_path = args.model_path.format(args.dataset)

    print('## Alpha-GAN {} Pretraining - {}'.format(args.dataset, args.timestamp))
    for k, v in vars(args).items():
        print('* {}: {}'.format(k, v))

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    main = Main()
    if not args.gen_img:
        main.train()
    else:
        main.gen_imgs()
