"""
Fine-tune the CVAE model
"""

import os
import torch
import random
import argparse
import torch.distributions as dists
from torch.utils.data import DataLoader
from utils import get_time, read_obj, SubjData, collate_func, mean_std_estimate, remove_nan, write_obj, plot_corrs
from vae_models import BaseEncoder, Prior, Generator
from models import ConCodeTrans


class Main:
    def __init__(self):

        self.subj2feas = read_obj(os.path.join(args.data_path, 'subj2feas.pickle'))
        self.subj2vols = read_obj(os.path.join(args.data_path, 'subj2vols.pickle'))
        self.subj2mri = read_obj(os.path.join(args.data_path, 'subj2mri_d{}.pickle'.format(args.down_factor)))

        all_subjs = list(set.intersection(set(self.subj2mri.keys()), set(self.subj2feas.keys())))

        split_file = os.path.join(args.data_path, 'split.pickle')
        if os.path.exists(split_file):
            train_subjs, valid_subjs, test_subjs = read_obj(file_path=split_file)
        else:
            assert sum(args.split) == 1.
            train_subjs, valid_subjs, test_subjs = [list(_) for _ in torch.utils.data.random_split(all_subjs, args.split)]
            write_obj(obj=[train_subjs, valid_subjs, test_subjs], file_path=split_file)

        assert sorted(train_subjs + valid_subjs + test_subjs) == sorted(all_subjs)
        for _ in test_subjs:
            assert _ not in train_subjs + valid_subjs
        print('* number of subjects: {} (train {}, valid {}, test {})'.format(len(all_subjs), len(train_subjs), len(valid_subjs), len(test_subjs)))

        self.check_subjs = test_subjs[:args.num_gen_samples]
        print('* subjects to check: {}'.format(self.check_subjs))

        self.fea2mean_std = mean_std_estimate(subj2feas=self.subj2feas, subjs=train_subjs)
        remove_nan(subj2feas=self.subj2feas, fea2mean_std=self.fea2mean_std)
        self.num_feas = len(self.fea2mean_std)
        print('* number of features: {}'.format(self.num_feas))

        self.train_dataset = SubjData(dataset=args.dataset,
                                      subjs=train_subjs,
                                      subj2feas=self.subj2feas,
                                      fea2mean_std=self.fea2mean_std,
                                      subj2vols=self.subj2vols,
                                      subj2mri=self.subj2mri,
                                      down_factor=args.down_factor)

        self.valid_dataset = SubjData(dataset=args.dataset,
                                      subjs=valid_subjs,
                                      subj2feas=self.subj2feas,
                                      fea2mean_std=self.fea2mean_std,
                                      subj2vols=self.subj2vols,
                                      subj2mri=self.subj2mri,
                                      down_factor=args.down_factor)

        self.test_dataset = SubjData(dataset=args.dataset,
                                     subjs=test_subjs,
                                     subj2feas=self.subj2feas,
                                     fea2mean_std=self.fea2mean_std,
                                     subj2vols=self.subj2vols,
                                     subj2mri=self.subj2mri,
                                     down_factor=args.down_factor)

        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=args.batch_size,
                                       collate_fn=collate_func,
                                       shuffle=True)
        self.valid_loader = DataLoader(dataset=self.valid_dataset,
                                       batch_size=args.batch_size,
                                       collate_fn=collate_func,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=args.batch_size,
                                      collate_fn=collate_func,
                                      shuffle=False)

    def train(self):
        encoder = BaseEncoder(channel_base=args.channel_base)
        enc_con = ConCodeTrans(hid_dim=args.channel_base * 32, num_fea=self.num_feas)
        prior_estimate = Prior(num_fea=self.num_feas, channel_base=args.channel_base)
        dec_con = ConCodeTrans(hid_dim=args.channel_base * 32, num_fea=self.num_feas)
        decoder = Generator(channel_base=args.channel_base)

        if args.num_devices > 1:
            encoder = torch.nn.DataParallel(encoder)
            enc_con = torch.nn.DataParallel(enc_con)
            prior_estimate = torch.nn.DataParallel(prior_estimate)
            dec_con = torch.nn.DataParallel(dec_con)
            decoder = torch.nn.DataParallel(decoder)

        if args.pre_timestamp:
            pre_model_dict, _ = read_obj(file_path=os.path.join(args.pre_model_path, '{}(model_{}).pickle'.format(args.pre_timestamp, args.pre_epoch)))
            encoder_dict = {k.replace('encoder.', ''): v for k, v in pre_model_dict.items() if 'encoder' in k}
            decoder_dict = {k.replace('decoder.', ''): v for k, v in pre_model_dict.items() if 'decoder' in k}

            encoder.load_state_dict(encoder_dict)
            decoder.load_state_dict(decoder_dict)

        encoder.cuda()
        enc_con.cuda()
        prior_estimate.cuda()
        dec_con.cuda()
        decoder.cuda()

        e_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.base_lr)
        e_con_optimizer = torch.optim.Adam(enc_con.parameters(), lr=args.con_lr)
        prior_optimizer = torch.optim.Adam(prior_estimate.parameters(), lr=args.con_lr)
        d_con_optimizer = torch.optim.Adam(dec_con.parameters(), lr=args.con_lr)
        d_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.base_lr)

        bce_cri = torch.nn.BCELoss(reduction='mean')
        vol_cri = torch.nn.L1Loss(reduction='mean')

        dis_penalty = torch.linspace(0, 1, args.dis_warmup)
        warm_up_count = -1

        max_corr = -999.
        for epoch in range(args.num_epochs):
            print('* epoch {}, time: {}'.format(epoch, get_time()))

            epoch_loss = 0.
            train_kld = 0.
            train_recon = 0.
            train_vol = 0.

            encoder.train()
            enc_con.train()
            prior_estimate.train()
            dec_con.train()
            decoder.train()

            epoch_log = {}

            for batch, batch_data in enumerate(self.train_loader):

                if warm_up_count < args.dis_warmup - 1:
                    warm_up_count += 1

                feas, vols, real_images = batch_data

                e_optimizer.zero_grad()
                e_con_optimizer.zero_grad()
                prior_optimizer.zero_grad()
                d_con_optimizer.zero_grad()
                d_optimizer.zero_grad()

                prior_z_loc, prior_z_std = prior_estimate(fea=feas)
                prior_z_dist = dists.normal.Normal(prior_z_loc, prior_z_std)

                enc_z = encoder.module.con_forward(real_images)
                enc_z = enc_con(z=enc_z, fea=feas)
                enc_z_loc, enc_z_std = encoder.module.con_dist(enc_z)
                enc_z_dist = dists.normal.Normal(enc_z_loc, enc_z_std)

                z_sample = enc_z_dist.rsample()

                dec_z = dec_con(z=z_sample, fea=feas)
                y_recon = decoder(z=dec_z)

                vol_recon = torch.sum(y_recon, dim=[2, 3, 4])[:, 0:3] * args.down_factor ** 3  # size: (batch_size, 3)
                vol_loss = vol_cri(vol_recon, vols[:, 0:3])

                kld_loss = dists.kl.kl_divergence(enc_z_dist, prior_z_dist).mean()

                bce_loss = bce_cri(y_recon.cuda(), real_images)

                batch_loss = args.kld_weight * kld_loss + args.bce_weight * bce_loss + args.vol_loss_w * vol_loss
                batch_loss = dis_penalty[warm_up_count] * batch_loss

                batch_loss.backward()
                torch.nn.utils.clip_grad_value_(parameters=encoder.parameters(), clip_value=args.gradient_clip_val)
                torch.nn.utils.clip_grad_value_(parameters=enc_con.parameters(), clip_value=args.gradient_clip_val)
                torch.nn.utils.clip_grad_value_(parameters=prior_estimate.parameters(), clip_value=args.gradient_clip_val)
                torch.nn.utils.clip_grad_value_(parameters=dec_con.parameters(), clip_value=args.gradient_clip_val)
                torch.nn.utils.clip_grad_value_(parameters=decoder.parameters(), clip_value=args.gradient_clip_val)

                e_optimizer.step()
                e_con_optimizer.step()
                prior_optimizer.step()
                d_con_optimizer.step()
                d_optimizer.step()

                epoch_loss += batch_loss.item()
                train_kld += kld_loss.item()
                train_recon += bce_loss.item()
                train_vol += vol_loss.item()

            epoch_log['train_kld'] = train_kld
            epoch_log['train_recon'] = train_recon
            epoch_log['train_vol'] = train_vol
            epoch_log['epoch_loss'] = epoch_loss

            epoch_log['dis_warmup'] = dis_penalty[warm_up_count]

            avg_corr, valid_loss, valid_kld, valid_recon, valid_vol, [wm_vols, gm_vols, csf_vols, bg_vols, roi_cors] = self.eval(
                'valid', encoder, enc_con, prior_estimate, dec_con, decoder, epoch_log)

            epoch_log['valid_loss'] = valid_loss
            epoch_log['valid_kld'] = valid_kld
            epoch_log['valid_recon'] = valid_recon
            epoch_log['valid_vol'] = valid_vol

            if max_corr < avg_corr or epoch == 0:
                max_corr = avg_corr

                self.save_model_img(name='best', encoder=encoder, enc_con=enc_con, prior_estimate=prior_estimate, dec_con=dec_con, decoder=decoder,
                                    e_optimizer=e_optimizer, e_con_optimizer=e_con_optimizer, prior_optimizer=prior_optimizer, d_con_optimizer=d_con_optimizer, d_optimizer=d_optimizer)

            test_roi2cors = self.test(args.timestamp)

            epoch_log['wm_coff_test'] = test_roi2cors['wm']
            epoch_log['gm_coff_test'] = test_roi2cors['gm']
            epoch_log['csf_coff_test'] = test_roi2cors['csf']
            epoch_log['bg_coff_test'] = test_roi2cors['bg']

    def save_model_img(self, name, encoder, enc_con, prior_estimate, dec_con, decoder,
                       e_optimizer, e_con_optimizer, prior_optimizer, d_con_optimizer, d_optimizer):
        encoder.eval()
        enc_con.eval()
        prior_estimate.eval()
        dec_con.eval()
        decoder.eval()

        with torch.no_grad():
            real_imgs = torch.stack([self.test_dataset.subj2mri[subj] for subj in self.check_subjs], dim=0).cuda()
            real_feas = torch.stack([self.test_dataset.subj2feas[subj] for subj in self.check_subjs], dim=0).cuda()

            prior_z_loc, prior_z_std = prior_estimate(fea=real_feas)
            prior_z_dist = dists.normal.Normal(prior_z_loc, prior_z_std)

            z_sample = prior_z_dist.rsample()
            dec_z = dec_con(z=z_sample, fea=real_feas)
            y_randn = decoder(z=dec_z).cpu()

            enc_z = encoder.module.con_forward(real_imgs)
            enc_z = enc_con(z=enc_z, fea=real_feas)
            enc_z_loc, enc_z_std = encoder.module.con_dist(enc_z)
            enc_z_dist = dists.normal.Normal(enc_z_loc, enc_z_std)

            z_sample = enc_z_dist.rsample()
            dec_z = dec_con(z=z_sample, fea=real_feas)
            y_recon = decoder(z=dec_z).cpu()

            write_obj(obj={'subjs': self.check_subjs, 'real': real_imgs.cpu(), 'recon': y_recon, 'fake': y_randn},
                      file_path=os.path.join(args.model_path, '{}(check_{}).pickle'.format(args.timestamp, name)))

            write_obj(obj=[encoder.state_dict(), enc_con.state_dict(), prior_estimate.state_dict(), dec_con.state_dict(),
                           decoder.state_dict(), e_optimizer.state_dict(), e_con_optimizer.state_dict(), prior_optimizer.state_dict(),
                           d_con_optimizer.state_dict(), d_optimizer.state_dict()],
                      file_path=os.path.join(args.model_path, '{}(model_{}).pickle'.format(args.timestamp, name)))

    def test(self, timestamp):
        encoder = BaseEncoder(channel_base=args.channel_base)
        enc_con = ConCodeTrans(hid_dim=args.channel_base * 32, num_fea=self.num_feas)
        prior_estimate = Prior(num_fea=self.num_feas, channel_base=args.channel_base)
        dec_con = ConCodeTrans(hid_dim=args.channel_base * 32, num_fea=self.num_feas)
        decoder = Generator(channel_base=args.channel_base)

        if args.num_devices > 1:
            encoder = torch.nn.DataParallel(encoder)
            enc_con = torch.nn.DataParallel(enc_con)
            prior_estimate = torch.nn.DataParallel(prior_estimate)
            dec_con = torch.nn.DataParallel(dec_con)
            decoder = torch.nn.DataParallel(decoder)

        (encoder_state_dict, enc_con_state_dict, prior_state_dict, dec_con_state_dict, decoder_state_dict) = read_obj(
            file_path=os.path.join(args.model_path, '{}(model_{}).pickle'.format(timestamp, 'best')))[:5]
        encoder.load_state_dict(encoder_state_dict)
        enc_con.load_state_dict(enc_con_state_dict)
        prior_estimate.load_state_dict(prior_state_dict)
        dec_con.load_state_dict(dec_con_state_dict)
        decoder.load_state_dict(decoder_state_dict)

        encoder.cuda()
        enc_con.cuda()
        prior_estimate.cuda()
        dec_con.cuda()
        decoder.cuda()

        avg_corr, test_loss, test_kld, test_recon, test_vol, [wm_vols, gm_vols, csf_vols, bg_vols, roi_cors] = (
            self.eval(mode='test', encoder=encoder, enc_con=enc_con, prior_estimate=prior_estimate, dec_con=dec_con, decoder=decoder, epoch_log={}))

        print('\t* test volume correlation {}'.format(roi_cors))

        write_obj(obj={'wm': wm_vols, 'gm': gm_vols, 'csf': csf_vols, 'bg': bg_vols},
                  file_path=os.path.join(args.model_path, '{}({}).pickle'.format(args.timestamp, 'test_vols')))

        plot_corrs(wm_vols, gm_vols, csf_vols, roi_cors,
                   os.path.join(args.model_path, '{}(corr_{}).pdf'.format(args.timestamp, 'test')))

        return roi_cors

    def eval(self, mode, encoder, enc_con, prior_estimate, dec_con, decoder, epoch_log):
        assert mode in ['valid', 'test'], 'invalid mode'
        if mode == 'valid':
            eval_loader = self.valid_loader
        else:
            eval_loader = self.test_loader

        encoder.eval()
        enc_con.eval()
        prior_estimate.eval()
        dec_con.eval()
        decoder.eval()

        eval_loss = 0
        eval_kld = 0
        eval_recon = 0
        eval_vol = 0

        bce_cri = torch.nn.BCELoss(reduction='mean')
        vol_cri = torch.nn.L1Loss(reduction='mean')

        wm_vols, gm_vols, csf_vols, bg_vols = [[], []], [[], []], [[], []], [[], []]
        real_imgs, gen_imgs = [], []

        with torch.no_grad():
            for batch, batch_data in enumerate(eval_loader):
                feas, vols, real_images = batch_data

                prior_z_loc, prior_z_std = prior_estimate(fea=feas)
                prior_z_dist = dists.normal.Normal(prior_z_loc, prior_z_std)

                z_sample = prior_z_dist.rsample()
                dec_z = dec_con(z=z_sample, fea=feas)
                y_randn = decoder(z=dec_z)
                y_randn_vol = torch.sum(input=y_randn, dim=[2, 3, 4]) * args.down_factor ** 3  # size: (batch_size, 4)
                y_randn_vol = torch.nan_to_num(y_randn_vol, nan=0.0, posinf=1e5, neginf=1e5)
                for _ in range(y_randn_vol.size(0)):
                    wm_vols[0].append(vols[_][0])
                    wm_vols[1].append(y_randn_vol[_][0])
                    gm_vols[0].append(vols[_][1])
                    gm_vols[1].append(y_randn_vol[_][1])
                    csf_vols[0].append(vols[_][2])
                    csf_vols[1].append(y_randn_vol[_][2])
                    bg_vols[0].append(vols[_][3])
                    bg_vols[1].append(y_randn_vol[_][3])

                enc_z = encoder.module.con_forward(real_images)
                enc_z = enc_con(z=enc_z, fea=feas)
                enc_z_loc, enc_z_std = encoder.module.con_dist(enc_z)
                enc_z_dist = dists.normal.Normal(enc_z_loc, enc_z_std)

                z_sample = enc_z_dist.rsample()
                dec_z = dec_con(z=z_sample, fea=feas)
                y_recon = decoder(z=dec_z)

                vol_recon = torch.sum(y_recon, dim=[2, 3, 4])[:, 0:3] * args.down_factor ** 3  # size: (batch_size, 3)
                vol_loss = vol_cri(vol_recon, vols[:, 0:3])

                kld_loss = dists.kl.kl_divergence(enc_z_dist, prior_z_dist).mean()

                bce_loss = bce_cri(y_recon.cuda(), real_images)

                batch_loss = args.kld_weight * kld_loss + args.bce_weight * bce_loss + args.vol_loss_w * vol_loss

                eval_loss += batch_loss.item()
                eval_kld += kld_loss.item()
                eval_recon += bce_loss.item()
                eval_vol += vol_loss.item()

                real_imgs.append(real_images.cpu())
                gen_imgs.append(y_randn.cpu())

            real_imgs, gen_imgs = torch.cat(real_imgs, dim=0), torch.cat(gen_imgs, dim=0)
            print('\t* {}/{} test images generated'.format(gen_imgs.size(0), real_imgs.size(0)))
            write_obj(obj={'real': real_imgs, 'fake': gen_imgs},
                      file_path=os.path.join(args.model_path, '{}(test_imgs).pickle'.format(args.timestamp)))

            wm_vols, gm_vols, csf_vols, bg_vols = (torch.FloatTensor(wm_vols), torch.FloatTensor(gm_vols),
                                                   torch.FloatTensor(csf_vols), torch.FloatTensor(bg_vols))

            roi_cors = {'wm': torch.corrcoef(wm_vols)[0, 1].item(),
                        'gm': torch.corrcoef(gm_vols)[0, 1].item(),
                        'csf': torch.corrcoef(csf_vols)[0, 1].item(),
                        'bg': torch.corrcoef(bg_vols)[0, 1].item()}

            epoch_log['wm_coff'] = roi_cors['wm']
            epoch_log['gm_coff'] = roi_cors['gm']
            epoch_log['csf_coff'] = roi_cors['csf']
            epoch_log['bg_coff'] = roi_cors['bg']

        return (roi_cors['wm'] + roi_cors['gm'] + roi_cors['csf']) / 3., eval_loss, eval_kld, eval_recon, eval_vol, [wm_vols, gm_vols, csf_vols, bg_vols, roi_cors]


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    args.add_argument('--dataset', type=str, default='CamCan')

    args.add_argument('--data_path', default='datasets/{}/output', type=str)

    args.add_argument('--down_factor', default=2, type=int)

    args.add_argument('--split', type=float, default=[0.8, 0.1, 0.1], nargs='+')

    args.add_argument('--batch_size', default=16, type=int)

    args.add_argument('--kld_weight', default=5e-4, type=float)

    args.add_argument('--bce_weight', default=1., type=float)

    args.add_argument('--vol_loss_w', default=100, type=float)

    args.add_argument('--num_gen_samples', default=3, type=int)

    args.add_argument('--num_epochs', default=2000, type=int)

    args.add_argument('--channel_base', default=32, type=int)

    args.add_argument('--pre_timestamp', default='', type=str)

    args.add_argument('--pre_epoch', default=0, type=int)

    args.add_argument('--pre_model_path', default='datasets/Aomic/models', type=str)

    args.add_argument('--continue_timestamp', default='', type=str)

    args.add_argument('--dis_warmup', default=200, type=int)

    args.add_argument('--base_lr', default=1e-6, type=float)

    args.add_argument('--con_lr', default=1e-4, type=float)

    args.add_argument('--random_seed', default=0, type=int)

    args.add_argument('--model_path', default='datasets/{}/models', type=str)

    args.add_argument('--num_devices', type=int, default=torch.cuda.device_count())

    args.add_argument('--timestamp', type=str, default=get_time())

    args.add_argument('--is_eval', action='store_true')

    args.add_argument('--gradient_clip_val', default=1., type=float)

    args = args.parse_args()

    args.data_path = args.data_path.format(args.dataset)
    args.model_path = args.model_path.format(args.dataset)

    print('## CVAE {} Finetuning - {}'.format(args.dataset, args.timestamp))
    for k, v in vars(args).items():
        print('* {}: {}'.format(k, v))

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    main = Main()
    if not args.is_eval:
        main.train()
        main.test(args.timestamp)
    else:
        args.timestamp = args.continue_timestamp
        main.test(args.timestamp)

