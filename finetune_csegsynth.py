"""
Finetune the CSegSynth model
"""

import os
import torch
import random
from models import Discriminator, Generator, calc_gradient_penalty, AlphaEncoder, CodeDiscriminator, ConCodeTrans
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils import read_obj, SubjData, mean_std_estimate, collate_func, get_time, write_obj, remove_nan, roi_mean_std_estimate, plot_corrs


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
        print('* number of subjects: {} (train {}, valid {}, test {})'.format(len(all_subjs), len(train_subjs),
                                                                              len(valid_subjs), len(test_subjs)))

        self.check_subjs = test_subjs[:args.num_gen_samples]
        print('* subjects to check: {}'.format(self.check_subjs))

        self.fea2mean_std = mean_std_estimate(subj2feas=self.subj2feas, subjs=train_subjs)
        remove_nan(subj2feas=self.subj2feas, fea2mean_std=self.fea2mean_std)
        print('* number of features: {}'.format(len(self.fea2mean_std)))
        self.roi2mean_std = roi_mean_std_estimate(subj2vols=self.subj2vols, subjs=train_subjs)
        self.margin_loss_margin = args.margin_weight * sum([self.roi2mean_std[_]['std'] for _ in ['wm', 'gm', 'csf']])
        print('* triplet margin: {}'.format(self.margin_loss_margin))

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

    def train(self):
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=collate_func,
                                  shuffle=True)
        valid_loader = DataLoader(dataset=self.valid_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=collate_func,
                                  shuffle=True)

        e_model = AlphaEncoder(channel_base=args.channel_base)
        cd_model = CodeDiscriminator(channel_base=args.channel_base)
        c_con_trans = ConCodeTrans(hid_dim=args.channel_base * 32, num_fea=len(self.fea2mean_std))
        g_model = Generator(channel_base=args.channel_base)
        d_model = Discriminator(channel_base=args.channel_base)

        if args.num_devices > 1:
            d_model = torch.nn.DataParallel(d_model)
            g_model = torch.nn.DataParallel(g_model)
            e_model = torch.nn.DataParallel(e_model)
            cd_model = torch.nn.DataParallel(cd_model)
            c_con_trans = torch.nn.DataParallel(c_con_trans)

        if args.continue_timestamp:
            (d_model_state_dict, g_model_state_dict, e_model_state_dict, cd_model_state_dict,
             c_con_trans_state_dict, d_optimizer_state_dict, g_optimizer_state_dict, e_optimizer_state_dict,
             cd_optimizer_state_dict, c_con_trans_optimizer_state_dict) = read_obj(
                file_path=os.path.join(args.model_path, '{}(model_best).pickle'.format(args.continue_timestamp)))
            d_model.load_state_dict(d_model_state_dict)
            g_model.load_state_dict(g_model_state_dict)
            e_model.load_state_dict(e_model_state_dict)
            cd_model.load_state_dict(cd_model_state_dict)
            c_con_trans.load_state_dict(c_con_trans_state_dict)
        else:
            if args.pre_timestamp:
                (d_model_state_dict, g_model_state_dict, e_model_state_dict,
                 cd_model_state_dict, d_optimizer_state_dict, g_optimizer_state_dict,
                 e_optimizer_state_dict, cd_optimizer_state_dict) = read_obj(
                    file_path=os.path.join(args.pre_model_path, '{}(model_{}).pickle'.format(args.pre_timestamp,
                                                                                             args.pre_epoch)))
                d_model.load_state_dict(d_model_state_dict)
                g_model.load_state_dict(g_model_state_dict)
                e_model.load_state_dict(e_model_state_dict)
                cd_model.load_state_dict(cd_model_state_dict)

        d_model.cuda()
        g_model.cuda()
        e_model.cuda()
        cd_model.cuda()
        c_con_trans.cuda()

        d_optimizer = torch.optim.Adam(d_model.parameters(), lr=args.d_lr)

        g_optimizer = torch.optim.Adam(g_model.parameters(), lr=args.g_lr)

        e_optimizer = torch.optim.Adam(e_model.parameters(), lr=args.e_lr)

        cd_optimizer = torch.optim.Adam(cd_model.parameters(), lr=args.cd_lr)

        c_con_trans_optimizer = torch.optim.Adam(c_con_trans.parameters(), lr=args.c_con_lr)

        if args.continue_timestamp:
            d_optimizer.load_state_dict(d_optimizer_state_dict)
            g_optimizer.load_state_dict(g_optimizer_state_dict)
            e_optimizer.load_state_dict(e_optimizer_state_dict)
            cd_optimizer.load_state_dict(cd_optimizer_state_dict)
            c_con_trans_optimizer.load_state_dict(c_con_trans_optimizer_state_dict)

        recon_cri = torch.nn.BCELoss(reduction='mean')
        classify_cri = torch.nn.L1Loss(reduction='mean')

        triplet_cri = torch.nn.TripletMarginLoss(margin=self.margin_loss_margin, p=1, reduction='mean')
        dis_penalty = torch.linspace(0, 1, args.dis_warmup)
        warm_up_count = -1

        max_avg_cor = -1.

        for epoch in range(args.num_epochs):

            print('* epoch {}, time: {}'.format(epoch, get_time()))

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

            train_triplet_loss = 0.

            train_vol_rand_loss = 0.
            train_vol_recon_loss = 0.

            train_c_real_score = 0.
            train_c_fake_score = 0.
            train_c_con_fake_score = 0.
            train_c_gp = 0.
            train_c_gp_con_randn = 0.
            train_c_loss = 0.

            num_batches = 0.

            epoch_log = {}

            d_model.train()
            g_model.train()
            e_model.train()
            cd_model.train()
            c_con_trans.train()

            for batch, batch_data in enumerate(train_loader):

                if warm_up_count < args.dis_warmup - 1:
                    warm_up_count += 1

                feas, vols, real_images = batch_data

                # train the encoder-generator
                for p in d_model.parameters():
                    p.requires_grad = False
                for p in cd_model.parameters():
                    p.requires_grad = False

                for p in e_model.parameters():
                    p.requires_grad = True
                for p in g_model.parameters():
                    p.requires_grad = True
                for p in c_con_trans.parameters():
                    p.requires_grad = True

                g_optimizer.zero_grad()
                e_optimizer.zero_grad()
                c_con_trans_optimizer.zero_grad()

                _batch_size = real_images.size(0)

                z_rand = torch.randn((_batch_size, args.channel_base * 32)).cuda()

                z_pos_rand = c_con_trans(z=z_rand, fea=feas)
                x_rand = g_model(z_pos_rand)

                neg_feas = feas + torch.randn_like(feas)
                z_neg_rand = c_con_trans(z=z_rand, fea=neg_feas)
                x_neg_rand = g_model(z_neg_rand)

                z_hat = e_model(real_images)
                x_hat = g_model(z_hat)

                c_loss = - cd_model(z_hat).mean() - cd_model(z_pos_rand).mean()

                d_fake_score = d_model(x_rand).mean()
                d_recon_score = d_model(x_hat).mean()
                eg_loss = - d_fake_score - d_recon_score

                x_hat_vol = torch.sum(input=x_hat, dim=[2, 3, 4])[:, 0:3] * args.down_factor ** 3  # size: (batch_size, 3)
                x_rand_vol = torch.sum(input=x_rand, dim=[2, 3, 4])[:, 0:3] * args.down_factor ** 3  # size: (batch_size, 3)
                vol_recon_loss = classify_cri(x_hat_vol, vols[:, 0:3])
                vol_rand_loss = classify_cri(x_rand_vol, vols[:, 0:3])
                vol_loss = vol_recon_loss + vol_rand_loss

                x_neg_rand_vol = torch.sum(input=x_neg_rand, dim=[2, 3, 4])[:, 0:3] * args.down_factor ** 3  # size: (batch_size, 3)
                triplet_loss = triplet_cri(vols[:, 0:3], x_rand_vol, x_neg_rand_vol)

                recon_loss = args.recon_w * recon_cri(x_hat, real_images)

                loss1 = args.cd_w * c_loss + eg_loss + recon_loss + args.vol_loss_w * vol_loss + args.triplet_w * triplet_loss
                loss1 = dis_penalty[warm_up_count] * loss1

                loss1.backward()

                torch.nn.utils.clip_grad_value_(parameters=e_model.parameters(), clip_value=args.gradient_clip_val)
                torch.nn.utils.clip_grad_value_(parameters=g_model.parameters(), clip_value=args.gradient_clip_val)
                torch.nn.utils.clip_grad_value_(parameters=c_con_trans.parameters(), clip_value=args.gradient_clip_val)

                e_optimizer.step()
                g_optimizer.step()
                c_con_trans_optimizer.step()

                train_g_fake_score += d_fake_score.item()
                train_g_recon_score += d_recon_score.item()
                train_e_score += - c_loss.item()

                train_triplet_loss += triplet_loss.item()
                train_recon_loss += recon_loss.item()
                train_vol_recon_loss += vol_recon_loss.item()
                train_vol_rand_loss += vol_rand_loss.item()
                train_eg_loss += loss1.item()

                num_batches += 1

                if num_batches % args.num_eg_iters:

                    if not args.freeze_img_dis:

                        # train the discriminator
                        for p in d_model.parameters():
                            p.requires_grad = True

                        for p in cd_model.parameters():
                            p.requires_grad = False
                        for p in e_model.parameters():
                            p.requires_grad = False
                        for p in g_model.parameters():
                            p.requires_grad = False
                        for p in c_con_trans.parameters():
                            p.requires_grad = False

                        d_optimizer.zero_grad()

                        z_rand = torch.randn((_batch_size, args.channel_base * 32)).cuda()
                        z_rand = c_con_trans(z=z_rand, fea=feas)
                        x_rand = g_model(z_rand)
                        d_fake_score = d_model(x_rand).mean()

                        z_hat = e_model(real_images)
                        x_hat = g_model(z_hat)
                        d_recon_score = d_model(x_hat).mean()

                        d_real_score = d_model(real_images).mean()

                        real_images.requires_grad = True
                        gp_fake = calc_gradient_penalty(d_model, real_images, x_rand)
                        gp_recon = calc_gradient_penalty(d_model, real_images, x_hat)
                        real_images.requires_grad = False

                        loss2 = d_fake_score + d_recon_score - 2 * d_real_score + args.gp_lambda * (gp_fake + gp_recon)
                        loss2 = dis_penalty[warm_up_count] * loss2

                        loss2.backward()
                        torch.nn.utils.clip_grad_value_(parameters=d_model.parameters(), clip_value=args.gradient_clip_val)

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
                    for p in c_con_trans.parameters():
                        p.requires_grad = False

                    cd_optimizer.zero_grad()

                    z_real_rand = torch.randn((_batch_size, args.channel_base * 32)).cuda()
                    z_real_rand.requires_grad = True
                    cd_real_score = cd_model(z_real_rand).mean()

                    z_hat = e_model(real_images)
                    cd_fake_score = cd_model(z_hat).mean()
                    gp_cd = calc_gradient_penalty(cd_model, z_real_rand, z_hat)

                    z_rand = torch.randn((_batch_size, args.channel_base * 32)).cuda()
                    z_rand = c_con_trans(z=z_rand, fea=feas)
                    cd_fake2_score = cd_model(z_rand).mean()
                    gp_cd2 = calc_gradient_penalty(cd_model, z_real_rand, z_rand)

                    loss3 = cd_fake_score + cd_fake2_score - 2 * cd_real_score + args.gp_lambda * (gp_cd + gp_cd2)
                    loss3 = dis_penalty[warm_up_count] * loss3

                    loss3.backward()
                    torch.nn.utils.clip_grad_value_(parameters=cd_model.parameters(), clip_value=args.gradient_clip_val)

                    cd_optimizer.step()

                    train_c_real_score += cd_real_score.item()
                    train_c_fake_score += cd_fake_score.item()
                    train_c_con_fake_score += cd_fake2_score.item()
                    train_c_gp += gp_cd.item()
                    train_c_gp_con_randn += gp_cd2.item()
                    train_c_loss += loss3.item()

            epoch_log['1_gen_fake_dis_score'] = train_g_fake_score / num_batches / args.batch_size
            epoch_log['1_gen_recon_dis_score'] = train_g_recon_score / num_batches / args.batch_size

            epoch_log['1_enc_fake_code_score'] = train_e_score / num_batches / args.batch_size

            epoch_log['1_recon_img_loss'] = train_recon_loss / num_batches / args.batch_size
            epoch_log['1_triplet_loss'] = train_triplet_loss / num_batches / args.batch_size
            epoch_log['1_recon_vol_loss'] = train_vol_recon_loss / num_batches / args.batch_size
            epoch_log['1_randn_vol_loss'] = train_vol_rand_loss / num_batches / args.batch_size
            epoch_log['loss1'] = train_eg_loss / num_batches / args.batch_size

            epoch_log['2_img_dis_real_score'] = train_d_real_score / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['2_img_dis_randn_fake_score'] = train_d_fake_score / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['2_img_dis_recon_score'] = train_d_recon_score / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['2_img_dis_randn_fake_gp'] = train_fake_gp / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['2_img_dis_recon_gp'] = train_recon_gp / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['loss2'] = train_d_loss / num_batches / args.batch_size * args.num_eg_iters

            epoch_log['3_code_dis_real_randn_score'] = train_c_real_score / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['3_code_dis_encoder_output_score'] = train_c_fake_score / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['3_code_dis_randn_trans_score'] = train_c_con_fake_score / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['3_code_dis_real_randn-encoder_output_gp'] = train_c_gp / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['3_code_dis_real_randn-randn_trans_score'] = train_c_gp_con_randn / num_batches / args.batch_size * args.num_eg_iters
            epoch_log['loss3'] = train_c_loss / num_batches / args.batch_size * args.num_eg_iters

            epoch_log['dis_warmup'] = dis_penalty[warm_up_count]

            epoch_log['d_lr'] = d_optimizer.param_groups[0]['lr']
            epoch_log['e_lr'] = e_optimizer.param_groups[0]['lr']
            epoch_log['g_lr'] = g_optimizer.param_groups[0]['lr']
            epoch_log['c_con_lr'] = c_con_trans_optimizer.param_groups[0]['lr']
            epoch_log['cd_lr'] = cd_optimizer.param_groups[0]['lr']

            avg_vol, loss1, loss2, loss3, [wm_vols, gm_vols, csf_vols, roi_cors] = (
                self.vol_valid(g_model=g_model, d_model=d_model, e_model=e_model, cd_model=cd_model,
                               c_con_trans=c_con_trans, data_loader=valid_loader, epoch_log=epoch_log))

            if avg_vol > max_avg_cor:
                print('\t* best epoch {}, average correlation coefficient improved: {} -> {}'.format(epoch, max_avg_cor, avg_vol))
                max_avg_cor = avg_vol

                self.save_model_img('best', e_model, g_model, d_model, cd_model, d_optimizer, g_optimizer, e_optimizer,
                                    cd_optimizer, None, c_con_trans, None, None,
                                    c_con_trans_optimizer, None)

            if epoch % args.save_freq == 0:
                self.save_model_img(epoch, e_model, g_model, d_model, cd_model, d_optimizer, g_optimizer,
                                    e_optimizer,
                                    cd_optimizer, None, c_con_trans, None, None,
                                    c_con_trans_optimizer, None)

            test_roi2cors = self.vol_test()

            epoch_log['wm_coff_test'] = test_roi2cors['wm']
            epoch_log['gm_coff_test'] = test_roi2cors['gm']
            epoch_log['csf_coff_test'] = test_roi2cors['csf']
            epoch_log['bg_coff_test'] = test_roi2cors['bg']

    def save_model_img(self, name, e_model, g_model, d_model, cd_model, d_optimizer, g_optimizer, e_optimizer, cd_optimizer,
                       c_classifier, c_con_trans, img_classifier, c_classifier_optimizer, c_con_trans_optimizer, img_classifier_optimizer):

        with torch.no_grad():
            e_model.eval()
            g_model.eval()

            real_imgs = torch.stack([self.test_dataset.subj2mri[subj] for subj in self.check_subjs], dim=0).cuda()

            feas = torch.stack([self.test_dataset.subj2feas[subj] for subj in self.check_subjs], dim=0).cuda()
            z_rand = torch.randn((args.num_gen_samples, args.channel_base * 32)).cuda()
            z_rand = c_con_trans(z=z_rand, fea=feas)
            fake_images = g_model(z_rand)

            recon_images = g_model(e_model(real_imgs))

            write_obj(obj={'subjs': self.check_subjs, 'real': real_imgs.cpu(), 'fake': fake_images.cpu(), 'recon': recon_images.cpu()},
                      file_path=os.path.join(args.model_path, '{}(check_{}).pickle'.format(args.timestamp, name)))

        write_obj(obj=[d_model.state_dict(), g_model.state_dict(), e_model.state_dict(), cd_model.state_dict(),
                       c_con_trans.state_dict(),
                       d_optimizer.state_dict(), g_optimizer.state_dict(), e_optimizer.state_dict(), cd_optimizer.state_dict(),
                       c_con_trans_optimizer.state_dict()],
                  file_path=os.path.join(args.model_path, '{}(model_{}).pickle'.format(args.timestamp, name)))

    def vol_valid(self, g_model, d_model, e_model, cd_model, c_con_trans, data_loader, epoch_log):
        with torch.no_grad():
            g_model.eval()
            d_model.eval()
            e_model.eval()
            cd_model.eval()
            c_con_trans.eval()

            recon_cri = torch.nn.BCELoss(reduction='mean')
            classify_cri = torch.nn.L1Loss(reduction='mean')
            triplet_cri = torch.nn.TripletMarginLoss(margin=self.margin_loss_margin, p=1, reduction='mean')

            wm_vols, gm_vols, csf_vols, bg_vols = [[], []], [[], []], [[], []], [[], []]
            loss1 = 0.
            loss2 = 0.
            loss3 = 0.
            for batch, batch_data in enumerate(data_loader):
                feas, vols, real_images = batch_data
                _batch_size = real_images.size(0)

                z_rand = torch.randn((_batch_size, args.channel_base * 32)).cuda()
                z_pos_rand = c_con_trans(z=z_rand, fea=feas)
                x_rand = g_model(z_pos_rand)

                neg_feas = feas + torch.randn_like(feas)
                z_neg_rand = c_con_trans(z=z_rand, fea=neg_feas)
                x_neg_rand = g_model(z_neg_rand)

                d_fake_score = d_model(x_rand).mean()

                z_hat = e_model(real_images)
                x_hat = g_model(z_hat)

                d_recon_score = d_model(x_hat).mean()
                d_real_score = d_model(real_images).mean()

                loss2 += d_fake_score + d_recon_score - 2 * d_real_score

                x_rand_vol = torch.sum(input=x_rand, dim=[2, 3, 4]) * args.down_factor ** 3  # size: (batch_size, 4)
                for _ in range(x_rand_vol.size(0)):
                    wm_vols[0].append(vols[_][0])
                    wm_vols[1].append(x_rand_vol[_][0])
                    gm_vols[0].append(vols[_][1])
                    gm_vols[1].append(x_rand_vol[_][1])
                    csf_vols[0].append(vols[_][2])
                    csf_vols[1].append(x_rand_vol[_][2])
                    bg_vols[0].append(vols[_][3])
                    bg_vols[1].append(x_rand_vol[_][3])

                c_loss = - cd_model(z_hat).mean() - cd_model(z_pos_rand).mean()

                d_fake_score = d_model(x_rand).mean()
                d_recon_score = d_model(x_hat).mean()

                eg_loss = - d_fake_score - d_recon_score

                x_hat_vol = torch.sum(input=x_hat, dim=[2, 3, 4])[:, 0:3] * args.down_factor ** 3  # size: (batch_size, 3)
                x_rand_vol = torch.sum(input=x_rand, dim=[2, 3, 4])[:, 0:3] * args.down_factor ** 3  # size: (batch_size, 3)

                vol_recon_loss = classify_cri(x_hat_vol, vols[:, 0:3])
                vol_rand_loss = classify_cri(x_rand_vol, vols[:, 0:3])
                vol_loss = vol_recon_loss + vol_rand_loss

                recon_loss = args.recon_w * recon_cri(x_hat, real_images)

                x_neg_rand_vol = torch.sum(input=x_neg_rand, dim=[2, 3, 4])[:, 0:3] * args.down_factor ** 3  # size: (batch_size, 3)
                triplet_loss = triplet_cri(vols[:, 0:3], x_rand_vol, x_neg_rand_vol)

                loss1 += args.cd_w * c_loss + eg_loss + recon_loss + args.vol_loss_w * vol_loss + args.triplet_w * triplet_loss

                z_real_rand = torch.randn((_batch_size, args.channel_base * 32)).cuda()

                cd_real_score = cd_model(z_real_rand).mean()

                cd_fake_score = cd_model(z_hat).mean()

                cd_fake2_score = cd_model(z_pos_rand).mean()

                loss3 += cd_fake_score + cd_fake2_score - 2 * cd_real_score

            wm_vols, gm_vols, csf_vols, bg_vols = (torch.FloatTensor(wm_vols), torch.FloatTensor(gm_vols),
                                                   torch.FloatTensor(csf_vols), torch.FloatTensor(bg_vols))

            roi_cors = {'wm': torch.corrcoef(wm_vols)[0, 1].item(),
                        'gm': torch.corrcoef(gm_vols)[0, 1].item(),
                        'csf': torch.corrcoef(csf_vols)[0, 1].item(),
                        'bg': torch.corrcoef(bg_vols)[0, 1].item()}

            print('\t* volume correlation {}'.format(roi_cors))

            epoch_log['wm_coff'] = roi_cors['wm']
            epoch_log['gm_coff'] = roi_cors['gm']
            epoch_log['csf_coff'] = roi_cors['csf']
            epoch_log['bg_coff'] = roi_cors['bg']

            return (roi_cors['wm'] + roi_cors['gm'] + roi_cors['csf']) / 3., loss1, loss2, loss3, [wm_vols, gm_vols, csf_vols, roi_cors]

    def vol_test(self):
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=args.batch_size, collate_fn=collate_func, shuffle=False)

        e_model = AlphaEncoder(channel_base=args.channel_base)
        cd_model = CodeDiscriminator(channel_base=args.channel_base)
        d_model = Discriminator(channel_base=args.channel_base)

        g_model = Generator(channel_base=args.channel_base)
        c_con_trans = ConCodeTrans(hid_dim=args.channel_base * 32, num_fea=len(self.fea2mean_std))

        if args.num_devices > 1:
            g_model = torch.nn.DataParallel(g_model)
            c_con_trans = torch.nn.DataParallel(c_con_trans)
            d_model = torch.nn.DataParallel(d_model)
            e_model = torch.nn.DataParallel(e_model)
            cd_model = torch.nn.DataParallel(cd_model)

        (d_model_state_dict, g_model_state_dict, e_model_state_dict,
         cd_model_state_dict, c_con_trans_state_dict, _, _, _, _, _) = read_obj(
            file_path=os.path.join(args.model_path, '{}(model_best).pickle'.format(args.timestamp)))
        g_model.load_state_dict(g_model_state_dict)
        d_model.load_state_dict(d_model_state_dict)
        e_model.load_state_dict(e_model_state_dict)
        cd_model.load_state_dict(cd_model_state_dict)
        c_con_trans.load_state_dict(c_con_trans_state_dict)

        g_model.cuda()
        c_con_trans.cuda()
        e_model.cuda()
        d_model.cuda()
        cd_model.cuda()

        with torch.no_grad():
            g_model.eval()
            d_model.eval()
            e_model.eval()
            cd_model.eval()
            c_con_trans.eval()

            wm_vols, gm_vols, csf_vols, bg_vols = [[], []], [[], []], [[], []], [[], []]
            real_imgs, gen_imgs = [], []

            for batch, batch_data in enumerate(test_loader):
                feas, vols, real_images = batch_data
                _batch_size = real_images.size(0)

                z_rand = torch.randn((_batch_size, args.channel_base * 32)).cuda()
                z_rand = c_con_trans(z=z_rand, fea=feas)
                x_rand = g_model(z_rand)

                x_rand_vol = torch.sum(input=x_rand, dim=[2, 3, 4]) * args.down_factor ** 3  # size: (batch_size, 4)
                for _ in range(x_rand_vol.size(0)):
                    wm_vols[0].append(vols[_][0])
                    wm_vols[1].append(x_rand_vol[_][0])
                    gm_vols[0].append(vols[_][1])
                    gm_vols[1].append(x_rand_vol[_][1])
                    csf_vols[0].append(vols[_][2])
                    csf_vols[1].append(x_rand_vol[_][2])
                    bg_vols[0].append(vols[_][3])
                    bg_vols[1].append(x_rand_vol[_][3])

                real_imgs.append(real_images.cpu())
                gen_imgs.append(x_rand.cpu())

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

            print('\t* test volume correlation {}'.format(roi_cors))

            write_obj(obj={'wm': wm_vols, 'gm': gm_vols, 'csf': csf_vols, 'bg': bg_vols},
                      file_path=os.path.join(args.model_path, '{}({}).pickle'.format(args.timestamp, 'test_vols')))

            plot_corrs(wm_vols, gm_vols, csf_vols, roi_cors,
                       os.path.join(args.model_path, '{}(corr_{}).pdf'.format(args.timestamp, 'test')))

            return roi_cors


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--dataset', type=str, default='CamCan')

    args.add_argument('--data_path', default='datasets/{}/output', type=str)

    args.add_argument('--down_factor', default=2, type=int)

    args.add_argument('--split', type=float, default=[0.8, 0.1, 0.1], nargs='+')

    args.add_argument('--batch_size', default=8, type=int)

    args.add_argument('--num_gen_samples', default=3, type=int)

    args.add_argument('--num_epochs', default=5000, type=int)

    args.add_argument('--channel_base', default=32, type=int)

    args.add_argument('--pre_timestamp', default='', type=str)

    args.add_argument('--pre_epoch', default=0, type=int)

    args.add_argument('--pre_model_path', default='datasets/Aomic/models', type=str)

    args.add_argument('--continue_timestamp', default='', type=str)

    args.add_argument('--gp_lambda', default=1, type=float)

    args.add_argument('--recon_w', default=10, type=float)

    args.add_argument('--cd_w', default=1, type=float)

    args.add_argument('--vol_loss_w', default=100., type=float)

    args.add_argument('--triplet_w', default=100., type=float)

    args.add_argument('--dis_warmup', default=200, type=int)

    args.add_argument('--num_eg_iters', default=4, type=int)

    args.add_argument('--d_lr', default=1e-6, type=float)

    args.add_argument('--g_lr', default=1e-6, type=float)

    args.add_argument('--e_lr', default=1e-6, type=float)

    args.add_argument('--cd_lr', default=1e-6, type=float)

    args.add_argument('--c_con_lr', default=1e-4, type=float)

    args.add_argument('--random_seed', default=0, type=int)

    args.add_argument('--model_path', default='datasets/{}/models', type=str)

    args.add_argument('--num_devices', type=int, default=torch.cuda.device_count())

    args.add_argument('--timestamp', type=str, default=get_time())

    args.add_argument('--is_eval', action='store_true')

    args.add_argument('--freeze_img_dis', action='store_true')

    args.add_argument('--save_freq', default=20, type=int)

    args.add_argument('--gradient_clip_val', default=1., type=float)

    args.add_argument('--margin_weight', default=0., type=float)

    args = args.parse_args()

    args.data_path = args.data_path.format(args.dataset)
    args.model_path = args.model_path.format(args.dataset)

    print('## CSegSynth Finetuning - {}'.format(args.timestamp))
    for k, v in vars(args).items():
        print('* {}: {}'.format(k, v))

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    main = Main()
    if not args.is_eval:
        main.train()
        main.vol_test()
    else:
        args.timestamp = args.continue_timestamp
        main.vol_test()

