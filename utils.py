"""
Some utility functions for data processing and visualization.
"""


import torch
import pickle
import numpy as np
from collections import defaultdict
from torch.utils import data
from datetime import datetime
import matplotlib.pyplot as plt


def read_obj(file_path: str) -> dict:
    print("\t* loading from {} at {} ...".format(file_path, get_time()))
    with open(file=file_path, mode="rb") as f:
        obj = pickle.load(file=f)
    return obj


def get_time() -> str:
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")


class MRIData(data.Dataset):
    def __init__(self, subjs: list, subj2mri: dict):
        super(MRIData, self).__init__()
        self.subjs = subjs
        self.subj2mri = {}
        self.rois = ['wm', 'gm', 'csf']
        for subj in self.subjs:
            mris = torch.stack([subj2mri[subj][_] for _ in self.rois])
            mris = torch.cat([mris, 1. - torch.sum(mris, dim=0, keepdim=True)], dim=0)
            mris = torch.clamp(mris, 0., 1.)
            self.subj2mri[subj] = mris

    def __len__(self):
        return len(self.subjs)

    def __getitem__(self, item):
        subj = self.subjs[item]
        return self.subj2mri[subj]


def mri_collate_func(batch_data: list):
    return torch.stack(batch_data, dim=0).cuda()


class SubjData(data.Dataset):
    def __init__(self, dataset: str,
                 subjs: list,
                 subj2feas: dict,
                 fea2mean_std: dict,
                 subj2vols: dict,
                 subj2mri: dict,
                 down_factor: int,
                 features: list=None):
        super(SubjData, self).__init__()
        if features:
            self.features = features
        else:
            if dataset == 'Aomic':
                self.features = ['age', 'sex', 'handedness', 'BMI', 'education_level', 'background_SES', 'IST_fluid',
                                 'IST_memory', 'IST_crystallised', 'IST_intelligence_total', 'BAS_drive', 'BAS_fun',
                                'BAS_reward', 'BIS', 'NEO_N', 'NEO_E', 'NEO_O',
                                'NEO_A', 'NEO_C', 'STAI_T']
            elif dataset == 'CamCan':
                self.features = list(fea2mean_std.keys())

        self.rois = ['wm', 'gm', 'csf']

        self.subjs = subjs
        self.subj2feas = {
            subj: torch.FloatTensor(
                [(subj2feas[subj][fea] - fea2mean_std[fea]['mean']) / (fea2mean_std[fea]['std'] + 1e-16) for fea in self.features]
            )
            for subj in self.subjs
        }

        self.subj2vols = {}
        self.subj2mri = {}
        for subj in self.subjs:
            vols = torch.FloatTensor([subj2vols[subj][_] for _ in self.rois])
            mris = torch.stack([subj2mri[subj][_] for _ in self.rois])
            assert mris.size(-1) == 256 // down_factor and mris.size(-2) == 256 // down_factor and mris.size(-3) == 160 // down_factor
            vols = torch.cat([vols, 160 * 256 * 256 - torch.sum(vols, dim=0, keepdim=True)], dim=0)
            mris = torch.cat([mris, 1. - torch.sum(mris, dim=0, keepdim=True)], dim=0)
            mris = torch.clamp(mris, 0., 1.)
            self.subj2vols[subj] = vols
            self.subj2mri[subj] = mris

    def __len__(self):
        return len(self.subjs)

    def __getitem__(self, item):
        subj = self.subjs[item]
        return self.subj2feas[subj], self.subj2vols[subj], self.subj2mri[subj]


def mean_std_estimate(subj2feas: dict, subjs: list):
    fea2mean_std = defaultdict(dict)
    fea2values = defaultdict(list)
    for subj in subjs:
        feas = subj2feas[subj]
        for fea, value in feas.items():
            if str(value) != 'nan':
                fea2values[fea].append(value)
    for fea, values in fea2values.items():
        std, mean = torch.std_mean(torch.FloatTensor(values))
        fea2mean_std[fea]['mean'] = mean.item()
        fea2mean_std[fea]['std'] = std.item()
    return fea2mean_std


def remove_nan(subj2feas: dict, fea2mean_std: dict):
    for subj, feas in subj2feas.items():
        for fea, value in feas.items():
            if str(value) == 'nan':
                subj2feas[subj][fea] = fea2mean_std[fea]['mean']
    return


def collate_func(batch_data: list):
    feas, vols, mris = [], [], []
    for _ in batch_data:
        feas.append(_[0])
        vols.append(_[1])
        mris.append(_[2])
    feas, vols, mris = torch.stack(feas, dim=0).cuda(), torch.stack(vols, dim=0).cuda(), torch.stack(mris, dim=0).cuda()
    return feas, vols, mris


def write_obj(obj: object, file_path: str) -> None:
    print("\t* dumping to {} at {} ...".format(file_path, get_time()))
    with open(file=file_path, mode="wb") as f:
        pickle.dump(obj=obj, file=f, protocol=4)


def roi_mean_std_estimate(subj2vols: dict, subjs: list):
    roi2mean_std = {}
    rois = ['wm', 'gm', 'csf']
    for roi in rois:
        roi_vols = torch.FloatTensor([subj2vols[subj][roi] for subj in subjs])
        roi2mean_std[roi] = {
            'mean': torch.mean(roi_vols).item(),
            'std': torch.std(roi_vols).item()
        }
    return roi2mean_std


def plot_corrs(wm_vols, gm_vols, csf_vols, roi_cors, save_path):
    plt.style.use('seaborn-darkgrid')

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].scatter(wm_vols[0], wm_vols[1])
    ax[0].set_title('WM volume correlation: {:.4f}'.format(roi_cors['wm']), fontsize=18)
    ax[0].set_xlabel('Ground-truth WM Volume', fontsize=18)
    ax[0].set_ylabel('Predicted WM Volume', fontsize=18)

    z = np.polyfit(wm_vols[0], wm_vols[1], 1)
    p = np.poly1d(z)
    ax[0].plot(wm_vols[0], p(wm_vols[0]), color='royalblue')


    ax[1].scatter(gm_vols[0], gm_vols[1])
    ax[1].set_title('GM volume correlation: {:.4f}'.format(roi_cors['gm']), fontsize=18)
    ax[1].set_xlabel('Ground-truth GM Volume', fontsize=18)
    ax[1].set_ylabel('Predicted GM Volume', fontsize=18)

    z = np.polyfit(gm_vols[0], gm_vols[1], 1)
    p = np.poly1d(z)
    ax[1].plot(gm_vols[0], p(gm_vols[0]), color='royalblue')

    ax[2].scatter(csf_vols[0], csf_vols[1])
    ax[2].set_title('CSF volume correlation: {:.4f}'.format(roi_cors['csf']), fontsize=18)
    ax[2].set_xlabel('Ground-truth CSF Volume', fontsize=18)
    ax[2].set_ylabel('Predicted CSF Volume', fontsize=18)

    z = np.polyfit(csf_vols[0], csf_vols[1], 1)
    p = np.poly1d(z)
    ax[2].plot(csf_vols[0], p(csf_vols[0]), color='royalblue')

    print('\t* plot the correlations to {}'.format(save_path))
    plt.savefig(save_path, format='pdf', dpi=900, bbox_inches='tight')
