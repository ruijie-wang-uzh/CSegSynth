"""
Data preprocessing
"""


import os
import torch
from tqdm import tqdm
import nibabel as nib
from functools import reduce
from argparse import ArgumentParser
from collections import defaultdict
from utils import write_obj, read_obj


def data_prep_aomic():
    subjs = [int(_[4:]) for _ in os.listdir(args.in_path) if _[:3] == 'sub']
    print('\t* number of subjects: {}'.format(len(subjs)))

    subj2mri = defaultdict(dict)
    subj2vols = defaultdict(dict)

    for subj in tqdm(subjs):
        mri_path = os.path.join(args.in_path, 'sub-{:04d}'.format(subj), 'anat')
        assert os.path.exists(mri_path), '{} does not exist'.format(mri_path)
        for run_name in ['run-1']:
            tmp_path = os.path.join(mri_path, run_name)
            if os.path.exists(tmp_path):
                for roi_idx, roi_name in zip([0, 1, 2], ['csf', 'gm', 'wm']):
                    image_obj = nib.load(os.path.join(
                        tmp_path, 'fast_sub-{:04d}_{}_pve_{}.nii.gz'.format(subj, run_name, roi_idx)))
                    image_tensor = image_obj.get_fdata()
                    image_tensor = torch.from_numpy(image_tensor).float()  # size: (160, 256, 256)
                    subj2mri[subj][roi_name] = image_tensor  # size: (160, 256, 256)

                    unit_vol = image_obj.header.get_zooms()
                    unit_vol = reduce(lambda x, y: x * y, unit_vol)  # unit volume in mm3
                    assert unit_vol == 1., print('unit volume is not 1!')
                    roi_vol = torch.sum(image_tensor)
                    subj2vols[subj][roi_name] = roi_vol

    write_obj(obj=subj2mri, file_path=os.path.join(args.out_path, 'subj2mri.pickle'))
    write_obj(obj=subj2vols, file_path=os.path.join(args.out_path, 'subj2vols.pickle'))

    subj2feas = defaultdict(dict)
    fea_path = os.path.join(args.in_path, 'participants.tsv')
    vec_dict = {'low': 0., 'medium': 0.5, 'high': 1., 'male': 0., 'female': 1., 'left': 0., 'right': 1.}
    with open(fea_path, 'r') as f:
        lines = f.readlines()
        feas = lines[0].strip().split('\t')[1:21]  # feature names
        for line in lines[1:]:
            elems = line.strip().split('\t')[:21]
            subj = int(elems[0][4:])
            if 'n/a' in elems:
                print('\t* found n/a feature for subject-{:04d}'.format(subj))
            else:
                for idx, elem in enumerate(elems[1:]):
                    subj2feas[subj][feas[idx]] = float(elem) if elem not in vec_dict.keys() else vec_dict[elem]
    write_obj(obj=subj2feas, file_path=os.path.join(args.out_path, 'subj2feas.pickle'))


def down_sampling():
    print('* downsizing mri data by factor {}'.format(args.down_factor))
    subj2mri = read_obj(file_path=os.path.join(args.out_path, 'subj2mri.pickle'))
    subj2vols = read_obj(file_path=os.path.join(args.out_path, 'subj2vols.pickle'))

    conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=args.down_factor,
                           stride=args.down_factor, padding=0, bias=False)
    conv.weight.data.fill_(1.)
    conv.eval()

    subj2d_mri = defaultdict(dict)
    with torch.no_grad():
        for subj, mris in tqdm(subj2mri.items()):
            for roi_name, mri_tensor in mris.items():
                subj2d_mri[subj][roi_name] = conv(mri_tensor.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0) / args.down_factor ** 3
                assert torch.isclose(torch.sum(subj2d_mri[subj][roi_name]) * args.down_factor ** 3, subj2vols[subj][roi_name]), \
                    print('subj-{}-{}'.format(subj, roi_name), torch.sum(subj2d_mri[subj][roi_name]), subj2vols[subj][roi_name])

    write_obj(obj=subj2d_mri, file_path=os.path.join(args.out_path, 'subj2mri_d{}.pickle'.format(args.down_factor)))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, default='Aomic')
    arg_parser.add_argument('--in_path', type=str, default='datasets/{}')
    arg_parser.add_argument('--out_path', type=str, default='datasets/{}/output')
    arg_parser.add_argument('--down_factor', type=int, default=2, help='downsize data by factor x on each dimension')
    args = arg_parser.parse_args()

    args.in_path = args.in_path.format(args.dataset)
    args.out_path = args.out_path.format(args.dataset)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    print('\n## Data Preprocessing - {}'.format(args.dataset))
    for k, v in vars(args).items():
        print('* {}: {}'.format(k, v))

    if args.dataset == 'Aomic':
        data_prep_aomic()
        down_sampling()

