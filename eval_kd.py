# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed

import json
from libs.datasets.tools import data_split_dir
import copy

################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config, training=False)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    # get {classes} and {description_dict}
    # subset_sile = "./splits/train_{r1}_test_{r2}/THUMOS14/{mode}/split_{r3}.list"
    subset_file = data_split_dir(cfg['dataset']['subset_file'], cfg['dataset']['task'], cfg['dataset']['data_split'], 'test', cfg['dataset']['split_num'])
    with open(subset_file, 'rt') as f:
        classes = [line[:-1] for line in f.readlines()]
    description_dict = json.load(open(cfg['description_file_path']))

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    #model = nn.DataParallel(model, device_ids=cfg['devices'])
    model.set_paparameters(args, classes, description_dict, device=cfg['devices'][0])
    model = model.to(cfg['devices'][0])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    #model.load_state_dict(checkpoint['state_dict_ema'])
    state_dict = checkpoint['state_dict_ema']
    model_state_dict = model.state_dict()

    # 只加载匹配的权重
    for key in state_dict.keys():
        if key in model_state_dict:
            if state_dict[key].shape == model_state_dict[key].shape:
                model_state_dict[key] = state_dict[key]
            else:
                print(f"Skipping loading '{key}' due to shape mismatch: "
                    f"{state_dict[key].shape} vs {model_state_dict[key].shape}")

    # 更新模型的状态字典
    model.load_state_dict(model_state_dict)
    del checkpoint

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds'], 
            subset_file = subset_file, 
            feat_folder = cfg['dataset']['feat_folder'], 
            file_prefix = cfg['dataset']['file_prefix'], 
            file_ext = cfg['dataset']['file_ext'], 
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""

    ## TTA-KD 复制预训练模型为教师模型
    teacher_model = copy.deepcopy(model)

    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq, 
        output_json_path = os.path.join('./results', os.path.basename(os.path.dirname(ckpt_file)), os.path.basename(ckpt_file).replace('.tar', '.json')), 
        write_json = args.write_json, 
        w_tent = args.tent, 
        w_eata = args.eata, 
        w_deyo = args.deyo, 
        w_sar = args.sar, 
        w_kd = True, 
        teacher_model = teacher_model
    )
    print(mAP)
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')

    ## Write json_file
    parser.add_argument('--write_json', action='store_true')
    
    ## InternVideo
    parser.add_argument('--internvideo', action='store_true')
    parser.add_argument('--internvideo_ckpt', type=str, default='/data/qianyihao/Datasets/InternVideo_ckpt/InternVideo-MM-L-14.ckpt')
    
    ## Clip
    parser.add_argument('--use_clip', action='store_true')
    parser.add_argument('--linear_type', type=str, choices=['no', 'only_visual', 'only_text', 'visual_text'], default='no') ##

    ## GAP
    parser.add_argument('--use_gap_clip', action='store_true')
    parser.add_argument('--feats_type', type=str, choices=['i3d_i3d', 'clip_i3d', 'fpn_i3d'], default='clip_i3d')

    ## TTT
    parser.add_argument('--use_ttt', action='store_true')
    parser.add_argument('--ttt_type', type=str, choices=['un_ttt', 'bi_ttt'], default='bi_ttt')
    parser.add_argument('--bi_ttt_type', type=str, choices=['single', 'double'], default='double')
    parser.add_argument('--mini_batch_size', type=int, default = 64)
    parser.add_argument('--window_size', type=int, default = 72)

    parser.add_argument('--encoder_version', type=str, choices=['v0', 'v1'], default='v0')  ##
    parser.add_argument('--num_ttt_encoders', type=int, help='Number of TTT Encoders (must be greater than 0)', default=6)  ##
    parser.add_argument('--ttt_pos', type=int, default = -1, help='Valid when encoder_version == v1')

    parser.add_argument('--ar_pred', action='store_true')   ## Autoregression Prediction
    parser.add_argument('--tsa_decoder', type=int, default=0)

    ## Text Prompt Tuning (TPT)
    parser.add_argument('--use_tpt', action='store_true')   ## from EffPrompt
    parser.add_argument('--use_tpt_stale', action='store_true')
    
    ## Memory-guided Prediction Refinement (MPR)
    parser.add_argument('--use_mpr', action='store_true')

    ## OIC
    parser.add_argument('--oic_loss', action='store_true')
    parser.add_argument('--oic_loss_weight', type=float, default = 1)

    ## OnZeta
    parser.add_argument('--onzeta', action='store_true')

    ## TTA
    parser.add_argument('--tent', action='store_true')
    parser.add_argument('--eata', action='store_true')
    parser.add_argument('--sar', action='store_true')
    parser.add_argument('--deyo', action='store_true')

    args = parser.parse_args()
    main(args)
