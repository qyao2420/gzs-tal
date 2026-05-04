import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss

from ..utils import batched_nms

from .clip_model import clip
import torchvision.ops.roi_align as ROIalign
import torchvision.ops.roi_pool as ROIpooling

from .prompt import text_prompt
from .cliprompt import CLIPrompt

from transformers import CLIPTokenizer, CLIPModel

from .InternVideo import internvideo

from .ttt_model.ttt_encoder_un import TTT_Encoder_Un
from .ttt_model.ttt_encoder_bi_s import TTT_Encoder_Bi_S
from .ttt_model.ttt_encoder_bi_d import TTT_Encoder_Bi_D

# from plts.blue_and_red import plot_classification_scores
# from plts.below import plot_classification_segments

from .tta_tools import configure_model, forward_with_tent, forward_with_eata, forward_with_sar

class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False, 
        tsa_decoder = 0
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        self.tsa_decoder = tsa_decoder
        # 0: 不变 multi-head
        # 1: 替换 tsa
        # 2: 插入2 multi-head + tsa
        # 3: 插入3 multi-head + tsa
        # 4: 插入4 multi-head + tsa
        if self.tsa_decoder != 0:
            self.ttt_layer = TTT_Encoder_Bi_D(nhead=4, num_layers=1, d_model=input_dim, use_cache=True, mini_batch_size=64, ttt_layer_type='mlp', window_size=64)
            # self.ttt_layer = TTT_Encoder_Bi_S(nhead=4, num_layers=1, d_model=input_dim, use_cache=True, mini_batch_size=64, ttt_layer_type='mlp', window_size=64)
            # self.ttt_layer = TTT_Encoder_Un(nhead=4, num_layers=1, d_model=input_dim, use_cache=True, mini_batch_size=64, ttt_layer_type='mlp', window_size=64)            

        if self.tsa_decoder != 1:
            # build the conv head
            self.head = nn.ModuleList()
            self.norm = nn.ModuleList()
            for idx in range(num_layers-1):
                if idx == 0:
                    in_dim = input_dim
                    out_dim = feat_dim
                else:
                    in_dim = feat_dim
                    out_dim = feat_dim
                self.head.append(
                    MaskedConv1D(
                        in_dim, out_dim, kernel_size,
                        stride=1,
                        padding=kernel_size//2,
                        bias=(not with_ln)
                    )
                )
                if with_ln:
                    self.norm.append(LayerNorm(out_dim))
                else:
                    self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat

            # 不变
            if self.tsa_decoder == 0:
                for idx in range(len(self.head)):
                    cur_out, _ = self.head[idx](cur_out, cur_mask)
                    cur_out = self.act(self.norm[idx](cur_out))
            # 替换
            elif self.tsa_decoder == 1:
                cur_out = self.ttt_layer(cur_feat.permute(0, 2, 1), cur_mask.permute(0, 2, 1)).permute(0, 2, 1)
            # 插入2 插入在前面
            elif self.tsa_decoder == 2:
                cur_out = self.ttt_layer(cur_feat.permute(0, 2, 1), cur_mask.permute(0, 2, 1)).permute(0, 2, 1)
                for idx in range(len(self.head)):
                    cur_out, _ = self.head[idx](cur_out, cur_mask)
                    cur_out = self.act(self.norm[idx](cur_out))
            # 插入3 插入在后面
            elif self.tsa_decoder == 3:
                for idx in range(len(self.head)):
                    cur_out, _ = self.head[idx](cur_out, cur_mask)
                    cur_out = self.act(self.norm[idx](cur_out))
                cur_out = self.ttt_layer(cur_feat.permute(0, 2, 1), cur_mask.permute(0, 2, 1)).permute(0, 2, 1)
            # 插入4 插入在中间
            elif self.tsa_decoder == 4:
                for idx in range(len(self.head)):
                    cur_out, _ = self.head[idx](cur_out, cur_mask)
                    cur_out = self.ttt_layer(cur_feat.permute(0, 2, 1), cur_mask.permute(0, 2, 1)).permute(0, 2, 1)
                    cur_out = self.act(self.norm[idx](cur_out))
            # 插入5
            elif self.tsa_decoder == 5:
                if l == 0:
                    cur_out = self.ttt_layer(cur_feat.permute(0, 2, 1), cur_mask.permute(0, 2, 1)).permute(0, 2, 1)
                else:
                    for idx in range(len(self.head)):
                        cur_out, _ = self.head[idx](cur_out, cur_mask)
                        cur_out = self.act(self.norm[idx](cur_out))
            # 替换6
            elif self.tsa_decoder == 6:
                for idx in range(len(self.head)):
                    cur_out = self.ttt_layer(cur_feat.permute(0, 2, 1), cur_mask.permute(0, 2, 1)).permute(0, 2, 1)
                    cur_out = self.act(self.norm[idx](cur_out))
            # 替换7
            elif self.tsa_decoder == 7:
                if l == 0:
                    for idx in range(len(self.head)):
                        cur_out = self.ttt_layer(cur_feat.permute(0, 2, 1), cur_mask.permute(0, 2, 1)).permute(0, 2, 1)
                        cur_out = self.act(self.norm[idx](cur_out))
                else:
                    for idx in range(len(self.head)):
                        cur_out, _ = self.head[idx](cur_out, cur_mask)
                        cur_out = self.act(self.norm[idx](cur_out))
            
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), )

        # fpn_masks remains the same
        return out_offsets


@register_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines #layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        num_classes,           # number of action classes
        train_cfg,             # other cfg for training
        test_cfg               # other cfg for testing
    ):
        super().__init__()
         # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*(1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer', 'conv']
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in' : input_dim, #2048
                    'n_embd' : embd_dim,    # 512
                    'n_head': n_head,   # 4
                    'n_embd_ks': embd_kernel_size,  # 3
                    'max_len': max_seq_len, # 2304
                    'arch' : backbone_arch, # (2, 2, 5)
                    'mha_win_size': self.mha_win_size,  # [19, 19, 19, 19, 19, 19]
                    'scale_factor' : scale_factor,  # 2
                    'with_ln' : embd_with_ln,   # True
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,  # 0.0
                    'path_pdrop' : self.train_droppath, #0.1
                    'use_abs_pe' : use_abs_pe,  # False
                    'use_rel_pe' : use_rel_pe,   # False
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln' : embd_with_ln
                }
            )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,   # 'identity'
            **{
                'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),   # 512 (2, 2, 5)
                'out_channel' : fpn_dim,    # 512
                'scale_factor' : scale_factor,  # 2
                'start_level' : fpn_start_level,    # 0
                'with_ln' : fpn_with_ln # True
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides,
                'regression_range' : self.reg_range
            }
        )

        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )
        self.fpn_dim = fpn_dim
        self.head_dim = head_dim
        self.head_kernel_size = head_kernel_size
        self.head_num_layers = head_num_layers
        self.head_with_ln = head_with_ln

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

        self.embd_dim = embd_dim

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]
    
    ## 为model新增一些参数
    def set_paparameters(self, args, classes, description_dict, device):

        self.classes_name = classes
        self.description_dict = description_dict
        self.num_classes = len(self.classes_name)

        ## TPT from EffPrompt
        self.use_tpt = args.use_tpt
        if self.use_tpt:
            actionlist, actiondict, actiontoken = text_prompt(self.classes_name, self.num_classes, device)
            self.cliprompt = CLIPrompt(actionlist, actiondict, actiontoken, device)
            #text_feats = self.cliprompt.myforward(self.classes_name)
            print('finish')
            
        ## TPT from STALE
        self.use_tpt_stale = args.use_tpt_stale
        if self.use_tpt_stale:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.txt_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").float().to(self.device)

        ## Memory-guided Prediction Refinement (MPR)
        self.use_mpr = args.use_mpr
        if self.use_mpr:
            # self.ttt_layer = TTT_Encoder_Un(nhead=8, num_layers=1, d_model=self.embd_dim, use_cache=True, mini_batch_size=64, ttt_layer_type='mlp', window_size=64)
            self.ttt_layer = TTT_Encoder_Bi_D(nhead=8, num_layers=1, d_model=self.embd_dim, use_cache=True, mini_batch_size=64, ttt_layer_type='mlp', window_size=64)
            self.ttt_linear_a = nn.Linear(2, self.embd_dim).to(self.device)
            self.ttt_linear_b = nn.Linear(self.embd_dim, 2).to(self.device)
            self.ttt_act = nn.ReLU()
            print('mpr')
        
        ## InternVideo
        self.internvideo = args.internvideo
        if self.internvideo:
            self.InternVideo_model = internvideo.load_model(args.internvideo_ckpt).to(self.device)
            print('internvideo')
            
            # Frozen internvideo params (fired text encoder)
            for param in self.InternVideo_model.visual.parameters():
                param.requires_grad = False
            for param in self.InternVideo_model.visual_ln_post.parameters():
                param.requires_grad = False  
            self.InternVideo_model.visual_proj.requires_grad = False

        ## OnZeta
        self.onzeta = args.onzeta
        
        # CLIP
        self.use_clip = args.use_clip
        if self.use_clip:
            '''self.clip_layer, preprocess = clip.load("ViT-B/16", device=self.device)
            self.clip_layer.set_paparameters(self.use_mlp)
            self.clip_layer = self.clip_layer.to(self.device)'''
            
            self.linear_type = args.linear_type
            if self.linear_type in {'only_visual', 'visual_text'}:
                self.visual_linear = nn.Linear(512, 512).to(self.device)
            if self.linear_type in {'only_text', 'visual_text'}:
                self.text_linear = nn.Linear(512, 512).to(self.device)

            self.clip_layer, preprocess = clip.load("ViT-B/16", device=self.device)
            self.clip_layer = self.clip_layer.to(self.device)
            self.text_encoder = self.clip_layer.encode_text
            self.logit_scale = self.clip_layer.logit_scale
            
            # Frozen CLIP params (fired text encoder)
            for param in self.clip_layer.visual.parameters():
                param.requires_grad = False
            
            # # Frozen clip_text_encoder
            # for param in self.clip_layer.parameters():
            #     param.requires_grad = True
        
        # GAP
        self.use_gap_clip = args.use_gap_clip
        if self.use_gap_clip:
            self.feats_type = args.feats_type
            if self.feats_type == 'i3d_i3d':
                self.linear_layer = nn.Linear(2048, 512).to(self.device)
                self.conv_layer = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1).to(self.device)

            self.clip_layer, preprocess = clip.load("ViT-B/16", device=self.device)
            self.clip_layer = self.clip_layer.to(self.device)
            self.text_encoder = self.clip_layer.encode_text
            self.logit_scale = self.clip_layer.logit_scale

            # 冻结clip_text_encoder
            for param in self.clip_layer.parameters():
                param.requires_grad = False

        # TTT
        self.use_ttt = args.use_ttt
        self.ttt_type = args.ttt_type
        self.bi_ttt_type = args.bi_ttt_type
        self.mini_batch_size = args.mini_batch_size
        self.window_size = args.window_size
        self.ttt_pos = args.ttt_pos
        self.encoder_version = args.encoder_version
        self.num_ttt_encoders = args.num_ttt_encoders
        self.ar_pred = args.ar_pred
        self.backbone.set_paparameters(self.use_ttt, self.ttt_type, self.bi_ttt_type, self.mini_batch_size, self.window_size, self.ttt_pos, self.encoder_version, self.num_ttt_encoders, self.ar_pred, self.device)

        if args.tsa_decoder != 0:
            self.reg_head = PtTransformerRegHead(
                self.fpn_dim, self.head_dim, len(self.fpn_strides),
                kernel_size=self.head_kernel_size,
                num_layers=self.head_num_layers,
                with_ln=self.head_with_ln, 
                tsa_decoder = args.tsa_decoder
            )
    
    def get_prompt(self, cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append("a video of a person doing"+" "+c)
        return temp_prompt
    
    # 获取text_feats
    def get_text_feats(self, cl_names, description_dict=None, device="cuda:0", target_type='prompt'):
        def get_prompt(cl_names):
            temp_prompt = []
            for c in cl_names:
                temp_prompt.append("a video of a person doing"+" "+c)
            return temp_prompt
        
        def get_description(cl_names):
            temp_prompt = []
            for c in cl_names:
                temp_prompt.append(description_dict[c]['Elaboration']['Description'][0]) # NOTE: default the idx of description is 0.
            return temp_prompt
        
        if target_type == 'prompt':
            act_prompt = get_prompt(cl_names)
        elif target_type == 'description':
            act_prompt = get_description(cl_names)
        elif target_type == 'name':
            act_prompt = cl_names
        else: 
            raise ValueError("Don't define this text_mode.")
        
        tokens = clip.tokenize(act_prompt).long().to(device) # input_ids->input_ids:[150,length]
        text_feats = self.text_encoder(tokens).float()

        return text_feats
    
    def _to_roi_align_format(self, rois, points, truely_length, scale_factor=1):
        #Convert RoIs to RoIAlign format.
        #Params:
        #    RoIs: normalized segments coordinates, shape (batch_size, num_segments, 2)
        #    T: length of the video feature sequence
        
        # transform to absolute axis
        B, N = rois.shape[:2]

        '''rois_center = rois[:, :, 0:1] # [B,N,1]
        rois_size = rois[:, :, 1:2] * scale_factor # [B,N,1]
        truely_length = truely_length.reshape(-1,1,1) # [B,1,1]
        rois_abs = torch.cat(
            (rois_center - rois_size/2, rois_center + rois_size/2), dim=2) * truely_length # [B,N,2]->"start,end"
        '''

        ds = rois[:, :, 0:1]
        de = rois[:, :, 1:2]
        points = points.repeat(B, 1, 1)
        seg_left = points[:, :, 0:1] - ds * points[:, :, 3:4]   # b x N x 1
        seg_right = points[:, :, 0:1] + de * points[:, :, 3:4]  # b x N x 1
        rois_abs = torch.cat((seg_left, seg_right), dim=2)  # b x N x 2
        
        truely_length = truely_length.reshape(-1,1,1) # [B,1,1]
        # expand the RoIs
        _max = truely_length.repeat(1,N,2)
        _min = torch.zeros_like(_max)
        rois_abs = torch.clamp(rois_abs, min=_min, max=_max)  # (B, N, 2)
        # transfer to 4 dimension coordination
        rois_abs_4d = torch.zeros((B,N,4),dtype=rois_abs.dtype,device=rois_abs.device)
        rois_abs_4d[:,:,0], rois_abs_4d[:,:,2] = rois_abs[:,:,0], rois_abs[:,:,1] # x1,0,x2,0

        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device) # [B,1,1]
        batch_ind = batch_ind.repeat(1, N, 1) # [B,N,1]
        rois_abs_4d = torch.cat((batch_ind, rois_abs_4d), dim=2) # [B,N,1+4]->"batch_id,x1,0,x2,0"
        # NOTE: stop gradient here to stablize training
        return rois_abs_4d.view((B*N, 5)).detach()


    def _roi_align(self, rois, points, origin_feat, mask, ROIalign_size, scale_factor=1):
        B,Q,_ = rois.shape
        B,T,C = origin_feat.shape
        truely_length = torch.sum(mask,dim=1) # [B]
        rois_abs_4d = self._to_roi_align_format(rois,points,truely_length,scale_factor) # (BxN) x 5
        feat = origin_feat.permute(0,2,1) # [B,dim,T]
        feat = feat.reshape(B,C,1,T)    # 2 x 512 x 1 x T
        #roi_feat = ROIalign(feat, rois_abs_4d, output_size=(1,ROIalign_size))
        #roi_feat = self.batch_roi(feat, rois_abs_4d, ROIalign_size) # (BxN) x 512 x
        roi_feat = ROIpooling(feat, rois_abs_4d, output_size=(1,ROIalign_size))
        roi_feat = roi_feat.reshape(B,Q,C,-1) # [B,Q,dim,output_width]
        roi_feat = roi_feat.permute(0,1,3,2) # [B,Q,output_width,dim]
        return roi_feat
    
    def batch_roi(self, feat, rois_abs_4d, ROIalign_size):
        roi_count = rois_abs_4d.shape[0]
        # 设置每个批次的大小
        batch_size_rois = 576  # 每个批次处理的 ROIs 数量 (可以根据 GPU 内存进行调整)
        num_batches = (roi_count + batch_size_rois - 1) // batch_size_rois  # 计算总共需要多少批次

        # 存储所有结果的列表
        outputs = []

        # 分批处理 ROIs
        for i in range(num_batches):
            start_index = i * batch_size_rois
            end_index = min(start_index + batch_size_rois, roi_count)
            
            # 取出当前批次的 ROIs
            rois_batch = rois_abs_4d[start_index:end_index]
            
            # 使用 ROI Align 计算特征
            output = ROIalign(feat, rois_batch, output_size=(1, ROIalign_size))
            
            # 将当前批次的输出保存到列表中
            outputs.append(output)

        # 将所有批次的输出堆叠在一起
        final_output = torch.cat(outputs, dim=0)
        return final_output
    
    def _temporal_pooling(self,pooling_type,coordinate,clip_feat,mask,ROIalign_size,text_feats, points):
        self.ROIalign_strategy = "before_pred"
        b,t,_ = coordinate.shape
        
        if pooling_type == "average":
            roi_feat = self._roi_align(rois=coordinate,points=points,origin_feat=clip_feat+1e-4,mask=mask,ROIalign_size=ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            # roi_feat = roi_feat.mean(-2) # [B,Q,dim]
            if self.ROIalign_strategy == "before_pred":
                roi_feat = roi_feat.mean(-2) # [B,Q,dim]

                '''# 加线性层
                roi_feat = roi_feat.reshape(-1, 512)
                roi_feat = self.temproal_linear(roi_feat)
                roi_feat = roi_feat.reshape(b, t, 512)'''

                ROIalign_logits = self._compute_similarity(roi_feat,text_feats) # [b,Q,num_classes]
            elif self.ROIalign_strategy == "after_pred":
                roi_feat = roi_feat # [B,Q,L,dim]
                ROIalign_logits = self._compute_similarity(roi_feat,text_feats) # [b,Q,L,num_classes]
                ROIalign_logits = ROIalign_logits.mean(-2) # [B,Q,num_classes]
            else:
                raise NotImplementedError
        
        elif pooling_type == "center2":
            rois = coordinate # b x t x 2(ds, de)

            '''ds = rois[:, :, 0:1]
            de = rois[:, :, 1:2]
            width = (de - ds) * points[:, :, 3:4]
            center_idx = torch.clamp((points[:, :, 0:1] + width / 2), min=0, max=2303).long()'''

            ds = rois[:, :, 0:1]
            de = rois[:, :, 1:2]
            points = points.repeat(rois.shape[0], 1, 1) # b x t x 4
            seg_left = points[:, :, 0:1] - ds * points[:, :, 3:4]   # 动作开始时间步
            seg_right = points[:, :, 0:1] + de * points[:, :, 3:4]  # 动作结束时间步
            center_idx = torch.clamp((seg_left + seg_right) / 2, min=0, max=2303).long() # 动作框中心时间步
            
            roi_feat = torch.gather(clip_feat + 1e-4, dim=1, index=center_idx.expand(-1, -1, clip_feat.shape[-1]))
            ROIalign_logits = self._compute_similarity(roi_feat, text_feats)
        else:
            raise ValueError

        return ROIalign_logits

    # @torch.no_grad()
    def _compute_similarity(self, visual_feats, text_feats):
        '''
        text_feats: [num_classes,dim]
        '''
        self.norm_embed = True
        self.exp_logit_scale = True
        if len(visual_feats.shape)==3:# batch,num_queries/snippet_length,dim
            if self.norm_embed:
                epsilon = 1e-8
                visual_feats = visual_feats / (visual_feats.norm(dim=-1,keepdim=True)+epsilon)  # 防止为0
                text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
                if self.exp_logit_scale:
                    logit_scale = self.logit_scale.exp()
                else:
                    logit_scale = self.logit_scale
                logits = torch.einsum("bqd,cd->bqc",visual_feats,text_feats)*logit_scale
            else:
                logits = torch.einsum("bqd,cd->bqc",visual_feats,text_feats)
            return logits
        
        else:
            raise NotImplementedError

    # 总参数量
    def count_parameters(self):

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"The parameter quantity of the model: {num_parameters / 1e6} M")

        return num_parameters
    
    # 各层参数量
    def count_module_parameters(self):
        total_params = 0
        # for name, param in self.backbone.stem.named_parameters():
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Parameters: {param.numel() / 1e6} M")
                total_params += param.numel()

        print(f"Total Trainable Parameters: {total_params / 1e6} M")
    
    def forward(self, video_list, label_dict=None, 
                w_tent=False, w_eata=False, w_sar=False, f_sar=False, w_deyo=False, is_techaer=False, t_out1=None, t_out2=None):
        
        '''for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Parameter Name: {name}, Shape: {param.shape}")'''
        # print('***************************************************************************************************')
        '''for name, param in self.InternVideo_model.named_parameters():
            # if param.requires_grad:
            #     print(f"Parameter Name: {name}, Shape: {param.shape}")
            param.requires_grad = True
            print(f"Parameter Name: {name}, Shape: {param.shape}, {param.requires_grad}")'''
        # self.count_module_parameters()
        # # self.count_parameters()
        # print('count_parameters')
        
        ## TTA 模型设置
        self.f_sar = f_sar
        self.w_tta = False
        if w_tent or w_eata or w_sar or w_deyo:
            self.w_tta = True
            tta_params = configure_model(self)
        
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        ## (2, 2048, T) (2, 1, T)   (2, 512, T) (2, 1, T)
        # batched_inputs, batched_masks, clip_batched_inputs, clip_batched_masks = self.preprocessing(video_list)
        batched_inputs, batched_masks = self.preprocessing(video_list)

        # forward the network (backbone -> neck -> heads)
        ## 6x(2, 512, T) 6x(2, 1, T)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        ## 6x(2, 512, T) 6x(2, 1, T) / 6x(16, 256, T) 6x(16, 1, T)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        '''cl_inputs = self.conv_clip_to_i3d(clip_batched_inputs)
        cl_feats, cl_masks = self.backbone(cl_inputs, clip_batched_masks)
        cl_fpn_feats, cl_fpn_masks = self.neck(cl_feats, cl_masks)'''

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        ## 6x(T, 4)
        points = self.point_generator(fpn_feats)

        # out_offset: List[B, 2, T_i]
        ## 6x(2, 2, T)
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        
        # out_cls: List[B, #cls + 1, T_i]
        ## 6x(2, 20, T)
        if self.use_clip:
            '''
            # old_clip
            label_list = [key for key, value in sorted(label_dict.items(), key=lambda item: item[1])]
            text = clip.tokenize(label_list).to(self.device)    # C x 77
            out_cls_logits = []
            for i in range(len(fpn_feats)):

                images = fpn_feats[i].permute(0, 2, 1).reshape(-1, fpn_feats[i].shape[1])    # (bxt) x 512
                if images.shape[1] != 512:
                    self.clip_layer.set_conv(images.shape[1])
                logits_per_image, logits_per_text = self.clip_layer(images, text)
                cls_logit = logits_per_image.reshape(fpn_feats[i].shape[0], fpn_feats[i].shape[2], -1).permute(0, 2, 1)
                
                # min_max_normaliz[-5, 5]
                cls_logit = self.min_max_normalize(cls_logit)
                mask_expanded = fpn_masks[i].expand(-1, self.num_classes, -1)
                cls_logit = cls_logit * mask_expanded.float()
                out_cls_logits.append(cls_logit)
            out_cls_logits = out_cls_logits'''

            
            # new_clip
            if self.use_tpt:
                self.cliprompt.to(self.device)
                self.cliprompt.train()
                text_feats = self.cliprompt.myforward(self.classes_name).float()
            elif self.use_tpt_stale:
                act_prompt = self.get_prompt(self.classes_name)
                texts = self.tokenizer(act_prompt, padding=True, return_tensors="pt").to(self.device)
                text_feats = self.txt_model.get_text_features(**texts)
                # self.text_feats = text_feats
            else:
                self.target_type = 'prompt'
                text_feats = self.get_text_feats(self.classes_name, self.description_dict, self.device, self.target_type)   # N_c x 512
            out_cls_logits = []
            for i in range(len(fpn_feats)):
                visual_feats = fpn_feats[i].permute(0, 2, 1)    # B x T_i x 512
                # linear_layers
                if self.linear_type in {'only_visual', 'visual_text'}:
                    B, T_i, _ = visual_feats.shape
                    visual_feats = self.visual_linear(visual_feats.reshape(-1, 512)).reshape(B, T_i, 512)   # B x T_i x 512
                if self.linear_type in {'only_text', 'visual_text'}:
                    text_feats = self.text_linear(text_feats)   # N_c x 512
                    
                if not self.training and self.onzeta:
                    if not hasattr(self, 'w') and not hasattr(self, 'rho'):
                        self.w = text_feats.clone()
                        self.rho = torch.zeros(self.num_classes).to(self.device)
                    
                    batch_feat = visual_feats
                    
                    beta = 0.5
                    lr = 1e-5
                    rlr = 1e-5
                    
                    text_label = self._compute_similarity(batch_feat, text_feats)  # B x T_i x N_c
                    text_label = text_label * torch.exp(self.rho)
                    vision_label = self._compute_similarity(batch_feat, self.w)  # B x T_i x N_c
                    cls_logit = (beta * text_label + (1 - beta) * vision_label).permute(0, 2, 1)
                    ## 更新参数rho, w
                    text_label_post = text_label / text_label.sum(dim=-1, keepdim=True)
                    grad_rho = (torch.mean(text_label_post, dim=(0,1)) - 1 / self.num_classes)
                    self.rho -= rlr * grad_rho
                    
                    grad_w = torch.outer(torch.mean(batch_feat, dim=(0,1)), torch.mean((text_label-vision_label), dim=(0,1))).T
                    self.w -= lr * grad_w
                    
                else:
                    cls_logit = self._compute_similarity(visual_feats, text_feats).permute(0, 2, 1)  # B x N_c x T_i
                    
                # cls_logit = self._compute_similarity(visual_feats, text_feats).permute(0, 2, 1)  # B x N_c x T_i    
                
                # min_max_normaliz[-5, 5]
                #cls_logit = self.min_max_normalize(cls_logit)
                cls_logit = cls_logit * fpn_masks[i].to(cls_logit.dtype)
                out_cls_logits.append(cls_logit)
            out_cls_logits = out_cls_logits
            
        # elif self.onzeta and not self.training:
        elif self.onzeta:
            if not hasattr(self, 'w') and not hasattr(self, 'rho'):
                texts = internvideo.tokenize(self.get_prompt(self.classes_name)).long().to(self.device)  # num_classes x 77
                text_feats = self.InternVideo_model.encode_text(texts)
                self.text_classifier = text_feats.permute(1, 0).float() # num_classes, D -> D, num_classes
                self.w = self.text_classifier.clone()
                self.rho = torch.zeros(self.num_classes).to(self.device)

            out_cls_logits = []
            for i in range(len(fpn_feats)):
                visual_feats = fpn_feats[i] # B x 512 x T_i
                B, _, T = visual_feats.shape
                
                cw = 0.5
                cr = 20
                beta = 0.8
                tau_t = 0.01
                tau_i = 0.04
                alpha = 1
                
                combo_label = torch.zeros(B, self.num_classes, T).to(self.device)
                text_label = torch.zeros(B, self.num_classes, T).to(self.device)
                for j in range(B):
                    for k in range(T):
                        # lr = cw / math.sqrt((j * T + k) + 1)
                        # rlr = cr / math.sqrt((j * T + k) + 1)
                        lr = 1e-5
                        rlr = 1e-5
                        beta = beta * math.sqrt(((j * T + k) + 1) / (B * T))
                        x = visual_feats[j, :, k]
                        tlabel = F.softmax(x @ self.text_classifier / tau_t, dim=0)
                        tlabel = tlabel * torch.exp(self.rho)
                        tlabel = tlabel / torch.sum(tlabel)
                        self.rho = self.rho - rlr * (tlabel - alpha / self.num_classes)
                        self.rho = torch.clamp_min(self.rho, 0)
                        text_label[j, :, k] = tlabel
                        vision_label = F.softmax(x @ self.w / tau_i, dim=0)
                        combo_label[j, :, k] = beta * vision_label + (1 - beta) * tlabel
                        grad = torch.outer(x, vision_label - tlabel)
                        self.w = self.w - (lr / tau_i) * grad
                        if torch.isnan(self.w).any():
                            print(f'{i} {j} {k} nan')
                        # self.w = F.normalize(self.w, dim=0)     
                # cls_logit = text_label
                cls_logit = combo_label
                cls_logit = cls_logit * fpn_masks[i].to(cls_logit.dtype)
                out_cls_logits.append(cls_logit)
            out_cls_logits = out_cls_logits
            print('onzeta')
        elif self.internvideo:
            if self.use_tpt_stale:
                act_prompt = self.get_prompt(self.classes_name)
                texts = self.tokenizer(act_prompt, padding=True, return_tensors="pt").to(self.device)
                text_feats = self.txt_model.get_text_features(**texts)
            else:
                if self.training:
                    texts = internvideo.tokenize(self.get_prompt(self.classes_name)).long().to(self.device)  # num_classes x 77
                    text_feats = self.InternVideo_model.encode_text(texts)
                else:
                    if not hasattr(self, '_cached_text_feats'):
                        texts = internvideo.tokenize(self.get_prompt(self.classes_name)).long().to(self.device)  # num_classes x 77
                        text_feats = self.InternVideo_model.encode_text(texts)
                        self._cached_text_feats = text_feats
                    else:
                        text_feats = self._cached_text_feats
            
            out_cls_logits = []
            for i in range(len(fpn_feats)):
                visual_feats = fpn_feats[i].permute(0, 2, 1)    # B x T_i x 512
                # cls_logit = self.InternVideo_model.myforward(visual_feats, texts).permute(0, 2, 1)  # B x num_classes x T_i
                cls_logit = self.InternVideo_model.myforward(visual_feats, text_feats).permute(0, 2, 1)  # B x num_classes x T_i
                
                # min_max_normaliz[-5, 5]
                #cls_logit = self.min_max_normalize(cls_logit)
                cls_logit = cls_logit * fpn_masks[i].to(cls_logit.dtype)
                out_cls_logits.append(cls_logit)
            out_cls_logits = out_cls_logits
        
        elif self.use_gap_clip:
            ## 6x(2, 20, T) / 6x(16, 1, T) 
            self.pooling_type = 'average'
            #self.pooling_type = 'center2'
            self.ROIalign_size = 16
            self.target_type = 'prompt'
            
            # gain text_feats
            #text_feats = torch.randn(20, 512).to(self.device)
            text_feats = self.get_text_feats(self.classes_name, self.description_dict, self.device, self.target_type) # [N classes,dim]
            
            ROIalign_logits_list = []
            if self.feats_type == 'clip_i3d':
                
                clip_feat = batched_inputs.permute(0, 2, 1)    # B x T x 512
                
                for i in range(len(out_offsets)):
                    
                    pred_boxes = out_offsets[i].permute(0, 2, 1)    # B x T x 2(ds, de)
                    point = points[i]
                    mask = batched_masks.squeeze(1)
                    # (B, Q, num_classes)2xQ2x20
                    ROIalign_logits = self._temporal_pooling(self.pooling_type, pred_boxes, clip_feat, mask, self.ROIalign_size, text_feats, point)
                    ROIalign_logits = ROIalign_logits.permute(0, 2, 1)
                    
                    ROIalign_logits = self.min_max_normalize(ROIalign_logits)
                    mask_expanded = fpn_masks[i].expand(-1, self.num_classes, -1)
                    ROIalign_logits = ROIalign_logits * mask_expanded.float()

                    ROIalign_logits_list.append(ROIalign_logits)
                out_cls_logits = ROIalign_logits_list
            elif self.feats_type == 'i3d_i3d':
                
                '''b_isd, _, t_i3d = batched_inputs.shape
                i3d_feats = batched_inputs.permute(0, 2, 1) # b x t x 2048
                i3d_feats = i3d_feats.reshape(-1, 2048) # (bxt) x 2048
                i3d_feats = self.linear_layer(i3d_feats)    # (bxt) x 512
                i3d_feats = i3d_feats.reshape(b_isd, t_i3d, 512)    # b x t x 512'''

                i3d_feats = self.conv_layer(batched_inputs).permute(0, 2, 1)    # b x t x 512

                for i in range(len(out_offsets)):

                    pred_boxes = out_offsets[i].permute(0, 2, 1)    # B x T x 2(ds, de)
                    point = points[i]
                    mask = batched_masks.squeeze(1)
                    # (B, Q, num_classes)2xQ2x20
                    ROIalign_logits = self._temporal_pooling(self.pooling_type, pred_boxes, i3d_feats, mask, self.ROIalign_size, text_feats, point)
                    ROIalign_logits = ROIalign_logits.permute(0, 2, 1)  # b x num_class x t

                    #ROIalign_logits = self.min_max_normalize(ROIalign_logits)   # [-5, 5]
                    mask_expanded = fpn_masks[i].expand(-1, self.num_classes, -1)
                    ROIalign_logits = ROIalign_logits * mask_expanded.float()

                    ROIalign_logits_list.append(ROIalign_logits)
                out_cls_logits = ROIalign_logits_list
            elif self.feats_type == 'fpn_i3d':
                
                in_feats = fpn_feats[0].permute(0, 2, 1)    # b x t x 512

                for i in range(len(out_offsets)):

                    pred_boxes = out_offsets[i].permute(0, 2, 1)    # B x T x 2(ds, de)
                    point = points[i]
                    mask = batched_masks.squeeze(1)
                    # (B, Q, num_classes)2xQ2x20
                    ROIalign_logits = self._temporal_pooling(self.pooling_type, pred_boxes, in_feats, mask, self.ROIalign_size, text_feats, point)
                    ROIalign_logits = ROIalign_logits.permute(0, 2, 1)  # b x num_class x t

                    #ROIalign_logits = self.min_max_normalize(ROIalign_logits)
                    mask_expanded = fpn_masks[i].expand(-1, self.num_classes, -1)
                    ROIalign_logits = ROIalign_logits * mask_expanded.float()

                    ROIalign_logits_list.append(ROIalign_logits)
                out_cls_logits = ROIalign_logits_list
        else:
            ## 6x(2, 20, T) / 6x(16, 1, T)
            out_cls_logits = self.cls_head(fpn_feats, fpn_masks)

        ## Memory-guided Prediction Refinement (MPR)
        if self.use_mpr:
            refined_out_offsets = []
            for i in range(len(out_offsets)):
                a = self.ttt_linear_a(out_offsets[i].permute(0, 2, 1))  # B x T x D
                refined_out_offset = self.ttt_layer(a+fpn_feats[i].permute(0, 2, 1), fpn_masks[i].permute(0, 2, 1))
                # refined_out_offset = self.ttt_layer(a, fpn_masks[i].permute(0, 2, 1))
                b = self.ttt_linear_b(refined_out_offset).permute(0, 2, 1)
                b = self.ttt_act(b)
                refined_out_offsets.append(b)
            out_offsets = refined_out_offsets
            # print('MPR finished')

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]       # b x T x 20
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]             # b x T x 2
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]                       # b x T

        ## teacher_model前向传播，返回cls_logits
        if is_techaer:
            all_out_cls_logits = torch.cat([out.view(-1, out.size(-1)) for out in out_cls_logits])
            all_out_cls_logits_mask = torch.cat([logits[mask.unsqueeze(-1).expand_as(logits)].view(-1, logits.size(-1)) 
                                                 for logits, mask in zip(out_cls_logits, fpn_masks)], dim=0)
            all_out_offsets = torch.cat([out.view(-1, out.size(-1)) for out in out_offsets])
            all_out_offsets_mask = torch.cat([logits[mask.unsqueeze(-1).expand_as(logits)].view(-1, logits.size(-1)) 
                                              for logits, mask in zip(out_offsets, fpn_masks)], dim=0)
            
            return all_out_cls_logits_mask, all_out_offsets_mask
        
        ## Inference with TTA
        if self.training and self.w_tta:
            if w_tent:
                all_out_cls_logits = torch.cat([out.view(-1, out.size(-1)) for out in out_cls_logits])
                all_out_cls_logits_mask = torch.cat([logits[mask.unsqueeze(-1).expand_as(logits)].view(-1, logits.size(-1)) 
                                                     for logits, mask in zip(out_cls_logits, fpn_masks)], dim=0)
                all_out_offsets_mask = torch.cat([logits[mask.unsqueeze(-1).expand_as(logits)].view(-1, logits.size(-1)) 
                                                  for logits, mask in zip(out_offsets, fpn_masks)], dim=0)
                # KD-TENT
                if t_out1 is not None:
                    forward_with_tent(outputs=all_out_cls_logits_mask, params=tta_params, 
                                      t_out1=t_out1, t_out2=t_out2, s_out2=all_out_offsets_mask)
                # TENT
                else:
                    forward_with_tent(outputs=all_out_cls_logits_mask, params=tta_params)

            elif w_eata:
                x = batched_inputs.view(-1, batched_inputs.shape[1])
                first_out_cls_logits = out_cls_logits[0].view(-1, out_cls_logits[0].size(-1))
                forward_with_eata(model=self, x=x, outputs=first_out_cls_logits, params=tta_params)

            elif w_sar:
                all_out_cls_logits = torch.cat([out.view(-1, out.size(-1)) for out in out_cls_logits])
                forward_with_sar(model=self, x=video_list, y=label_dict, outputs=all_out_cls_logits, params=tta_params)

            else:
                pass

            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets
            )

            return results
        
        # SAR 的第二次前向传播
        if self.training and f_sar:
            
            return torch.cat([out.view(-1, out.size(-1)) for out in out_cls_logits])
        
        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels)

            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets
            )
            return losses

        else:
            '''
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]
            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels)
            '''
            
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets
            )

            '''
            ## blue and red
            if True:
                frames = torch.sum(fpn_masks[0])

                gt_logit = gt_cls_labels[0][:frames]
                plot_classification_segments(gt_logit, self.classes_name, './plts/bar_gt.png')

                af_logit_yuan = out_cls_logits[0][0]
                #af_logit_yuan = self.min_max_normalize(af_logit_yuan)
                af_logit = af_logit_yuan.sigmoid()
                #af_logit = af_logit_yuan.softmax(-1)
                af_logit = af_logit[:frames]
                max_af_logit, _ = torch.max(af_logit, dim = 1)
                plot_classification_segments(af_logit, self.classes_name, './plts/bar_af.png')

                af_logit_pp = af_logit
                af_logit_pp[af_logit_pp < 0.001] = 0
                plot_classification_segments(af_logit_pp, self.classes_name, './plts/bar_af_pp.png')

                clip_logit_yuan = self.InternVideo_model.myforward(batched_inputs.permute(0, 2, 1), text_feats)[0]
                #clip_logit_yuan = self.min_max_normalize(clip_logit_yuan)
                clip_logit = clip_logit_yuan.sigmoid()
                #clip_logit = clip_logit_yuan.softmax(-1)
                clip_logit = clip_logit[:frames]
                max_clip_logit, _ = torch.max(clip_logit, dim = 1)
                plot_classification_segments(clip_logit, self.classes_name, './plts/bar_clip.png')
                
                plot_classification_scores(max_clip_logit, max_af_logit, './plts/bar_v0.png')

                print('hello')
            '''

            return results
        

    # 归一化
    # input: [B, num_classes, N]
    def min_max_normalize(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val) * 10 - 5 # 控制在[-5, 0]时,loss会较小,but无济于事

    '''def min_max_normalize(self, tensor):
        min_val = tensor.min(dim=-1, keepdim=True)[0]  # shape: (batch_size, num_queries, 1)
        max_val = tensor.max(dim=-1, keepdim=True)[0]  # shape: (batch_size, num_queries, 1)
        return ((tensor - min_val) / (max_val - min_val)) * 20 - 10'''

    '''def min_max_normalize(self, tensor):
        return tensor / 10'''

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        # clip_feats = [x['clip_feats'] for x in video_list]
        # clip_feats_lens = torch.as_tensor([feat.shape[-1] for feat in clip_feats])
        # clip_max_len = clip_feats_lens.max(0).values.item()

        if self.training and not self.w_tta and not self.f_sar:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)

            # assert clip_max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # # set max_len to self.max_seq_len
            # clip_max_len = self.max_seq_len
            # # batch input shape B, C, T
            # batch_shape = [len(clip_feats), clip_feats[0].shape[0], clip_max_len]
            # clip_batched_inputs = feats[0].new_full(batch_shape, padding_val)
            # for feat, pad_feat in zip(clip_feats, clip_batched_inputs):
            #     pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

            # if clip_max_len <= self.max_seq_len:
            #     clip_max_len = self.max_seq_len
            # else:
            #     # pad the input to the next divisible size
            #     stride = self.max_div_factor
            #     clip_max_len = (clip_max_len + (stride - 1)) // stride * stride
            # padding_size = [0, clip_max_len - clip_feats_lens[0]]
            # clip_batched_inputs = F.pad(
            #     clip_feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
        # clip_batched_masks = torch.arange(clip_max_len)[None, :] < clip_feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)
        # clip_batched_inputs = clip_batched_inputs.to(self.device)
        # clip_batched_masks = clip_batched_masks.unsqueeze(1).to(self.device)

        # return batched_inputs, batched_masks, clip_batched_inputs, clip_batched_masks
        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels) # 2 x 4536 x 20
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)  # 2 x 4536

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(    # num_pos x 2
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'final_loss' : final_loss}

    @torch.no_grad()
    def inference(
        self,
        video_list,
        points, fpn_masks,
        out_cls_logits, out_offsets
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
            ):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh) # test_pre_nms_thresh
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs =  torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms,
                    sigma = self.test_nms_sigma,
                    voting_thresh = self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
            
            # 4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels}
            )

        return processed_results