import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, TransformerBlock_TTT, Trans_TTT_Block, MaskedConv1D,
                     ConvBlock, LayerNorm)

from .ttt_model.ttt_encoder_v4 import TTT_Encoder_Block

@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        self.n_embd = n_embd
        self.n_head = n_head

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def set_paparameters(self, use_ttt, ttt_type, bi_ttt_type, mini_batch_size, window_size, 
                         ttt_pos, encoder_version, num_ttt_encoders, ar_pred, device):
        
        self.use_ttt = use_ttt
        self.device = device
        
        if self.use_ttt:
            self.ttt_type = ttt_type
            self.bi_ttt_type = bi_ttt_type
            self.mini_batch_size = mini_batch_size
            self.window_size = window_size
            self.ttt_pos = ttt_pos
            self.encoder_version = encoder_version
            self.num_ttt_encoders = num_ttt_encoders
            self.ar_pred = ar_pred
            
            if self.encoder_version == 'v0':
                # stem network using (vanilla) transformer
                self.stem = nn.ModuleList()
                for idx in range(self.arch[1]): # x2
                    self.stem.append(
                        TransformerBlock_TTT(
                            self.n_embd, self.n_head,
                            n_ds_strides=(1, 1),
                            attn_pdrop=0.0,
                            proj_pdrop=0.0,
                            path_pdrop=0.0,
                            mha_win_size=self.mha_win_size[0],
                            use_rel_pe=self.use_rel_pe, 
                            ttt_type=self.ttt_type, 
                            bi_ttt_type=self.bi_ttt_type, 
                            mini_batch_size = self.mini_batch_size, 
                            window_size = self.window_size, 
                            ar_pred = self.ar_pred
                        ).to(self.device)
                    )

                # main branch using transformer with pooling
                self.branch = nn.ModuleList()
                for idx in range(min(self.num_ttt_encoders-1, self.arch[2])): # x5
                    self.branch.append(
                        TransformerBlock_TTT(
                            self.n_embd, self.n_head,
                            n_ds_strides=(self.scale_factor, self.scale_factor),
                            attn_pdrop=0.0,
                            proj_pdrop=0.0,
                            path_pdrop=0.0,
                            mha_win_size=self.mha_win_size[1 + idx],
                            use_rel_pe=self.use_rel_pe, 
                            ttt_type=self.ttt_type, 
                            bi_ttt_type=self.bi_ttt_type, 
                            mini_batch_size = self.mini_batch_size, 
                            window_size = self.window_size, 
                            ar_pred = self.ar_pred
                        ).to(self.device)
                    )
                for idx in range(min(self.num_ttt_encoders-1, self.arch[2]), self.arch[2]):
                    self.branch.append(
                        TransformerBlock(
                            self.n_embd, self.n_head,
                            n_ds_strides=(self.scale_factor, self.scale_factor),
                            attn_pdrop=0.0,
                            proj_pdrop=0.0,
                            path_pdrop=0.0,
                            mha_win_size=self.mha_win_size[1 + idx],
                            use_rel_pe=self.use_rel_pe
                        ).to(self.device)
                    )
            elif self.encoder_version == 'v1':
                # stem network using (vanilla) transformer
                self.stem = nn.ModuleList()
                for idx in range(self.arch[1]): # x2
                    self.stem.append(
                        Trans_TTT_Block(
                            self.n_embd, self.n_head,
                            n_ds_strides=(1, 1),
                            attn_pdrop=0.0,
                            proj_pdrop=0.0,
                            path_pdrop=0.0,
                            mha_win_size=self.mha_win_size[0],
                            use_rel_pe=self.use_rel_pe, 
                            ttt_type=self.ttt_type, 
                            bi_ttt_type=self.bi_ttt_type, 
                            mini_batch_size = self.mini_batch_size, 
                            window_size = self.window_size, 
                            ttt_pos = self.ttt_pos
                        ).to(self.device)
                    )

                # main branch using transformer with pooling
                self.branch = nn.ModuleList()
                for idx in range(min(self.num_ttt_encoders-1, self.arch[2])): # x5
                    self.branch.append(
                        Trans_TTT_Block(
                            self.n_embd, self.n_head,
                            n_ds_strides=(self.scale_factor, self.scale_factor),
                            attn_pdrop=0.0,
                            proj_pdrop=0.0,
                            path_pdrop=0.0,
                            mha_win_size=self.mha_win_size[1 + idx],
                            use_rel_pe=self.use_rel_pe, 
                            ttt_type=self.ttt_type, 
                            bi_ttt_type=self.bi_ttt_type, 
                            mini_batch_size = self.mini_batch_size, 
                            window_size = self.window_size, 
                            ttt_pos = self.ttt_pos
                        ).to(self.device)
                    )
                for idx in range(min(self.num_ttt_encoders-1, self.arch[2]), self.arch[2]):
                    self.branch.append(
                        TransformerBlock(
                            self.n_embd, self.n_head,
                            n_ds_strides=(self.scale_factor, self.scale_factor),
                            attn_pdrop=0.0,
                            proj_pdrop=0.0,
                            path_pdrop=0.0,
                            mha_win_size=self.mha_win_size[1 + idx],
                            use_rel_pe=self.use_rel_pe
                        ).to(self.device)
                    )

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection    特征投影
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        # embedding network 嵌入网络    Nan!!
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # training: using fixed length position embeddings  位置嵌入【训练】
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences   位置嵌入【推理】
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):   # len(self.stem) = 2
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)): # len(self.branch) = 5
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks


@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """
    def __init__(
        self,
        n_in,               # input feature dimension
        n_embd,             # embedding dimension (after convolution)
        n_embd_ks,          # conv kernel size of the embedding network
        arch = (2, 2, 5),   # (#convs, #stem convs, #branch convs)
        scale_factor = 2,   # dowsampling rate for the branch
        with_ln=False,      # if to use layernorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.n_in = n_in
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using convs
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))

        # main branch using convs with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks