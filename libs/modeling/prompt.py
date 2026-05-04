from .clip_eff import clip
import torch
import numpy as np
from collections import OrderedDict
import json


def convert_to_token(xh):
    xh_id = clip.tokenize(xh).cpu().data.numpy()
    return xh_id

def text_prompt(actionlist, num_classes, device, clipbackbone='ViT-B/16'):

    actionprompt, actiontoken = {}, []

    # load the CLIP model
    clipmodel, _ = clip.load(clipbackbone, device=device, jit=False)
    for paramclip in clipmodel.parameters():
        paramclip.requires_grad = False

    # convert to token, will automatically padded to 77 with zeros
    actionlist = np.array([a for a in actionlist])
    actiontoken = np.array([convert_to_token(a) for a in actionlist])
    # More datasets to be continued

    # query the vector from dictionary
    with torch.no_grad():
        actionembed = clipmodel.encode_text_light(torch.tensor(actiontoken).to(device))

    actiondict = OrderedDict((actionlist[i], actionembed[i].cpu().data.numpy()) for i in range(num_classes))
    actiontoken = OrderedDict((actionlist[i], actiontoken[i]) for i in range(num_classes))

    return actionlist, actiondict, actiontoken

