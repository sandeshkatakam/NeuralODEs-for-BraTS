import os
import sys

import torch
from torch.optim import SGD, Adam

from .neural_ode_convnet import *
from .neural_ode_unet import *
from .unet import *

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import config
from utils import *


def load_model(params, class_num, mode):

    # Device Init
    device = config.device

    # Model Init
    if params["model"] == 'unet':
        net = UNet(params["in_channel"], class_num, drop_rate=params["drop_rate"])
    elif params["model"] == 'neural_ode_convnet':
        net = neural_ode_convnet()
    elif params["model"] == 'neural_ode_unet':
        net = neural_ode_unet()
    else:
        raise ValueError('params["model"] ERROR')

    # Optimizer Init
    if mode == 'train':
        resume = params["resume"]
        #optimizer = Adam(net.parameters(), lr=args.lr)
        optimizer = SGD(net.parameters(), lr=params["lr"], momentum=0.9, weight_decay=1e-4)
    elif mode == 'test':
        resume = True
        optimizer = None
    else:
        raise ValueError('load_model mode ERROR')

    # Model Load
    if resume:
        checkpoint = Checkpoint(net, optimizer)
        checkpoint.load(os.path.join(params["ckpt_root"], params["model"]+'.tar'))
        best_score = checkpoint.best_score
        start_epoch = checkpoint.epoch+1
    else:
        best_score = 0
        start_epoch = 1

    if device == 'cuda':
        net.cuda()
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark=True

    return net, optimizer, best_score, start_epoch






####################### Old Version ################################
# def load_model(args, class_num, mode):

#     # Device Init
#     device = config.device

#     # Model Init
#     if args.model == 'unet':
#         net = UNet(args.in_channel, class_num, drop_rate=args.drop_rate)
#     elif args.model == 'neural_ode_convnet':
#         net = neural_ode_convnet()
#     elif args.model == 'neural_ode_unet':
#         net = neural_ode_unet()
#     else:
#         raise ValueError('args.model ERROR')

#     # Optimizer Init
#     if mode == 'train':
#         resume = args.resume
#         #optimizer = Adam(net.parameters(), lr=args.lr)
#         optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
#     elif mode == 'test':
#         resume = True
#         optimizer = None
#     else:
#         raise ValueError('load_model mode ERROR')

#     # Model Load
#     if resume:
#         checkpoint = Checkpoint(net, optimizer)
#         checkpoint.load(os.path.join(args.ckpt_root, args.model+'.tar'))
#         best_score = checkpoint.best_score
#         start_epoch = checkpoint.epoch+1
#     else:
#         best_score = 0
#         start_epoch = 1

#     if device == 'cuda':
#         net.cuda()
#         net = torch.nn.DataParallel(net)
#         torch.backends.cudnn.benchmark=True

#     return net, optimizer, best_score, start_epoch