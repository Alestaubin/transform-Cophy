"""
Counterfactual learning

# debugging:
ipython cf_learning/main.py

K : number of objects
T : number of timesteps
B : batch size
"""
import json
from dataloaders.dataset_collision import Collision_CF
from dataloaders.dataset_blocktower import Blocktower_CF
from dataloaders.dataset_balls import Balls_CF
from cf_learning.model import CoPhyNet, CopyC
from torch import optim
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse
import ipdb
import os
from tqdm import *
from random import choice
import torch.nn.functional as F
import time
from dataloaders.utils import *

def set_seed(seed):
    """ set the random seed for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def get_losses(pred_pose_d, pred_stab_d, pred_presence_cd,
               gt_pose_d, gt_stab_d, gt_presence_cd, loss_func="mse",
               w_stab=1., w_pose=1.,
               ):
    loss_stab = F.binary_cross_entropy_with_logits(pred_stab_d, gt_stab_d)
    presence = pred_presence_cd * gt_presence_cd
    stab = (pred_stab_d > 0).float() * gt_stab_d
    T = stab.shape[1]
    binary = (1 - stab) * presence.unsqueeze(1).repeat(1, T, 1)
    loss_3d_denom = torch.sum(binary)
    if loss_3d_denom.item() > 1e-5:
        if loss_func == "mse":
            loss_3d = torch.sum(((pred_pose_d - gt_pose_d) ** 2).mean(-1) * binary) / (torch.sum(binary))  # (B,K)
        elif loss_func == "huber":        
            # beta=1.0 means it behaves like MSE for errors < 1, and L1 for errors > 1
            raw_loss = F.smooth_l1_loss(pred_pose_d, gt_pose_d, reduction='none', beta=1.0)
            
            # Average across the last dimension (x, y, z coordinates)
            raw_loss = raw_loss.mean(-1) 
            
            # Apply mask
            loss_3d = torch.sum(raw_loss * binary) / loss_3d_denom
        else:
            raise NotImplementedError(f"Loss function {loss_func} not implemented.")
    else: 
        loss_3d = torch.tensor(0.0, device=pred_pose_d.device)
    total_loss = w_stab * loss_stab + w_pose * loss_3d
    #print("total_loss",total_loss.item(),"loss_stab:", loss_stab.item(), "loss_3d:", loss_3d.item())

    return total_loss, (loss_stab, loss_3d)


def get_acc_stab(pred, gt):
    acc = 1. - torch.mean(torch.abs((pred > 0).float() - gt))
    return acc


def train_one_epoch(model, device, loader, optimizer,
                    log_file,
                    print_freq=10, D=3,
                    is_rgb=False, 
                    w_stab=1., w_pose=1.):
    model.train()

    end = time.time()
    list_acc_stab, list_mse_3d, list_total_loss, list_loss_stab, list_loss_3d = [], [], [], [], []
    loader.dataset.is_rgb = False
    
    for i, input in enumerate(tqdm(loader)):
        data_time = time.time() - end
        if is_rgb:
            rgb_ab = input['rgb_ab'].to(device)
            rgb_c = input['rgb_cd'][:,:1].to(device)
            pred_pose_d, pred_presence_cd, pred_stab_d = model(rgb_ab, rgb_c)
        else:
            pred_presence_cd = input['pred_presence_cd'].to(device)
            pred_presence_ab = input['pred_presence_ab'].to(device)
            pred_pose_cd = input['pred_pose_3D_cd'][:, :1].to(device)
            pred_pose_ab = input['pred_pose_3D_ab'].to(device)
            pred_pose_d, pred_presence_cd, pred_stab_d = model(None, None,
                                                          pred_presence_ab,
                                                          pred_pose_ab,
                                                          pred_presence_cd,
                                                          pred_pose_cd,
                                                          )


        end = time.time()

        # gt
        gt_pose_cd = input['pose_3D_cd'].to(device)
        gt_pose_d = gt_pose_cd[:, 1:]
        gt_presence_cd = input['presence_cd'].to(device)
        gt_stab_d = input['stab_cd'][:, :-1].to(device)
        # print("pred_stab_d:", pred_stab_d.shape, "gt_stab_d:", gt_stab_d.shape)
        # print("pred_pose_d:", pred_pose_d.shape, "gt_pose_d:", gt_pose_d.shape)
        # print("pred_presence_cd:", pred_presence_cd.shape, "gt_presence_cd:", gt_presence_cd.shape)
        # loss
        loss, (loss_stab, loss_3d) = get_losses(pred_pose_d, pred_stab_d, pred_presence_cd,
                             gt_pose_d, gt_stab_d, gt_presence_cd,
                             w_stab=w_stab, w_pose=w_pose)
        list_total_loss.append(loss.item())
        list_loss_stab.append(loss_stab.item())
        list_loss_3d.append(loss_3d.item())
        # backprop
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # print the loss norm 
        optimizer.step()

        end = time.time()

        if i % print_freq == 0:
            # metrics
            mse_3d = get_mse(pred_pose_d, gt_pose_d, pred_presence_cd, D=D).mean()
            acc_stab = 100. * get_acc_stab(pred_stab_d, gt_stab_d)
            list_mse_3d.append(mse_3d.item())
            list_acc_stab.append(acc_stab.item())

            print(f"{i}/{len(loader)} "
                  f"Data = {data_time:.3f} "
                  f"Loss = {loss:.4f} "
                  f"Acc_stab = {np.mean(list_acc_stab):.2f} "
                  f"MSE_3d = {np.mean(list_mse_3d):.6f}"
                  )

    # append to log file
    with open(log_file, "a+") as f:
        f.write(f"Acc_presence={np.mean(list_acc_stab):.2f} "
                f"MSE_3d={np.mean(list_mse_3d):.6f}\n")
    return list_acc_stab, list_mse_3d, list_total_loss, list_loss_stab, list_loss_3d


def get_mse(pred, gt, presence, D=3):
    T = pred.shape[1]
    dist = ((pred[:, :, :, :D] - gt[:, :, :, :D]) ** 2).mean(-1) * presence.unsqueeze(1)  # (B,T,K)
    mse = dist.sum((1, 2)) / (presence.sum(1) * T)
    return mse


def validate(model, device, loader, log_dir, log_file, print_freq=100, D=3, is_rgb=True):
    model.eval()

    end = time.time()
    list_mse_3d = []
    loader.dataset.is_rgb = is_rgb
    for i, input in enumerate(tqdm(loader)):
        data_time = time.time() - end

        # pred
        if is_rgb:
            # from RGB
            rgb_ab = input['rgb_ab'].to(device)
            rgb_c = input['rgb_cd'][:, :1].to(device)
            pred_pose_d, pred_presence_cd, stab_d = model(rgb_ab, rgb_c)
        else:
            #fro preextracted visual object properties
            pred_presence_cd = input['pred_presence_cd'].to(device)
            pred_presence_ab = input['pred_presence_ab'].to(device)
            pred_pose_cd = input['pred_pose_3D_cd'][:, :1].to(device)
            pred_pose_ab = input['pred_pose_3D_ab'].to(device)
            pred_pose_d, pred_presence_cd, stab_d = model(None, None,
                                                          pred_presence_ab,
                                                          pred_pose_ab,
                                                          pred_presence_cd,
                                                          pred_pose_cd,
                                                          )
        end = time.time()

        # gt
        gt_pose_cd = input['pose_3D_cd'].to(device)
        gt_pose_d = gt_pose_cd[:, 1:]

        # metrics
        mse_3d = get_mse(pred_pose_d, gt_pose_d, pred_presence_cd, D=D).mean()
        list_mse_3d.append(mse_3d.item())

        if i % print_freq == 0:
            print(f"{i}/{len(loader)} "
                  f"Data = {data_time:.3f} "
                  f"MSE_3d = {np.mean(list_mse_3d):.6f}"
                  )
    # append to log file
    to_write = f"MSE_3d={np.mean(list_mse_3d):.6f}\n"
    print(f"\n***Results: {to_write}***\n")
    with open(log_file, "a+") as f:
        f.write(to_write)
    return list_mse_3d


def get_dataloaders(dataset_name, dataset_dir, kwargs_loader, num_objects=3, type='normal',
                    preextracted_obj_vis_prop_dir='',
                    train_from_rgb=False,
                    evaluate_on_test_only=False):
    # choice of dataset
    if dataset_name == 'balls':
        if not evaluate_on_test_only:
            train_dataset = Balls_CF(num_balls=num_objects,
                                     root_dir=dataset_dir,
                                     split='train',
                                     is_rgb=train_from_rgb,
                                     only_cd=False,
                                     use_preextracted_object_properties=not train_from_rgb,
                                     preextracted_object_properties_dir=preextracted_obj_vis_prop_dir, #'/usr/local/google/home/fbaradel/Documents/extracted_object_properties/ballCF'
                                     )
            val_dataset = Balls_CF(num_balls=num_objects,
                                   root_dir=dataset_dir,
                                   split='val',
                                   is_rgb=train_from_rgb,
                                   only_cd=False,
                                   use_preextracted_object_properties=not train_from_rgb,
                                   preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                   )
        else:
            train_dataset, val_dataset = None, None
        test_dataset = Balls_CF(num_balls=num_objects,
                                root_dir=dataset_dir,
                                split='test',
                                is_rgb=True,
                                only_cd=False,
                                use_preextracted_object_properties=False,
                                preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                )
        D = 2
    elif dataset_name == 'collision':
        if not evaluate_on_test_only:
            train_dataset = Collision_CF(type=type,
                                         root_dir=dataset_dir,
                                         split='train',
                                         is_rgb=train_from_rgb,
                                         only_cd=False,
                                         use_preextracted_object_properties=not train_from_rgb,
                                         preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                         )
            val_dataset = Collision_CF(type=type,
                                       root_dir=dataset_dir,
                                       split='val',
                                       is_rgb=train_from_rgb,
                                       only_cd=False,
                                       use_preextracted_object_properties=not train_from_rgb,
                                       preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                       )
        else:
            train_dataset, val_dataset = None, None
        test_dataset = Collision_CF(type=type,
                                    root_dir=dataset_dir,
                                    split='test',
                                    is_rgb=True,
                                    only_cd=False,
                                    use_preextracted_object_properties=False,
                                    preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                    )
        D = 3
    elif dataset_name == 'blocktower':
        if not evaluate_on_test_only:
            train_dataset = Blocktower_CF(type=type,
                                          num_blocks=num_objects,
                                          root_dir=dataset_dir,
                                          split='train',
                                          is_rgb=train_from_rgb,
                                          only_cd=False,
                                          use_preextracted_object_properties=not train_from_rgb,
                                          preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                          )
            val_dataset = Blocktower_CF(type=type,
                                        num_blocks=num_objects,
                                        root_dir=dataset_dir,
                                        split='val',
                                        is_rgb=train_from_rgb,
                                        only_cd=False,
                                        use_preextracted_object_properties=not train_from_rgb,
                                        preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                        )
        else:
            train_dataset, val_dataset = None, None
        test_dataset = Blocktower_CF(type=type,
                                     num_blocks=num_objects,
                                     root_dir=dataset_dir,
                                     split='test',
                                     is_rgb=True,
                                     only_cd=False,
                                     use_preextracted_object_properties=False,
                                     preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                     )
        D = 3
    else:
        raise NameError('Unkown dataset name.')

    # loader
    if not evaluate_on_test_only:
        train_loader = DataLoader(train_dataset, shuffle=True, **kwargs_loader)
        kwargs_loader_val = kwargs_loader.copy()
        kwargs_loader_val['batch_size'] = 8
        val_loader = DataLoader(val_dataset, **kwargs_loader_val)
        test_loader = DataLoader(test_dataset, **kwargs_loader_val)
        return train_loader, val_loader, test_loader, D
    else:
        kwargs_loader_val = kwargs_loader.copy()
        kwargs_loader_val['batch_size'] = 8
        test_loader = DataLoader(test_dataset, **kwargs_loader_val)
        return None, None, test_loader, D


def get_trainable_params(model):
    """ get list of parameters to train of a network """
    trainable_params = []
    for name_c, child in model.named_children():
        for name_p, param in child.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

    return trainable_params


def main(args):
    # write all params to a json file in the log dir
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    if args.seed is not None:
        set_seed(args.seed)
    # kwargs
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    kwargs_loader = {'batch_size': args.batch_size}
    if device.type in ['cuda', 'mps']:
        kwargs_loader.update({'num_workers': args.workers, 'pin_memory': True})

    # datasets and loaders
    train_loader, val_loader, test_loader, D = get_dataloaders(args.dataset_name,
                                                               args.dataset_dir,
                                                               kwargs_loader,
                                                               args.num_objects,
                                                               args.type,
                                                               preextracted_obj_vis_prop_dir=args.preextracted_obj_vis_prop_dir,
                                                               train_from_rgb=args.train_from_rgb,
                                                               evaluate_on_test_only=args.evaluate,
                                                               )

    # model
    dict_model_fn = {'copy_c': CopyC, 'cophynet': CoPhyNet}
    model_fn = dict_model_fn[args.model]
    model = model_fn(num_objects=test_loader.dataset.num_objects, encoder_type=args.encoder_type).to(device)

    # freeze derendering
    for param in model.derendering.parameters():
        param.requires_grad = False
    
    history = {
        'train_acc_stab': [],
        'train_mse_3d': [],
        'train_total_loss': [],
        'train_loss_stab': [],
        'train_loss_3d': [],
        'val_mse_3d': [],
        'epoch_time': [],
    }

    # optim and training
    if len(get_trainable_params(model)) > 0 and not args.evaluate:
        # load the derendering module
        pretrained_dict = torch.load(args.derendering_ckpt, map_location=torch.device('cpu'))
        pretrained_dict = {'derendering.' + k: v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # training
        os.makedirs(args.log_dir, exist_ok=True)
        log_file_val = os.path.join(args.log_dir, 'val.txt')
        log_file_train = os.path.join(args.log_dir, 'train.txt')
        for epoch in range(1, args.epochs):
            print(f"\n=== Epoch: {epoch}/{args.epochs} ===")
            print(f"Training...")
            start_time = time.time()
            list_acc_stab, list_mse_3d, list_total_loss, \
                list_loss_stab, list_loss_3d = train_one_epoch(model, device, train_loader, 
                                                               optimizer, log_file_train, D=D, 
                                                               w_stab=args.w_stab, w_pose=args.w_pose)
            end_time = time.time()
            epoch_time = end_time - start_time
            history['epoch_time'].append(epoch_time)
            history['train_acc_stab'].append(float(np.mean(list_acc_stab)))
            history['train_mse_3d'].append(float(np.mean(list_mse_3d)))
            history['train_total_loss'].append(float(np.mean(list_total_loss)))
            history['train_loss_stab'].append(float(np.mean(list_loss_stab)))
            history['train_loss_3d'].append(float(np.mean(list_loss_3d)))
            json_path = os.path.join(args.log_dir, 'metrics_history.json')
            with open(json_path, 'w') as f:
                json.dump(history, f, indent=4)

            log_dir_val = os.path.join(args.log_dir, 'vizu_val', f"{epoch:02d}")
            os.makedirs(log_dir_val, exist_ok=True)
            print(f"Validation...")
            list_mse_3d = validate(model, device, val_loader, log_dir_val, log_file_val)
            history['val_mse_3d'].append(float(np.mean(list_mse_3d)))
            with open(json_path, 'w') as f:
                json.dump(history, f, indent=4)
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'model_state_dict.pt'))

    # testing
    if args.evaluate:
        if len(get_trainable_params(model)) > 0:
            print("Loading pretrained checkpoint for evaluation...")
            # load ckpt
            model.load_state_dict(torch.load(args.pretrained_ckpt, map_location=torch.device('cpu')), strict=True)
        else:
            # only load the rendering
            pretrained_dict = torch.load(args.derendering_ckpt, map_location=torch.device('cpu'))
            pretrained_dict = {'derendering.' + k: v for k, v in pretrained_dict.items()}
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model.load_state_dict(pretrained_dict, strict=False)
        
        model.to(device)

        os.makedirs(args.log_dir, exist_ok=True)
        log_file_test = os.path.join(args.log_dir, 'test.txt')
        log_dir_test = os.path.join(args.log_dir, 'vizu_test', f"00")
        val_mse_3d_list = validate(model, device, test_loader, log_dir_test, log_file_test, D=D, is_rgb=True)
        
        history['val_mse_3d'].append(float(np.mean(val_mse_3d_list)))
        with open(json_path, 'w') as f:
            json.dump(history, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of the derendering module.')
    parser.add_argument('--dataset_dir',
                        # default='/usr/local/google/home/fbaradel/Dataset/CoPhy_224/ballsCF',
                        # default='/usr/local/google/home/fbaradel/Dataset/CoPhy_224/collisionCF',
                        default='/usr/local/google/home/fbaradel/Dataset/CoPhy_224/blocktowerCF',
                        type=str,
                        help='Location of the data.')
    parser.add_argument('--num_objects',
                        default=3,
                        type=int,
                        help='Number of objects for training.')
    parser.add_argument('--type',
                        default='normal',
                        type=str,
                        help='Type of train/val/test split.')
    parser.add_argument('--derendering_ckpt',
                        # default='/usr/local/google/home/fbaradel/Documents/log_dir/ballsCF/model_state_dict.pt',
                        default='/usr/local/google/home/fbaradel/Documents/log_dir/blocktowerCF/model_state_dict.pt',
                        type=str,
                        help='Location of the pre-trained derendering module.')
    parser.add_argument('--log_dir',
                        default='/tmp/cophy_cf_learning',
                        type=str,
                        help='Location of the log dir.')
    parser.add_argument('--dataset_name',
                        # default='balls',
                        # default='collision',
                        default='blocktower',
                        type=str,
                        help='Which dataset to take (balls, collision, blocktower).')
    parser.add_argument('--model',
                        # default='copy_c',
                        # default='copy_b',
                        default='cophynet',
                        type=str,
                        help='Model name to use.')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size.')
    parser.add_argument('--workers', default=8, type=int, help='Workers.')
    parser.add_argument('--epochs', default=20, type=int, help='Num epochs.')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate.')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--no-evaluate', dest='evaluate', action='store_false')
    parser.set_defaults(evaluate=False)
    parser.add_argument('--train-from-rgb', dest='train_from_rgb', action='store_true')
    parser.add_argument('--no-train-from-rgb', dest='train_from_rgb', action='store_false')
    parser.set_defaults(train_from_rgb=False)
    parser.add_argument('--pretrained_ckpt',
                        default='/usr/local/google/home/fbaradel/Documents/log_dir/blocktowerCF/model_state_dict.pt',
                        type=str,
                        help='Location of the pre-trained derendering module.')
    parser.add_argument('--preextracted_obj_vis_prop_dir',
                        default='',
                        type=str,
                        help='Location of the pre-extracted object visual properties.')
    parser.add_argument('--encoder_type',
                        default='rnn',
                        type=str,
                        help='Type of encoder to use (rnn or transformer).')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--w_stab',
                        default=1.,
                        type=float,
                        help='Stability weight.')
    parser.add_argument('--w_pose',
                        default=1.,
                        type=float,
                        help='3D pose weight.')
    args = parser.parse_args()

    main(args)
