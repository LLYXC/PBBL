import os
import pickle
from tqdm import tqdm
from datetime import datetime
import random
import logging

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,\
    recall_score, f1_score, average_precision_score, roc_curve

import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, WeightedRandomSampler
from torch.utils.data.dataset import Subset
from torchvision import transforms as T
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from config import ex
from data.util import get_dataset, IdxDataset, ZippedDataset, ConfounderDataset, get_confusion_matrix
from module.util import get_model
from module.loss import GeneralizedCELoss
from util import MultiDimAverageMeter, set_logging

import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_tag', type=str, default='DenseNet121', help="Use densenet121 as the backbone of our model.")
    parser.add_argument('--data_dir', type=str, default='./dataset', help="Address where you store the csv files.")
    parser.add_argument('--idle', type=bool, default=False, help="False for saving result or Ture for not saving result.")
    parser.add_argument('--dataset_tag', type=str, default='MIMIC_CXR', help="Gender_pneumothorax, MIMIC_CXR")
    parser.add_argument('--log_dir', type=str, default='./log', help="Address to store the log files.")
    parser.add_argument('--case', type=str, default='_case1', help="_case1, _case2, case_3")
    parser.add_argument('--seed', type=str, default='-Seed2', help="-Seed1, -Seed2, -Seed3")
    parser.add_argument('--main_optimizer_tag', type=str, default='Adam', help="Adam, AdamW")
    parser.add_argument('--device', type=int, default=0, help="0, 1")
    parser.add_argument('--target_attr_idx', type=int, default=0, help="0, 1")
    parser.add_argument('--bias_attr_idx', type=int, default=1, help="0, 1")
    parser.add_argument('--main_valid_freq', type=int, default=10, help="valid frequency")
    parser.add_argument('--main_batch_size', type=int, default=256)
    parser.add_argument('--main_learning_rate', type=int, default=1e-4)
    parser.add_argument('--main_weight_decay', type=int, default=5e-4)
    parser.add_argument('--main_num_steps', type=int, default=1000, help = "1000 for GbP, 1360 for SbP")
    parser.add_argument('--repeat_times', type=int, default=1, help = "2 for GbP, 1 for SbP")

    args = parser.parse_args()

    main_tag = args.dataset_tag + args.case + args.seed
    dataset_tag = args.dataset_tag + args.case
    log_dir = os.path.join(args.log_dir, args.dataset_tag)

    if "Seed1" in main_tag:
        seed = 1
    elif "Seed2" in main_tag:
        seed = 2
    elif "Seed3" in main_tag:
        seed = 3
    else:
        raise NotImplementedError
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    pretrain_num_steps = 1000

    device = torch.device(args.device)
    if not args.idle:
        print('saving the result.')
        exp_name = 'rs_bb_unknown_iterative_{}'.format(pretrain_num_steps)
        log_dir = log_dir + '_' + exp_name + '-seed_{}-'.format(seed)
        writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag))
        # set_logging(exp_name, 'INFO', str(log_dir))

    train_dataset = get_dataset(
        dataset_tag,
        data_dir=args.data_dir,
        dataset_split="train",
        transform_split="train"
    )

    valid_dataset = get_dataset(
        dataset_tag,
        data_dir=args.data_dir,
        dataset_split="eval",
        transform_split="eval"
    )
    test_dataset = get_dataset(
        dataset_tag,
        data_dir=args.data_dir,
        dataset_split="test",
        transform_split="test"
    )
    test_dataset = IdxDataset(test_dataset)

    train_target_attr = train_dataset.attr[:, args.target_attr_idx]
    train_bias_attr = train_dataset.attr[:, args.bias_attr_idx]
    attr_dims = []
    attr_dims.append(torch.max(train_target_attr).item() + 1)
    attr_dims.append(torch.max(train_bias_attr).item() + 1)
    num_classes = attr_dims[0]
    train_dataset = IdxDataset(train_dataset)
    valid_dataset = IdxDataset(valid_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.main_batch_size*2,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.main_batch_size*2,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    # define model and optimizer
    mlp_classifier = True
    model = get_model(args.model_tag, num_classes, mlp_classifier=mlp_classifier).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.main_learning_rate,
        weight_decay=args.main_weight_decay,
        betas=(0.9, 0.999)
    )

    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    label_criterion = torch.nn.CrossEntropyLoss(reduction="none")
    bias_criterion = GeneralizedCELoss()

    # define evaluation function
    def evaluate_acc_ap_auc(model, data_loader):
        model.eval()
        gts = torch.LongTensor().to(device)
        bias_gts = torch.LongTensor().to(device)
        probs = torch.FloatTensor().to(device)
        attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for index, data, attr in tqdm(data_loader, leave=False):
            label = attr[:, args.target_attr_idx]
            bias_label = attr[:, args.bias_attr_idx]
            data = data.to(device)
            attr = attr.to(device)
            label = label.to(device)
            bias_label = bias_label.to(device)
            with torch.no_grad():
                logit = model(data)
                prob = torch.softmax(logit, dim=1)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
            gts = torch.cat((gts, label), 0)
            bias_gts = torch.cat((bias_gts, bias_label), 0)
            probs = torch.cat((probs, prob), 0)

            attr = attr[:, [args.target_attr_idx, args.bias_attr_idx]]
            attrwise_acc_meter.add(correct.cpu(), attr.cpu())

        accs = attrwise_acc_meter.get_mean()

        gts_numpy = gts.cpu().detach().numpy()
        probs_numpy = probs.cpu().detach().numpy()
        bias_gts_numpy = bias_gts.cpu().detach().numpy()

        # overall auc and ap
        aps, aucs = [], []
        aps.append(average_precision_score(gts_numpy, probs_numpy[:, 1]))
        aucs.append(roc_auc_score(gts_numpy, probs_numpy[:, 1]))
        # aligned auc and ap
        idx1 = np.where((bias_gts_numpy == 0) & (gts_numpy == 0))
        idx2 = np.where((bias_gts_numpy == 1) & (gts_numpy == 1))
        aps.append(average_precision_score(np.concatenate([gts_numpy[idx1], gts_numpy[idx2]]),
                                           np.concatenate([probs_numpy[idx1][:, 1], probs_numpy[idx2][:, 1]])))
        aucs.append(roc_auc_score(np.concatenate([gts_numpy[idx1], gts_numpy[idx2]]),
                                           np.concatenate([probs_numpy[idx1][:, 1], probs_numpy[idx2][:, 1]])))
        # conflict auc and ap
        idx1 = np.where((bias_gts_numpy == 0) & (gts_numpy == 1))
        idx2 = np.where((bias_gts_numpy == 1) & (gts_numpy == 0))
        aps.append(average_precision_score(np.concatenate([gts_numpy[idx1], gts_numpy[idx2]]),
                                           np.concatenate([probs_numpy[idx1][:, 1], probs_numpy[idx2][:, 1]])))
        aucs.append(roc_auc_score(np.concatenate([gts_numpy[idx1], gts_numpy[idx2]]),
                                  np.concatenate([probs_numpy[idx1][:, 1], probs_numpy[idx2][:, 1]])))

        model.train()

        return accs, aps, aucs

    # ------------------------------------- #
    # iterative update the pseudo bias label
    # ------------------------------------- #

    # Initialize pseudo bias label
    new_attr = torch.stack((train_dataset.dataset.attr[:, args.target_attr_idx], train_dataset.dataset.attr[:, args.target_attr_idx]), dim=1)
    train_dataset.dataset.attr = new_attr

    for bias_iteration in range(args.repeat_times):

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.main_batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

        train_iter_pre = None

        # pre-training biased model with GCE loss
        model_b_pre = get_model(args.model_tag, num_classes, mlp_classifier=mlp_classifier).to(device)
        optimizer_b_pre = torch.optim.Adam(
            model_b_pre.parameters(),
            lr=args.main_learning_rate,
            weight_decay=args.main_weight_decay,
            betas=(0.9, 0.999)
        )
        model_b_pre.train()

        for step in tqdm(range(pretrain_num_steps)):
            try:
                index, data, attr = next(train_iter_pre)
            except:
                train_iter_pre = iter(train_loader)
                index, data, attr = next(train_iter_pre)

            data = data.to(device)
            attr = attr.to(device)
            label = attr[:, args.bias_attr_idx]

            logit_b_pre = model_b_pre(data)

            loss_b_update = bias_criterion(logit_b_pre, label).mean()

            optimizer_b_pre.zero_grad()
            loss_b_update.backward()
            optimizer_b_pre.step()

        model_b_pre.eval()

        # update the pseudo bias label
        gts = torch.zeros(train_target_attr.size(0)).long().to(device)
        probs_b = torch.zeros(train_target_attr.size(0), num_classes).to(device)
        bias_score = torch.zeros(train_target_attr.size(0)).to(device)
        preds_b = torch.zeros(train_target_attr.size(0)).to(device)
        for step in tqdm(range(len(train_loader))):
            try:
                index, data, attr = next(train_iter_pre)
            except:
                train_iter_pre = iter(train_loader)
                index, data, attr = next(train_iter_pre)
            data = data.to(device)
            attr = attr.to(device)
            label = attr[:, args.target_attr_idx]
            with torch.no_grad():
                logit_b = model_b_pre(data)
                prob_b = torch.softmax(logit_b, dim=1)
                pred_b = logit_b.data.max(1, keepdim=True)[1].squeeze(1)
                correct_b = (pred_b == label).long()
                preds_b[index] = pred_b * 1.
            gts[index] = label
            probs_b[index] = prob_b
            bias_score[index] = torch.abs(correct_b - prob_b.max(1)[0])

        gts_numpy = gts.cpu().detach().numpy()
        probs_b_numpy = probs_b.cpu().detach().numpy()

        print('train_auc: ', roc_auc_score(gts_numpy, probs_b_numpy[:, 1]))

        # thre by auc
        fpr, tpr, threshold = roc_curve(gts_numpy, probs_b_numpy[:, 1])
        youden = tpr + 1-fpr
        th = threshold[np.where(youden==youden.max())][0]
        pseudo_bias = (probs_b_numpy[:, 1] > th) * 1.
        pseudo_bias = torch.from_numpy(pseudo_bias)
        new_attr = torch.stack((train_dataset.dataset.attr[:, args.target_attr_idx], pseudo_bias.long()), dim=1)
        train_dataset.dataset.attr = new_attr

    confusion_matrix_org, confusion_matrix, confusion_matrix_by = get_confusion_matrix(num_classes=2,
                                                                                       targets=train_dataset.dataset.attr[
                                                                                               :, args.target_attr_idx],
                                                                                       biases=train_dataset.dataset.attr[
                                                                                              :, args.bias_attr_idx])
    print('after bias learning: ', confusion_matrix_org)
    print('after bias learning: ', confusion_matrix)

    confusion_matrix = confusion_matrix.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    train_iter = None

    valid_attrwise_accs_list = []

    # ------------------------------------------------------- #
    # train the debiased model based on the pseudo bias labels
    # ------------------------------------------------------- #

    for step in tqdm(range(args.main_num_steps)):
        try:
            index, data, attr = next(train_iter)
        except:
            train_iter = iter(train_loader)
            index, data, attr = next(train_iter)

        data = data.to(device)
        attr = attr.to(device)

        label = attr[:, args.target_attr_idx]
        pseudo_biases = attr[:, args.bias_attr_idx]

        logit = model(data)

        # for bias balanced loss
        prior = confusion_matrix[pseudo_biases]
        logit += torch.log(prior + 1e-9)

        loss_per_sample = label_criterion(logit.squeeze(1), label)
        loss = loss_per_sample.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step % args.main_valid_freq == 0):  # and (not args.idle):
            valid_attrwise_accs, valid_attrwise_aps, valid_attrwise_aucs = evaluate_acc_ap_auc(model, valid_loader)
            valid_attrwise_accs_list.append(valid_attrwise_accs)
            valid_accs = torch.mean(valid_attrwise_accs)
            writer.add_scalar("acc/valid", valid_accs, step)
            eye_tsr = torch.eye(num_classes)
            writer.add_scalar(
                "acc/valid_aligned",
                valid_attrwise_accs[eye_tsr > 0.0].mean(),
                step
            )
            writer.add_scalar(
                "acc/valid_skewed",
                valid_attrwise_accs[eye_tsr == 0.0].mean(),
                step
            )
            writer.add_scalar("ap/valid", valid_attrwise_aps[0], step)
            writer.add_scalar("ap/valid_aligned", valid_attrwise_aps[1], step)
            writer.add_scalar("ap/valid_skewed", valid_attrwise_aps[2], step)
            writer.add_scalar("ap/valid_balanced", (valid_attrwise_aps[1]+valid_attrwise_aps[2]) / 2, step)
            writer.add_scalar("auc/valid", valid_attrwise_aucs[0], step)
            writer.add_scalar("auc/valid_aligned", valid_attrwise_aucs[1], step)
            writer.add_scalar("auc/valid_skewed", valid_attrwise_aucs[2], step)
            writer.add_scalar("auc/valid_balanced", (valid_attrwise_aucs[1]+valid_attrwise_aucs[2]) / 2, step)

            # test
            test_attrwise_accs, test_attrwise_aps, test_attrwise_aucs = evaluate_acc_ap_auc(model, test_loader)
            test_accs = torch.mean(test_attrwise_accs)
            writer.add_scalar("acc/test", test_accs, step)
            eye_tsr = torch.eye(num_classes)
            writer.add_scalar(
                "acc/test_aligned",
                test_attrwise_accs[eye_tsr > 0.0].mean(),
                step
            )
            writer.add_scalar(
                "acc/test_skewed",
                test_attrwise_accs[eye_tsr == 0.0].mean(),
                step
            )
            writer.add_scalar("ap/test", test_attrwise_aps[0], step)
            writer.add_scalar("ap/test_aligned", test_attrwise_aps[1], step)
            writer.add_scalar("ap/test_skewed", test_attrwise_aps[2], step)
            writer.add_scalar("ap/test_balanced", (test_attrwise_aps[1] + test_attrwise_aps[2]) / 2, step)
            writer.add_scalar("auc/test", test_attrwise_aucs[0], step)
            writer.add_scalar("auc/test_aligned", test_attrwise_aucs[1], step)
            writer.add_scalar("auc/test_skewed", test_attrwise_aucs[2], step)
            writer.add_scalar("auc/test_balanced", (test_attrwise_aucs[1] + test_attrwise_aucs[2]) / 2, step)

    os.makedirs(os.path.join(log_dir, "result", main_tag), exist_ok=True)
    result_path = os.path.join(log_dir, "result", main_tag, "result.th")
    valid_attrwise_accs_list = torch.cat(valid_attrwise_accs_list)
    with open(result_path, "wb") as f:
        torch.save({"valid/attrwise_accs": valid_attrwise_accs_list}, f)
    model_path = os.path.join(log_dir, "result", main_tag, "model.th")
    state_dict = {
        'steps': step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    with open(model_path, "wb") as f:
        torch.save(state_dict, f)

if __name__ == '__main__':
    train()


