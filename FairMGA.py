from utils.utils import setup_seed
from dataset.av_dataset import AVDataset_CD
import copy
from torch.utils.data import DataLoader
from models.models import AVClassifier
from sklearn import metrics
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import re
from min_norm_solvers import MinNormSolver
import numpy as np
from tqdm import tqdm
import argparse
import os
import pickle
from operator import mod
from weight_methods import WeightMethods
from models.mbt import MBTClassifier
import torch.nn.functional as F
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='KineticSound, CREMAD')
    parser.add_argument('--model', default='model', type=str)
    parser.add_argument('--n_classes', default=6, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--optimizer', default='sgd',
                        type=str, choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=30, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1,
                        type=float, help='decay coefficient')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--ckpt_path', default='log_cd',
                        type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--clip_grad', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--use_tensorboard', default=True,
                        type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='log_cd',
                        type=str, help='path to save tensorboard logs')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0,1,2,3',
                        type=str, help='GPU ids')
    parser.add_argument('--fair_alpha', default=1.0, type=float, help='FairGrad alpha')
    parser.add_argument('--fair_max_norm', default=1.0, type=float, help='FairGrad max grad norm')
        # get_arguments() 增加
    parser.add_argument('--use_mbt', action='store_true', help='Use MBT architecture')
    parser.add_argument('--pretrained', action='store_true', help='Load ViT/AST pretrain')

    return parser.parse_args()


def get_shared_parameters(model, args):
    m = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    if args.use_mbt:
        # ✅ MBT：排除分类头，其余全部参与
        return [
            p for n, p in m.named_parameters()
            if p.requires_grad and not any(x in n for x in ["head"])
        ]
    
    shared_prefixes = ("audio_net.", "visual_net.", "fusion.")
    return [p for n, p in m.named_parameters() if p.requires_grad and n.startswith(shared_prefixes)]

def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, weight_method, writer=None):
    criterion = nn.CrossEntropyLoss()
    model.train()
    print(f"Start training epoch {epoch}...")

    _loss = 0.0
    shared_params = get_shared_parameters(model, args)

    for step, (spec, images, label) in enumerate(dataloader):
        optimizer.zero_grad()

        images = images.to(device)
        spec = spec.to(device)
        label = label.to(device)

        # 前向传播
        out, out_a, out_v = model(spec.float(), images.float())

        # Debug（可留可删）
        if step % 100 == 0:
            print(f"Pred: {torch.bincount(out.argmax(dim=1), minlength=args.n_classes)}")
            print(f"Real: {torch.bincount(label, minlength=args.n_classes)}")

        # 三个任务 loss
        loss_mm = criterion(out, label)
        loss_a = criterion(out_a, label)
        loss_v = criterion(out_v, label)

        # ✅ 标准 FairGrad
        losses = torch.stack([loss_mm, loss_a, loss_v])

        _, extra_info = weight_method.backward(
            losses=losses,
            shared_parameters=shared_params,
        )
        optimizer.step()

        weights = extra_info["weights"]  # 已经是 np.ndarray
        if step % 10 == 0:
            print(f"Batch {step}: FairGradient weights: {weights}")

        _loss += losses.mean().item()

    return _loss / len(dataloader)

def valid(args, model, device, dataloader):

    n_classes = args.n_classes


    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a= [0.0 for _ in range(n_classes)]
        acc_v= [0.0 for _ in range(n_classes)]

        for step, (spec, images, label) in tqdm(enumerate(dataloader)):
            spec = spec.to(device)
            images = images.to(device)
            label = label.to(device)
            prediction_all = model(spec.float(), images.float())


            prediction=prediction_all[0]
            prediction_audio=prediction_all[1]
            prediction_visual=prediction_all[2]

            for i, item in enumerate(label):

                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                # print(index_ma, label_index)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0
                
                ma_audio=prediction_audio[i].cpu().data.numpy()
                index_ma_audio = np.argmax(ma_audio)
                if index_ma_audio == label[i]:
                    acc_a[label[i]] += 1.0


                ma_visual=prediction_visual[i].cpu().data.numpy()
                index_ma_visual = np.argmax(ma_visual)
                if index_ma_visual == label[i]:
                    acc_v[label[i]] += 1.0


    return sum(acc) / sum(num), sum(acc_a) / sum(num),sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    
    # 1. 参数隔离：仅在 CGMNIST 时修改类别数，不影响其他数据集默认的 6 类
    if args.dataset == 'CGMNIST':
        args.n_classes = 10
    elif args.dataset == 'KineticSound':
        args.n_classes = 31
    elif args.dataset == 'CREMAD':
        args.n_classes = 6
    
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # 2. 设备初始化
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')

    # 3. 模型实例化隔离：根据数据集自动切换模型
    if args.use_mbt:
        model = MBTClassifier(
            num_classes=args.n_classes,
            pretrained=args.pretrained
        )
    elif args.dataset == 'CGMNIST':
        from models.models import CGClassifier
        model = CGClassifier(args)
    else:
        from models.models import AVClassifier
        model = AVClassifier(args)
    
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    # 4. 数据集加载隔离：确保每个分支都正确赋值 train_dataset
    if args.dataset == 'CREMAD':
        train_dataset = AVDataset_CD(mode='train')
        test_dataset = AVDataset_CD(mode='test')
    elif args.dataset == 'KineticSound':
        from dataset.kinetic_sound_loader import KineticSoundDataset
        train_dataset = KineticSoundDataset(mode='train')
        test_dataset  = KineticSoundDataset(mode='test')
    elif args.dataset == 'CGMNIST':
        # ⭐ 关键修正：这里加载的是 Dataset，而不是重新加载模型
        from dataset.cgmnist_loader import CGMNISTDataset
        train_dataset = CGMNISTDataset(mode='train')
        test_dataset  = CGMNISTDataset(mode='test')
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # --- 现在的逻辑下，无论跑哪个数据集，train_dataset 都有值了 ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True   
    )

    # ✅ 新增：在这里创建 test_dataloader，这样 valid 函数就能找到了
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False, # 测试集通常不 shuffle
        num_workers=8,
        pin_memory=True,
        persistent_workers=True   
    )
    # ... 后续代码保持不变 ...
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    print(len(train_dataloader))


    if args.train:
        best_acc = -1

        # ✅ 放在 epoch 外
        weight_method = WeightMethods(
            method="fairgrad",
            n_tasks=3,
            device=device,
            alpha=args.fair_alpha,
            max_norm=args.fair_max_norm,
        )

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            # ✅ 传进去
            batch_loss = train_epoch(
                args, epoch, model, device,
                train_dataloader, optimizer, scheduler,
                weight_method
            )

            acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

            scheduler.step()

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_{}_of_{}_{}_epoch{}_batch{}_lr{}.pth'.format(
                    args.model, args.optimizer,  args.dataset, args.epochs, args.batch_size, args.learning_rate)

                saved_dict = {'saved_epoch': epoch,
                                'acc': acc,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)

                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.4f}, Acc: {:.4f}, Acc_a: {:.4f}, Acc_v: {:.4f}".format(
                    batch_loss, acc, acc_a,acc_v))
            else:
                print("Loss: {:.4f}, Acc: {:.4f}, Acc_a: {:.4f}, Acc_v: {:.4f},Best Acc: {:.4f}".format(
                    batch_loss, acc,acc_a,acc_v,best_acc))



if __name__ == "__main__":
    main()
