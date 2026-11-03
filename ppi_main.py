import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from evaluation import compute_roc, compute_aupr, compute_mcc, acc_score, micro_score, compute_performance
import os
import time
from torch.nn.parameter import Parameter
from utils import ProDataset
from torch_geometric.utils import dense_to_sparse
import pandas as pd
import random


from Model_alpha import AverageMeter, PPI_Model

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = "cuda" if torch.cuda.is_available() else "cpu"

early_stopping = False
use_pLM = False

map_cutoff = 16  # 14
THREADHOLD = 0.4
Freq = 50


def train_epoch(model, loader, optimizer, epoch, all_epochs, print_freq=Freq):
    batch_time = AverageMeter()
    losses = AverageMeter()

    global THREADHOLD
    # Model on train mode
    model.train()

    all_preds = []
    all_trues = []
    end = time.time()

    for batch_idx, (seq_data, dssp_data,    atom_onehot_embedding, edge_index_atom, res2atom,
        bert_data, pseudo_dis, dis_data, Res_matrix,node_features, label, coords_atom, seq1) in enumerate(loader):


        # dssp_data=dssp_data[:,:,:9] #   仅仅使用9维的二级结构
        # bert_data=bert_data[:,:,512:]
        node_number = np.array(seq_data)[0].shape[0]

        # 将密集矩阵转换为稀疏表示
        edge_index1, _ = dense_to_sparse(dis_data)
        edge_index = edge_index1.to(torch.int64)  # .t()

        # g = dgl.graph((edge_index1[0],edge_index1[1])).to('cuda:0')

        # edge_attr=cal_edge_attr(edge_index1,pseudo_dis.squeeze(0),node_features)
        # edge_attr=torch.tensor(edge_attr)
        # edge_attr = edge_attr.permute(1, 0) #   [4565,2]

        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():

                node_features = torch.autograd.Variable(node_features.cuda().float())
                bert_var = torch.autograd.Variable(bert_data.cuda().float())
                pseudo_dis = torch.autograd.Variable(pseudo_dis.cuda().float())  # .float())
                Res_matrix = torch.autograd.Variable(Res_matrix.cuda().float())  # .float()

                atom_onehot_embedding = torch.autograd.Variable(atom_onehot_embedding.cuda().float())  # .float()
                edge_index_atom = torch.autograd.Variable(edge_index_atom.cuda())
                coords_atom = torch.autograd.Variable(coords_atom.cuda().float())
                res2atom = torch.autograd.Variable(res2atom.cuda())

                target_var = torch.autograd.Variable(label.cuda())  # .float()
                seq_var = torch.autograd.Variable(seq_data.cuda().float())

            else:
                node_features = torch.autograd.Variable(node_features.float())
                bert_var = torch.autograd.Variable(bert_data.float())
                pseudo_dis = torch.autograd.Variable(pseudo_dis.float())
                Res_matrix = torch.autograd.Variable(Res_matrix).float()  # .float()

                atom_onehot_embedding = torch.autograd.Variable(atom_onehot_embedding).float()  # .float()
                edge_index_atom = torch.autograd.Variable(edge_index_atom)
                coords_atom = torch.autograd.Variable(coords_atom)
                res2atom = torch.autograd.Variable(res2atom.to(torch.int64))

                target_var = torch.autograd.Variable(label)  # .float()
                seq_var = torch.autograd.Variable(seq_data.float())


        # compute output with mixed precision to save VRAM and accelerate on RTX 4050
        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                output, cl_loss = model(node_number, node_features, bert_var, pseudo_dis, Res_matrix, atom_onehot_embedding, edge_index_atom, res2atom, coords_atom)
                output = output.transpose(0, 1)

            output = torch.nan_to_num(output.float(), nan=0.5, posinf=0.9999, neginf=0.0001)
            output = torch.clamp(output, 1e-7, 1.0 - 1e-7)

            all_trues.append(target_var.float().detach().cpu().numpy())
            all_preds.append(output.detach().cpu().numpy())

            loss = torch.nn.functional.binary_cross_entropy(output, target_var.float())
            loss = loss + cl_loss.float() * 0.1

            # measure accuracy and record loss
            batch_size = label.size(0)
            pred_out = output.ge(THREADHOLD)

            pred_out = pred_out.data.cpu().numpy()
            target_var = target_var.data.cpu().numpy()

            MiP, MiR, MiF, PNum, RNum = micro_score(pred_out, target_var)
            losses.update(loss.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if node_number > 500 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"Warning: GPU OOM on protein index {batch_idx} (len {node_number}). Clearing cache and continuing.")
            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, all_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'f_max:%.6f' % (MiP),
                'p_max:%.6f' % (MiR),
                'r_max:%.6f' % (MiF),
                't_max:%.2f' % (PNum)
            ])
            print('\n', res)

    return _, losses.avg,


def eval_epoch(model, loader, print_freq=Freq, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    global THREADHOLD
    # Model on eval mode
    model.eval()

    all_trues = []
    all_preds = []
    end = time.time()

    for batch_idx, (
            seq_data, dssp_data, atom_onehot_embedding, edge_index_atom, res2atom, bert_data, pseudo_dis, dis_data,
            Res_matrix, node_features, label, coords_atom, seq1) in enumerate(loader):

        # dssp_data=dssp_data[:,:,:9] #   仅仅使用9维的二级结构
        # bert_data=bert_data[:,:,512:]
        node_number = np.array(seq_data)[0].shape[0]

        # 将密集矩阵转换为稀疏表示
        edge_index1, _ = dense_to_sparse(dis_data)
        edge_index = edge_index1.to(torch.int64)  # .t()

        # g = dgl.graph((edge_index1[0],edge_index1[1])).to('cuda:0')

        # edge_attr=cal_edge_attr(edge_index1,pseudo_dis.squeeze(0),node_features)
        # edge_attr=torch.tensor(edge_attr)
        # edge_attr = edge_attr.permute(1, 0) #   [4565,2]

        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():

                node_features = torch.autograd.Variable(node_features.cuda().float())
                bert_var = torch.autograd.Variable(bert_data.cuda().float())
                pseudo_dis = torch.autograd.Variable(pseudo_dis.cuda().float())  # .float())
                Res_matrix = torch.autograd.Variable(Res_matrix.cuda().float())  # .float()

                atom_onehot_embedding = torch.autograd.Variable(atom_onehot_embedding.cuda().float())  # .float()
                edge_index_atom = torch.autograd.Variable(edge_index_atom.cuda())
                coords_atom = torch.autograd.Variable(coords_atom.cuda().float())
                res2atom = torch.autograd.Variable(res2atom.cuda())

                target_var = torch.autograd.Variable(label.cuda())  # .float()
                seq_var = torch.autograd.Variable(seq_data.cuda().float())

            else:
                node_features = torch.autograd.Variable(node_features.float())
                bert_var = torch.autograd.Variable(bert_data.float())
                pseudo_dis = torch.autograd.Variable(pseudo_dis.float())
                Res_matrix = torch.autograd.Variable(Res_matrix).float()  # .float()

                atom_onehot_embedding = torch.autograd.Variable(atom_onehot_embedding).float()  # .float()
                edge_index_atom = torch.autograd.Variable(edge_index_atom)
                coords_atom = torch.autograd.Variable(coords_atom)
                res2atom = torch.autograd.Variable(res2atom.to(torch.int64))

                target_var = torch.autograd.Variable(label)  # .float()
                seq_var = torch.autograd.Variable(seq_data.float())

        # compute output
        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                output, cl_loss = model(node_number, node_features, bert_var, pseudo_dis, Res_matrix, atom_onehot_embedding,
                                        edge_index_atom, res2atom, coords_atom, )
                output = output.transpose(0, 1)

            output = torch.nan_to_num(output.float(), nan=0.5, posinf=0.9999, neginf=0.0001)
            output = torch.clamp(output, 1e-7, 1.0 - 1e-7)
            loss = torch.nn.functional.binary_cross_entropy(output, target_var.float())
            loss = loss + cl_loss.float() * 0.1
        except torch.cuda.OutOfMemoryError:
            print(f"Warning: GPU OOM on eval index {batch_idx}. Skipping.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue


        # measure accuracy and record loss
        batch_size = label.size(0)
        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        all_trues.append(label.numpy())
        all_preds.append(output.data.cpu().numpy())  # 使用教师模型的输出：output

    trues, preds, preds_pLM = [], [], []
    for i in range(len(all_trues)):
        trues.append(np.array(all_trues[i]).flatten())
        preds.append(np.array(all_preds[i]).flatten())

    all_trues = np.concatenate(trues, axis=0)  # np.array(all_trues)
    all_preds = np.concatenate(preds, axis=0)


    auc ,tprs= compute_roc(all_preds, all_trues)
    aupr, mean_precision, mean_recall = compute_aupr(all_preds, all_trues)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_preds, all_trues)

    if predictions_max is None:
        predictions_max = np.zeros_like(all_trues, dtype=float)
        print('HH' * 100)

    acc_val = acc_score(predictions_max, all_trues)
    mcc = compute_mcc(predictions_max, all_trues)

    return batch_time.avg, losses.avg, acc_val, f_max, p_max, r_max, auc, aupr, t_max, mcc


def train(model, train_data_set, valid_data_set, n_epochs=60, batch_size=32, lr=0.0008, seed=None, save_path="results/", resume=False, checkpoint_path=None):
    os.makedirs("model_save", exist_ok=True)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=1,
                                               pin_memory=(torch.cuda.is_available()),
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_data_set, batch_size=1,
                                               pin_memory=(torch.cuda.is_available()),
                                               num_workers=0)

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    model_wrapper = model

    # Train model state
    best_aupr = 0.0
    count = 0
    start_epoch = 0

    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, min_lr=1e-6)

    # Resume from checkpoint if requested
    if resume or checkpoint_path:
        ckpt_file = checkpoint_path if checkpoint_path else "model_save/checkpoint_latest.pth"
        if os.path.exists(ckpt_file):
            print(f"[*] Resuming training from checkpoint: {ckpt_file}")
            ckpt = torch.load(ckpt_file, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model_wrapper.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt.get("epoch", 0)
                best_aupr = ckpt.get("best_aupr", 0.0)
                count = ckpt.get("count", 0)
                global THREADHOLD
                THREADHOLD = ckpt.get("threshold", THREADHOLD)
                print(f"[*] Resumed at Epoch {start_epoch + 1}/{n_epochs} | Previous Best AUPR: {best_aupr:.6f}")
            else:
                model_wrapper.load_state_dict(ckpt)
                print("[*] Loaded weights from model state dict.")
        else:
            print(f"[!] Warning: Checkpoint '{ckpt_file}' not found. Starting from scratch (Epoch 1).")

    if start_epoch >= n_epochs:
        print(f"[*] Model has already reached {start_epoch} epochs (target: {n_epochs}). Nothing to train.")
        return

    try:
        for epoch in range(start_epoch, n_epochs):
            _, train_loss = train_epoch(
                model=model_wrapper,
                loader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                all_epochs=n_epochs,
            )

            _, valid_loss, acc, f_max, p_max, r_max, auc, aupr, t_max, mcc = eval_epoch(
                model=model_wrapper,
                loader=valid_loader,
                is_test=(not valid_loader)
            )

            scheduler.step(aupr)
            print(f'Epoch {epoch + 1}, Current Learning Rate:', optimizer.param_groups[0]['lr'])

            with open('results.txt', 'a') as file:
                file.write(
                    f'{epoch + 1}, {valid_loss}, {acc}, {f_max}, {p_max}, {r_max}, {auc}, {aupr}, {mcc}, {t_max}\n')

            print(
                'epoch:%03d,valid_loss:%0.5f:\nacc:%0.6f,F_value:%0.6f, precision:%0.6f,recall:%0.6f,auc:%0.6f,aupr:%0.6f,mcc:%0.6f,threshold:%0.6f' % (
                    (epoch + 1), valid_loss, acc, f_max, p_max, r_max, auc, aupr, mcc, t_max))

            if aupr > best_aupr:
                count = 0
                best_aupr = aupr
                THREADHOLD = t_max
                print('-' * 20, "new best AUPR:{0}(threshold:{1})".format(aupr, THREADHOLD), '-' * 20)
                torch.save(model.state_dict(), "model_save/best_model_"+str(epoch)+"_"+str(best_aupr)+".pth")
                torch.save(model.state_dict(), "model_save/best_model.pth")
            else:
                count += 1
                if count >= 250:
                    print("[*] Early stopping triggered.")
                    break

            # Save latest checkpoint for resumption
            latest_ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_aupr': best_aupr,
                'count': count,
                'threshold': THREADHOLD,
            }
            torch.save(latest_ckpt, "model_save/checkpoint_latest.pth")
            print(f"[*] Checkpoint saved for epoch {epoch + 1} -> model_save/checkpoint_latest.pth\n")

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("[!] Training paused by user.")
        print("[*] Saved latest state to 'model_save/checkpoint_latest.pth'.")
        print("[*] Resume anytime with: python ppi_main.py --resume")
        print("=" * 50)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EGCPPIS Training Script with Pause/Resume Support")
    parser.add_argument("--epochs", type=int, default=60, help="Total number of epochs to train (default: 60)")
    parser.add_argument("--resume", action="store_true", help="Resume training from model_save/checkpoint_latest.pth")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint to resume from")
    parser.add_argument("--lr", type=float, default=0.0008, help="Initial learning rate (default: 0.0008)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    file_path_train = "Datasets/Train_335.fa"
    file_path_varify = "Datasets/Test_60.fa"
    train_dataSet = ProDataset(file_path_train)
    varify_dataSet = ProDataset(file_path_varify)

    model = PPI_Model()

    # Train the model with pause/resume support
    train(
        model=model,
        train_data_set=train_dataSet,
        valid_data_set=varify_dataSet,
        n_epochs=args.epochs,
        lr=args.lr,
        resume=args.resume,
        checkpoint_path=args.checkpoint
    )
    print('Done!!!')




def set_seed(seed=100000):
    # 设置Python的随机种子
    random.seed(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    # 如果使用GPU，也需要固定GPU的随机数种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    # 确保GPU上的随机性固定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 测试随机性是否一致
def test_randomness():
    print("Random:", random.random())
    print("Numpy Random:", np.random.rand())
    print("Torch Random:", torch.rand(1))

    if torch.cuda.is_available():
        print("Torch CUDA Random:", torch.cuda.FloatTensor(1).uniform_())
    else:
        print("No CUDA")


if __name__ == '__main__':
    main()




