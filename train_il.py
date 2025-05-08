import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import numpy as np
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
# from torch_geometric.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

from policy.modules.graph_pointer import GraphPointerPolicy, GraphQNet
from policy.rl_algorithms.arguments import args as rl_args

from torch.utils.tensorboard import SummaryWriter

rl_args.graph_num_graph_padding = -1
rl_args.graph_embedding_dim = 64

# rl_args.graph_node_feature_dim = 5
# rl_args.graph_node_feature_dim = 4
rl_args.graph_node_feature_dim = 102
rl_args.graph_edge_feature_dim = 3

# 设备自动选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphSupervisedDataset(Dataset):
    """加载存储在.npy文件中的图监督学习数据集"""
    # def __init__(self, data_dir, encoder_type='GCN'):
    def __init__(self, data_dir, encoder_type='GCN', data_num=100000):
        self.data_dir = data_dir
        self.encoder_type = encoder_type
        al_file_ls = os.listdir(data_dir)[:data_num]
        self.samples = [f for f in al_file_ls if f.endswith('.npy')]
        
        # 验证数据完整性
        sample_path = os.path.join(data_dir, self.samples[0])
        sample_data = np.load(sample_path, allow_pickle=True).item()
        assert 'current_state' in sample_data and 'policy_acton_idx' in sample_data and 'policy_acton_idx_copy_llm' in sample_data, "数据格式不符合要求"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 加载单个样本
        file_path = os.path.join(self.data_dir, self.samples[idx])
        data = np.load(file_path, allow_pickle=True).item()
        
        # 解析特征和标签
        state_dict = data['current_state']
        label = data['policy_acton_idx_copy_llm']
        # label = data['policy_acton_idx']

        # state_dict['pyg_graph'].x[:, 3:24] /= state_dict['pyg_graph'].x[:, 24:45] # dis_revise
        # state_dict['pyg_graph'].x = torch.cat([state_dict['pyg_graph'].x[:, 0:100], state_dict['pyg_graph'].x[:, 150:]], dim=1)

        # 根据编码器类型构造输入状态
        if self.encoder_type in ['GCN', 'GAT']:
            # 构造PyG图数据对象
            graph = Data(
                x=torch.FloatTensor(state_dict['pyg_graph'].x),        # 节点特征 [N, D]
                edge_index=torch.LongTensor(state_dict['pyg_graph'].edge_index).contiguous(),  # 边连接 [2, E]
                edge_attr=torch.FloatTensor(state_dict['pyg_graph'].edge_attr)
            )
            
            # 其他状态元素
            current_idx = torch.LongTensor([state_dict['current_idx']])      # 当前节点索引 [1]
            action_idx = torch.LongTensor(state_dict['action_idxes'])        # 候选动作索引 [M]
            action_mask = torch.FloatTensor(state_dict['action_mask'])            # 动作掩码 [M]
            
            state = (graph, current_idx, action_idx, action_mask)
            
        # elif self.encoder_type == 'Transformer':
        #     # 构造Transformer需要的矩阵输入
        #     nodes = torch.FloatTensor(state_dict['node_features'])               # 节点特征 [N, D]
        #     node_padding_mask = torch.LongTensor(state_dict['padding_mask'])     # 填充掩码 [N]
        #     edge_matrix = torch.FloatTensor(state_dict['adjacency_matrix'])      # 邻接矩阵 [N, N]
            
        #     current_idx = torch.LongTensor([state_dict['current_node_idx']])     # [1]
        #     action_idx = torch.LongTensor(state_dict['candidate_actions'])       # [M]
        #     action_mask = torch.LongTensor(state_dict['action_mask'])           # [M]
            
        #     state = (nodes, node_padding_mask, edge_matrix, current_idx, action_idx, action_mask)
            
        else:
            raise ValueError(f"不支持的编码器类型: {self.encoder_type}")

        return state, torch.LongTensor([label])  # 标签需要是整数索引


def collate_fn(batch):
    """自定义批处理函数"""
    states, labels = zip(*batch)
    
    # 根据编码器类型处理不同输入格式
    if isinstance(states[0][0], Data):  # GCN/GAT情况
        # 解包状态元组
        graphs = [s[0] for s in states]
        current_indices = torch.stack([s[1] for s in states]).to(device)          # [B, 1]
        action_indices = torch.cat([s[2] for s in states]).to(device)         # [B, M]
        action_masks = torch.cat([s[3] for s in states]).to(device)          # [B, M]
        # print("graphs:", graphs)
        # 使用PyG的批处理
        # print("current_indices:",current_indices.shape)
        # print("\n\n\n\n\n")
        graph_batch = Batch.from_data_list(graphs)
        state = (graph_batch, current_indices, action_indices, action_masks)
        
    # else:  # Transformer情况
    #     nodes = torch.stack([s[0] for s in states])                 # [B, N, D]
    #     node_masks = torch.stack([s[1] for s in states])           # [B, N]
    #     edge_matrices = torch.stack([s[2] for s in states])        # [B, N, N]
    #     current_indices = torch.cat([s[3] for s in states])        # [B, 1]
    #     action_indices = torch.stack([s[4] for s in states])       # [B, M]
    #     action_masks = torch.stack([s[5] for s in states])        # [B, M]
        
    #     state = (nodes, node_masks, edge_matrices, current_indices, action_indices, action_masks)
    
    return state, torch.cat(labels)  # 标签形状 [B]


# 数据加载（假设已实现自定义Dataset）
train_dataset = GraphSupervisedDataset(
    data_dir="il_data_frontier_score_revise_intention_for_train/",
    encoder_type='GAT',  # 根据实际情况修改
    data_num=9100
)

val_dataset = GraphSupervisedDataset(
    data_dir="il_data_frontier_score_revise_intention_for_train_val/",
    encoder_type='GAT',  # 根据实际情况修改
    data_num=120
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# 假设已定义GraphPointerPolicy及相关组件

# 初始化模型
model = GraphPointerPolicy(
    node_dim=rl_args.graph_node_feature_dim,
    edge_dim=rl_args.graph_edge_feature_dim,
    embedding_dim=rl_args.graph_embedding_dim,
    num_graph_padding=rl_args.graph_num_graph_padding,
    encoder_type='GAT'  # 根据实际情况选择编码器类型
).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

date_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
train_note = "_multi_check_il_gt_train_frontier_score_revise_intention_for_train_old_label_only_score_dis" # 注释当前训练处于什么阶段
logger_file_name = "./log_files_train_il/log_"+date_time+train_note
writer = SummaryWriter(logger_file_name)


def save_checkpoint(epoch, model, optimizer):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state':optimizer.state_dict(),
    }
    
    # 常规保存
    torch.save(state, f'checkpoints_frontier_score_revise_intention_for_train_old_label_only_score_dis/epoch_{epoch+1}.pt')


# 训练循环
def train_supervised(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0
        for batch in train_loader:
            # 假设batch包含graphs, current_idx, action_idx, action_mask, labels
            # 根据encoder_type构造state
            if model._encoder_type in ['GCN', 'GAT']:
                # print("batch:", len(batch))
                # state = (batch.graphs, batch.current_idx, batch.action_idx, batch.action_mask)
                state = (batch[0][0].to(device), batch[0][1].to(device), batch[0][2].to(device), batch[0][3].to(device))
            elif model._encoder_type == 'Transformer':
                state = (batch.nodes, batch.node_padding_mask, batch.edge_matrix, 
                         batch.current_idx, batch.action_idx, batch.action_mask)
            else:
                raise ValueError("Unsupported encoder type")
            
            # 前向传播
            optimizer.zero_grad()
            logits = model(state, rl_args)
            
            # 计算损失
            # loss = criterion(logits, batch.labels)
            loss = criterion(logits, batch[1].to(device))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_correct += (torch.argmax(logits, dim=-1) == batch[1].to(device)).sum().item()
            total_samples += len(batch[1])
        
        avg_loss = total_loss / len(train_loader)
        accu = total_correct/ total_samples

        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accu: {accu:.4f}, Total_num: {total_samples:.0f}')

        save_checkpoint(epoch, model, optimizer)

        writer.add_scalar('Result/train_loss', avg_loss, epoch+1)
        writer.add_scalar('Result/train_accu', accu, epoch+1)     

        # 验证集验证
        model.eval()   
        eval_loss = 0.0
        eval_correct = 0
        eval_samples = 0
        for batch in val_loader:
            state = (batch[0][0].to(device), batch[0][1].to(device), batch[0][2].to(device), batch[0][3].to(device))
            logits = model(state, rl_args)
            loss = criterion(logits, batch[1].to(device))

            eval_loss += loss.item()
            eval_correct += (torch.argmax(logits, dim=-1) == batch[1].to(device)).sum().item()
            eval_samples += len(batch[1])

        eval_loss = eval_loss / len(val_loader)
        eval_accuracy = eval_correct / eval_samples

        writer.add_scalar('Result/eval_loss', eval_loss, epoch+1)
        writer.add_scalar('Result/eval_accu', eval_accuracy, epoch+1)     



# 执行训练
train_supervised(model, train_loader, val_loader, criterion, optimizer, num_epochs=10000)