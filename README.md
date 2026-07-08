# FedDG MoE Extended

FedDG MoE Extended 是一个用于图像分类的联邦领域泛化框架。当前代码已经固定为最终基线：冻结 ViT/CLIP backbone，只训练轻量级 decoupled adapter、分类器与类别原型；测试时使用 grouped raw-batch source-whitened diagonal W2 进行 source-private style 参数融合。

最终基线为：

```text
grouped early/middle/late raw-batch source-whitened diagonal W2
early/middle/late 三组全部参与 source-private style_down/style_up fusion
source_val_ensemble_acc_avg checkpoint selection
selection tie-break: >= 选择更新 round
prototype ensemble = 0.5 cls + 0.5 proto
train-time full forward source style statistics
random training augmentation 保留
```

训练和 checkpoint selection 只使用 source 域训练/验证信息。target test 指标会在每轮记录，并在最终 best checkpoint 上重新评估，但不参与训练目标、优化器更新或 checkpoint selection。

## 代码结构

```text
FedDG_MoE_Extended/
|-- main_feddg_moe.py          # CLI 入口与联邦训练主循环
|-- client_train.py            # source client 本地训练与 grouped style 统计
|-- feddg_eval.py              # source/target 评估与 grouped raw-batch TTA
|-- feddg_utils.py             # FedAvg、checkpoint、日志、随机种子、状态读写
|-- requirements.txt           # Python 依赖
|-- data/
|   `-- feddg_loaders.py       # PACS / OfficeHome / VLCS dataloader 与 transform
`-- network/
    |-- adapters.py            # DecoupledAdapter：content/style/router
    `-- get_network.py         # ViT/CLIP 构建与 adapter 注入
```

## 模型架构

`GetNetwork()` 返回：

```text
nn.Sequential(featurizer, classifier)
```

整体前向路径为：

```text
image
  -> frozen ViT/CLIP featurizer with injected adapters
  -> feature vector
  -> linear classifier
  -> class logits
```

### Backbone

入口脚本固定使用：

```python
args.model = "vit_clip"
```

`vit_clip` 对应 timm 模型：

```text
vit_base_patch16_clip_224.laion2b
```

当 `pretrained=True` 时，代码从本地 Hugging Face cache 加载权重：

```text
../hf_hub_cache/models--timm--vit_base_patch16_clip_224.laion2b/snapshots/<snapshot>/
```

snapshot 中需要包含以下任意一种权重文件：

```text
model.safetensors
pytorch_model.bin
```

ViT 原始分类头会被替换为 `nn.Identity()`。任务分类器是单独的线性层：

```python
nn.Linear(in_features=feature_level, out_features=num_classes, bias=True)
```

### Adapter 注入位置

adapter 注入到 ViT MLP 输出投影附近。匹配的模块名包括：

```text
fc2
c_proj
mlp.fc2
mlp.c_proj
```

每个目标线性层会被包装成：

```text
output = base_linear(x) + DecoupledAdapter(x)
```

backbone 参数冻结。训练中可更新的状态包括：

```text
adapter parameters
classifier parameters
class prototypes
```

其中 `style_down` 和 `style_up` 是 source-private 状态，不进入全局 adapter 聚合。

### DecoupledAdapter

每个 `DecoupledAdapter` 包含 content path、style path 和 router。

构造参数：

```python
DecoupledAdapter(
    in_features,
    out_features,
    content_r=16,
    style_r=4,
    adapter_scale=1.0,
    router_temp_init=5.0,
    router_temp_min=1.5,
)
```

CLI 默认 rank：

```text
--content_rank 32
--style_rank 4
```

content path：

```text
x
  -> LayerNorm
  -> Linear(in_features, content_r)
  -> InstanceNorm1d(content_r)
  -> SiLU
  -> Dropout(p=0.2)
  -> Linear(content_r, out_features)
  -> scale by adapter_scale / content_r
```

style path：

```text
x
  -> LayerNorm
  -> patch tokens
  -> raw style statistics: concat(mean, std)
  -> Linear(2 * in_features, style_r)
  -> SiLU
  -> Dropout(p=0.2)
  -> Linear(style_r, out_features)
  -> scale by adapter_scale / style_r
  -> expand to all tokens
```

router：

```text
CLS token
  -> Linear(in_features, 64)
  -> SiLU
  -> Linear(64, 2)
  -> softmax(logits / router_temperature)
```

最终 adapter 输出：

```text
adapter_out = gamma * (alpha * content_out + beta * style_out)
```

其中 `alpha` 和 `beta` 是 router 的两路 softmax 权重。

### Adapter 模式

adapter 有两种模式：

```text
full          正常前向：content path、style path、router 全部启用
extract_style 采集 raw style statistics，并返回零 adapter 输出
```

训练阶段使用 `full`。target batch 估计 style fusion 权重时先使用 `extract_style`，随后切回 `full` 进行分类前向。

### Router 温度调度

每轮联邦训练开始时：

```python
router_progress = min(1.0, round_idx / router_anneal_rounds)
```

每个 adapter 内部执行 cosine annealing：

```text
temperature =
  router_temp_min
  + (router_temp_init - router_temp_min)
    * 0.5 * (1 + cos(pi * router_progress))
```

???? `--router_anneal_rounds`???????? `--comm`?

### Style 分组

当前 ViT-Base 配置要求正好有 12 个 `DecoupledAdapter`。按 `featurizer.named_modules()` 的顺序分为三组：

```text
early:  adapters 0-3
middle: adapters 4-7
late:   adapters 8-11
```

三组都会参与 source-private `style_down` / `style_up` 融合。

## 联邦训练流程

每个任务采用 leave-one-domain-out 设置：一个域作为 target，其余域作为 source clients。

每个 communication round 的流程：

1. server 持有当前全局 non-style adapter、classifier 和 class prototypes。
2. 对每个 source domain，server 基于 global model 创建 client model。
3. 如果该 source 已有 source-private style 参数，则加载到 client model。
4. client 使用本地域数据进行本地训练，adapter mode 为 `full`。
5. full forward 期间收集 early/middle/late grouped raw style statistics。
6. client 返回：
   - non-style adapter state
   - source-private `style_down` / `style_up` state
   - classifier state
   - prototype state 与 class counts
   - grouped raw style statistics
   - average train loss
7. server 聚合 non-style adapter、classifier 和 prototypes。
8. 在 source validation 上评估，用于 checkpoint selection。
9. 在 target test 上运行最终 TTA 评估，用于监测和最终报告。

### Client 本地目标函数

本地训练 loss：

```text
loss = CE(classifier_logits, labels) + round_lambda_proto * prototype_loss
```

prototype loss 权重线性 warm up：

```python
proto_ratio = min(1.0, round_idx / proto_warmup_rounds)
round_lambda_proto = lambda_proto * proto_ratio
```

prototype 分支使用归一化特征和归一化 prototype：

```text
proto_logits = normalize(features) @ normalize(global_prototypes).T / proto_tau
```

没有有效 prototype 的类别会在 prototype loss 和 prototype prediction 中被 mask。

### 训练期 Grouped Style Statistics

本地训练 full forward 时，每个 adapter 会保存 raw style feature：

```text
last_style_stats_raw = concat(patch_token_mean, patch_token_std)
```

client 按三组收集：

```python
STYLE_GROUP_NAMES = ("early", "middle", "late")
```

返回的 `style_stats` 至少包含：

```python
{
    "group_stats": {
        "early": {
            "global_count": int,
            "global_mean": Tensor,
            "global_std": Tensor,
            "adapter_names": tuple[str, ...],
        },
        "middle": {...},
        "late": {...},
    }
}
```

返回的 `style_state` 保存所有 adapter 的：

```text
style_down
style_up
```

### Server 聚合

server 聚合：

```text
non-style adapter parameters
classifier parameters
class prototypes
```

`style_down` 和 `style_up` 是 source-private 参数，不放回 global adapter state。`extract_adapter_state_from_model()` 因此会排除这两类参数。

prototype 聚合使用 class counts。若某个类别在所有 source client 中本轮都没有样本，则对该类别使用 sample-count weighted fallback。

## Checkpoint Selection

checkpoint selection 只依赖 source validation ensemble accuracy：

```python
selection_score = float(source_val_ensemble_acc_avg)

if selection_score >= best_val_acc:
    best_val_acc = selection_score
    best_round = round_idx + 1
    save_checkpoint(...)
```

保存到 checkpoint 的 criterion 字段为：

```text
source_val_ensemble_acc_avg
```

`>=` 表示同分时选择更新的 round。

checkpoint 中保存的主要内容：

```text
round
target_domain
source validation accuracies
target test monitoring accuracies
main_expert_state
classifier_state
prototype_state
client_style_states
client_style_stats
tta_report
args
```

训练结束后，脚本会重新加载 best checkpoint，在 target test loader 上再次评估，写入 `final_tta_report`，并保存 `_with_test.pth` 文件。

## Target Evaluation 与 Grouped Raw-Batch TTA

最终 target evaluation 在 `evaluate_target_with_style_mode()` 中实现。

对选中的 checkpoint：

1. 分别为 early/middle/late 三组构建 source W2 geometry。
2. 每个 target batch 先用 `extract_style` 模式收集 target raw style features。
3. 每组分别计算 target batch 到各 source 的 raw-batch source-whitened diagonal W2 distances。
4. 每组使用 direct softmax 将 distances 转为 source weights。
5. `_fuse_grouped_style_state()` 按三组权重融合 source-private `style_down` / `style_up`。
6. 将 fused style state 加载到模型。
7. 切回 `full` 模式，计算 classifier、prototype 和 ensemble predictions。

ensemble prediction 使用：

```text
ensemble_probability = 0.5 * classifier_probability + 0.5 * prototype_probability
```

TTA report 对每组记录：

```text
distances
weights
source_temperature
reference_scale_mean
reference_scale_min
reference_scale_max
```

同时记录：

```text
mode = raw_batch_grouped
metric = grouped_source_whitened_diag_w2
stabilization = none
```

## 数据集与划分

支持的数据集：

| Dataset | Domains | Classes |
|---|---|---:|
| `pacs` | `art_painting`, `cartoon`, `photo`, `sketch` | 7 |
| `officehome` | `art`, `clipart`, `product`, `real_world` | 65 |
| `vlcs` | `caltech`, `labelme`, `pascal`, `sun` | 5 |

PACS 使用 split files：

```text
train -> train
val   -> crossval
test  -> test
```

OfficeHome 和 VLCS 会优先使用已有的 `train`、`crossval` / `val`、`test` 文件夹。如果没有 train/val 文件夹，则从 domain 文件夹构造确定性的 90/10 train/val split。

默认数据根目录解析：

```text
../datasets/PACS
../datasets/Office-Home
../datasets/VLCS
```

`--data_root` 可以传入具体数据集目录，也可以传入父级 `datasets` 目录。

## Transform

训练 transform：

```text
RandomResizedCrop(224, scale=(0.7, 1.0))
RandomHorizontalFlip()
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
RandomGrayscale()
ToTensor()
Normalize(ImageNet mean/std)
```

验证和测试 transform：

```text
Resize([224, 224])
ToTensor()
Normalize(ImageNet mean/std)
```

random training augmentation 是最终基线的一部分。

## 环境安装

建议创建独立环境：

```bash
conda create -n feddg-moe python=3.10 -y
conda activate feddg-moe
pip install -r requirements.txt
```

如果本机 CUDA / driver 与 pinned package 不匹配，请安装与你机器匹配的 PyTorch 版本。

默认 `vit_clip` 需要本地 timm/Hugging Face 权重 cache：

```text
../hf_hub_cache/models--timm--vit_base_patch16_clip_224.laion2b/snapshots/<snapshot>/
```

## 命令示例

PACS

```bash
python main_feddg_moe.py \
  --dataset pacs \
  --test_domain all \
  --local_epochs 4 \
  --comm 20 \
  --batch_size 64 \
  --test_batch_size 128 \
  --lr 1e-3 \
  --lambda_proto 0.8 \
  --proto_tau 0.1 \
  --proto_warmup_rounds 2 \
  --content_rank 16 \
  --style_rank 4
```


OfficeHome

```bash
python main_feddg_moe.py \
  --dataset officehome \
  --test_domain all \
  --local_epochs 4 \
  --comm 20 \
  --batch_size 64 \
  --test_batch_size 128 \
  --lr 1e-3 \
  --lambda_proto 0.8 \
  --proto_tau 0.1 \
  --proto_warmup_rounds 2 \
  --content_rank 16 \
  --style_rank 4
```

VLCS

```bash
python main_feddg_moe.py \
  --dataset vlcs \
  --test_domain all \
  --local_epochs 4 \
  --comm 20 \
  --batch_size 64 \
  --test_batch_size 128 \
  --lr 1e-3 \
  --lambda_proto 0.8 \
  --proto_tau 0.1 \
  --proto_warmup_rounds 2 \
  --content_rank 16 \
  --style_rank 4
```


自动选择 dataloader worker 数：

```bash
python main_feddg_moe.py \
  --dataset pacs \
  --data_root ../datasets/PACS \
  --test_domain photo \
  --num_workers -1
```



## CLI 参数

Data 参数：

| Argument | Default | Description |
|---|---:|---|
| `--dataset` | `pacs` | `pacs`、`officehome`、`vlcs` |
| `--data_root` | empty | 数据集目录或父级 datasets 目录 |
| `--test_domain` | `sketch` | held-out target domain，或 `all` |
| `--num_workers` | `2` | dataloader workers；`-1` 表示自动选择 |
| `--max_train_samples` | `0` | 每个 domain 最大训练样本数；`0` 表示不限制 |
| `--max_eval_samples` | `0` | 每个 domain 最大 val/test 样本数；`0` 表示不限制 |

Training 参数：

| Argument | Default | Description |
|---|---:|---|
| `--num_classes` | dataset default | 手动覆盖类别数 |
| `--batch_size` | `32` | source train 和 source val batch size |
| `--test_batch_size` | `64` | target test batch size |
| `--local_epochs` | `5` | 每轮本地训练 epoch 数 |
| `--comm` | `5` | communication rounds |

Optimization / model 参数：

| Argument | Default | Description |
|---|---:|---|
| optimizer | `sgd` | Fixed SGD; no CLI switch |
| `--lr` | `1e-3` | learning rate |
| `--sgd_momentum` | `0.9` | SGD momentum |
| `--weight_decay` | `0.0` | weight decay |
| `--lambda_proto` | `1.0` | prototype loss 最终权重 |
| `--proto_tau` | `0.1` | prototype temperature |
| `--proto_warmup_rounds` | `4` | prototype loss warmup rounds |
| `--router_anneal_rounds` | same as `--comm` | router annealing rounds |
| `--content_rank` | `32` | content adapter bottleneck rank |
| `--style_rank` | `4` | style adapter bottleneck rank |

Runtime / logging 参数：

| Argument | Default | Description |
|---|---:|---|
| `--seed` | `0` | random seed |
| `--device` | auto | 有 CUDA 时默认为 `cuda`，否则 `cpu` |
| `--log_dir` | `training_logs` | 文本日志和 checkpoint 目录 |
| `--tb_log_dir` | `tb_runs` | TensorBoard log 目录 |

## 输出文件

每个任务的文本日志：

```text
training_logs/train_<dataset>_<target>_<timestamp>.log
```

checkpoint：

```text
training_logs/checkpoints/best_model_<dataset>_<target>_<timestamp>.pth
training_logs/checkpoints/best_model_<dataset>_<target>_<timestamp>_with_test.pth
```

TensorBoard：

```text
tb_runs/<dataset>_<target>_<timestamp>/
```

常用 TensorBoard scalars：

```text
Source_Val_Ensemble_Accuracy_Avg
Source_Val_Cls_Accuracy_Avg
Source_Val_Proto_Accuracy_Avg
Target_Test_Ensemble_Accuracy
Target_Test_Cls_Accuracy
Target_Test_Proto_Accuracy
Train_Loss
Selection_Score
```

启动 TensorBoard：

```bash
tensorboard --logdir tb_runs
```

## 复现说明

脚本会设置 Python、NumPy 和 PyTorch 随机种子。dataloader worker 使用确定性 worker seed，train/val/test dataloader generator 使用 split 和 domain 相关的确定性 offset。

训练 augmentation 保持随机，但受 dataloader generator 与 worker seed 控制。

## 静态检查

编译核心文件：

```bash
python -m py_compile \
  client_train.py \
  feddg_eval.py \
  feddg_utils.py \
  main_feddg_moe.py \
  network/adapters.py \
  network/get_network.py \
  data/feddg_loaders.py
```

如果需要检查旧实验分支词汇，请把待查 pattern 放在本地 shell 变量或临时脚本里，不要把这些词重新写回 README：

```bash
grep -R "$OLD_EXPERIMENT_KEYWORD_PATTERN" .
```

Windows PowerShell 没有 `grep` 时可用：

```powershell
rg $env:OLD_EXPERIMENT_KEYWORD_PATTERN .
```
