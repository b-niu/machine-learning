#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 导入必要的库
import datetime
import glob
import logging
import os
import re

# 导入一些绘图的库
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image  # 导入图像处理的库
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset  # 导入数据集和数据加载器的类
from tqdm import tqdm  # 导入tqdm库，用于显示进度条

# 定义超参数
num_classes = 24  # 根据chr[0-9XY]+的可能性
num_epochs = 300
patience = 30
batch_size = 64
learning_rate = 0.01
weight_decay = 1e-4
num_workers = 8
# 定义数据集和数据加载器
train_dir = "/home/ubuntu/train"
val_dir = "/home/ubuntu/val"
test_dir = "/home/ubuntu/test"

log_file = "train_log_20231120.txt"  # 训练日志文件
PRETRAINED_WEIGHTS = None
BEST_MODEL_FILE = "weights/best_20231120.pth"  # 最佳模型文件
LAST_MODEL_FILE = "weights/last_20231120.pth"  # 最后模型文件

os.makedirs('weights', exist_ok=True)

img_mean = [0.6800340224826795, 0.6800340224826795, 0.6800340224826795]
img_std = [0.2612217413908067, 0.2612217413908067, 0.2612217413908067]


logging.basicConfig(
    filename=log_file,
    filemode="a",
    # level=logging.DEBUG,
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(message)s",
)


# 定义一个函数，从文件名中提取标签
def get_label(image_f):
    match = re.search(r"chr[0-9XY]+", os.path.basename(image_f))
    if match:
        label = match.group()
    else:
        # label = "unknown"
        raise ValueError(f'{image_f}: no label found.')
    return label


CHR_IDS = [i for i in range(24)]
CHR_LABELS = [f'chr{i}' for i in range(1, 23)]
CHR_LABELS.extend(['chrX', 'chrY'])
assert len(CHR_IDS) == 24
assert len(CHR_LABELS) == 24

CHR_ID_TO_LABEL_DICT = {}
for i, k in enumerate(CHR_IDS):
    CHR_ID_TO_LABEL_DICT[k] = CHR_LABELS[i]
    
    
CHR_LABEL_TO_ID_DICT = {}
for i, k in enumerate(CHR_LABELS):
    CHR_LABEL_TO_ID_DICT[k] = i


# 定义一个函数，将标签转换为数字
def label_to_num(label):
    return int(CHR_LABEL_TO_ID_DICT[label])


# 定义一个函数，将数字转换为标签
def num_to_label(num):
    return str(CHR_ID_TO_LABEL_DICT[num])


# 定义一个自定义的数据集类，继承自torch.utils.data.Dataset
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        super(CustomDataset, self).__init__()
        # 重写samples属性，将标签从文件名中提取并转换为数字
        self.samples = [
            (path, label_to_num(get_label(path)))
            for path in glob.glob(f'{root}/*.png')
        ]
        self.transform = transform  # 保存数据预处理的方法

    def __len__(self):  # 定义一个函数，返回数据集的长度
        return len(self.samples)

    def __getitem__(self, index):  # 定义一个函数，根据索引返回一个数据样本
        path, label = self.samples[index]  # 从samples中获取图片路径和标签数据
        image = Image.open(path).convert('RGB')  # 打开图片文件
        if self.transform:  # 如果有数据预处理的方法
            image = self.transform(image)  # 对图片进行预处理
        return image, label  # 返回一个元组


# 定义数据预处理
transform = transforms.Compose(
    [
        transforms.RandomRotation(30, fill=255, expand=True),  # 随机旋转，角度为正负180度以下，旋转时使用白色补齐图像
        transforms.RandomHorizontalFlip(0.5), # 随机水平翻转
        transforms.RandomVerticalFlip(0.5),
        transforms.Resize((224, 224)),  # 图片resize为224， 224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(img_mean, img_std),  # 图片的灰度值做normalize
    ]
)
   
# 创建数据集对象
train_dataset = CustomDataset(train_dir, transform)
val_dataset = CustomDataset(val_dir, transform)
test_dataset = CustomDataset(test_dir, transform)

# 创建数据加载器对象
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)


# 定义模型
model = torchvision.models.resnet18(weights=PRETRAINED_WEIGHTS)  # 使用预训练的RESNET18模型
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # 替换最后一层为适合分类任务的全连接层

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
)  # 使用随机梯度下降优化器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)  # 使用余弦退火策略设置学习率


# 定义一个函数，计算准确率
def accuracy(outputs, labels):
    # outputs是模型的输出，形状为(batch_size, num_classes)
    # labels是真实的标签，形状为(batch_size,)
    # 返回一个标量，表示准确率
    _, preds = torch.max(outputs, 1) # 取每行的最大值，返回最大值和对应的索引
    return torch.sum(preds == labels).item() # 计算预测正确的个数


# 定义一个函数，加载模型的权重
def load_model(model, name):
    model.load_state_dict(torch.load(name + ".pth"))


# 定义一个函数，训练模型
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    patience,
):
    # 初始化一些变量
    best_acc = 0.0  # 记录最佳的验证集准确率
    best_epoch = 0  # 记录最佳的验证集准确率对应的轮次
    counter = 0  # 记录早停的计数器
    train_acc_list = []  # 记录每轮的训练集准确率
    val_acc_list = []  # 记录每轮的验证集准确率
    train_loss_list = []  # 记录每轮的训练集损失
    val_loss_list = []  # 记录每轮的验证集损失

    # 使用DDP的方式进行并行训练
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 开始训练
    for epoch in range(num_epochs):
        # 训练阶段
        model.train() # 设置模型为训练模式
        train_loss = 0.0 # 记录训练集的总损失
        train_acc = 0.0 # 记录训练集的总准确率
        for inputs, labels in train_loader: # 遍历训练集的每个批次
            inputs = inputs.to(device) # 将输入数据移动到设备上
            labels = labels.to(device) # 将标签数据移动到设备上
            optimizer.zero_grad() # 清空梯度
            outputs = model(inputs) # 前向传播，得到输出
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播，计算梯度
            optimizer.step() # 更新参数
            train_loss += loss.item() * inputs.size(0) # 累加损失
            train_acc += accuracy(outputs, labels) * inputs.size(0) # 累加准确率
        train_loss = train_loss / len(train_loader.dataset) # 计算训练集的平均损失
        train_acc = train_acc / len(train_loader.dataset) # 计算训练集的平均准确率
        train_acc_list.append(train_acc) # 将训练集的平均准确率添加到列表中
        train_loss_list.append(train_loss) # 将训练集的平均损失添加到列表中
    
       # 验证阶段
        model.eval() # 设置模型为评估模式
        val_loss = 0.0 # 记录验证集的总损失
        val_acc = 0.0 # 记录验证集的总准确率
        with torch.no_grad(): # 不计算梯度，节省内存
            for inputs, labels in val_loader: # 遍历验证集的每个批次
                inputs = inputs.to(device) # 将输入数据移动到设备上
                labels = labels.to(device) # 将标签数据移动到设备上
                outputs = model(inputs) # 前向传播，得到输出
                loss = criterion(outputs, labels) # 计算损失
                val_loss += loss.item() * inputs.size(0) # 累加损失
                val_acc += accuracy(outputs, labels) * inputs.size(0) # 累加准确率
        val_loss = val_loss / len(val_loader.dataset) # 计算验证集的平均损失
        val_acc = val_acc / len(val_loader.dataset) # 计算验证集的平均准确率
        val_acc_list.append(val_acc) # 将验证集的平均准确率添加到列表中
        val_loss_list.append(val_loss) # 将验证集的平均损失添加到列表中

        # 调整学习率
        scheduler.step(val_loss)

        # 打印训练和验证的结果
        logging.info(f'Epoch {epoch+1}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

        # 保存最佳的模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            counter = 0 # 重置早停的计数器
            logging.info(f"Best model saved at epoch {best_epoch}")  # 写入日志文件
            torch.save(model.state_dict(), 'best_model.pth') # 保存模型的参数
        else:
            counter += 1 # 递增早停的计数器
        
        # 判断是否达到早停的条件
        if counter == patience:
            logging.info(f'Early stopping at epoch {epoch+1}, the best epoch is {best_epoch}, the best val acc is {best_acc:.4f}')
            break
    
    # 返回训练和验证的结果
    return train_acc_list, val_acc_list, train_loss_list, val_loss_list


# 绘制训练和验证的损失和准确率的曲线
def plot_results(train_acc_list, val_acc_list, train_loss_list, val_loss_list):
    # 设置绘图的风格和大小
    sns.set_style('darkgrid')
    plt.figure(figsize=(12, 8))

    # 绘制训练和验证的损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(train_loss_list, label='train loss')
    plt.plot(val_loss_list, label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # 绘制训练和验证的准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(train_acc_list, label='train acc')
    plt.plot(val_acc_list, label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    # 保存和显示图片
    plt.savefig('results.png')
    plt.show()


# 加载最佳的模型参数，对测试集进行预测和评估
def test_model(model, test_loader, criterion, device):
    # 加载最佳的模型参数
    model.load_state_dict(torch.load('best_model.pth'))

    # 设置模型为评估模式
    model.eval()

    # 初始化一些变量
    test_loss = 0.0 # 记录测试集的总损失
    test_acc = 0.0 # 记录测试集的总准确率
    y_true = [] # 记录测试集的真实标签
    y_pred = [] # 记录测试集的预测标签

    # 不计算梯度，节省内存
    with torch.no_grad():
        # 遍历测试集的每个批次
        for inputs, labels in test_loader:
            # 将输入数据移动到设备上
            inputs = inputs.to(device)
            # 将标签数据移动到设备上
            labels = labels.to(device)
            # 前向传播，得到输出
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 累加损失
            test_loss += loss.item() * inputs.size(0)
            # 累加准确率
            test_acc += accuracy(outputs, labels) * inputs.size(0)
            # 将真实标签和预测标签添加到列表中
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
    
    # 计算测试集的平均损失和准确率
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_acc / len(test_loader.dataset)

    # 打印测试集的结果
    print(f'Test loss: {test_loss:.4f}, test acc: {test_acc:.4f}')

    # 返回真实标签和预测标签
    return y_true, y_pred


# 分析模型的预测结果
def analyze_results(y_true, y_pred):
    # 导入一些分析的库

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 绘制混淆矩阵的热力图
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # 打印分类报告
    print(classification_report(y_true, y_pred))


# 保存模型的整体结构和参数
def save_model(model, path):
    # 保存模型的整体结构和参数
    torch.save(model, path)


# 调用上面定义的函数
# 训练模型
train_acc_list, val_acc_list, train_loss_list, val_loss_list = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    patience,
)

# 保存模型
save_model(model, BEST_MODEL_FILE)

    
# 绘制训练和验证的结果
plot_results(train_acc_list, val_acc_list, train_loss_list, val_loss_list)

# 测试模型
y_true, y_pred = test_model(model, test_loader, criterion, 'cuda:0')

# 分析模型的结果
analyze_results(y_true, y_pred)
