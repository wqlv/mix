import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from tensorboardX import SummaryWriter

# 导入ResNet模型，可以使用预训练的ResNet-50
import model as models

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class CustomResNet(nn.Module):
    def __init__(self, num_classes, num_spectral_features):
        super(CustomResNet, self).__init__()
        # 加载预训练的ResNet模型，这里以ResNet-50为例
        self.resnet = models.resnet50(pretrained=True)

        # 修改最后一层全连接层，以适应您的任务
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

        # 光谱信息处理层，这里使用全连接层，您可以根据需要自定义
        self.spectral_fc = nn.Linear(num_spectral_features, num_classes)

    def forward(self, images, spectra):
        # 将SEM图像输入ResNet
        x_image = self.resnet(images)

        # 将光谱数据输入光谱信息处理层
        x_spectral = self.spectral_fc(spectra)

        # 将两个输入的结果相加，或者采用其他方式融合
        x = x_image + x_spectral

        return x


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用 {} 设备.".format(device))

    # 定义数据预处理的transform
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "data_set", "Sem")
    assert os.path.exists(image_path), "{} 路径不存在.".format(image_path)

    # 创建自定义数据集类，包括SEM图像和光谱数据
    from your_dataset import CustomSemSpectrumDataset  # 请替换成您自定义的数据集类
    train_dataset = CustomSemSpectrumDataset(image_path, data_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    print("用于训练的图像数量：{}".format(len(train_dataset)))

    # 创建您自定义的深度学习模型，传入图像和光谱特征的维度
    num_classes = 4  # 根据您的任务定义类别数
    num_spectral_features = 10  # 根据您的光谱数据维度定义
    model = CustomResNet(num_classes, num_spectral_features)
    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 500
    best_acc = 0.0
    save_path = './your_model.pth'  # 请替换成您希望保存模型的路径
    train_steps = len(train_loader)

    # 创建SummaryWriter对象
    writer = SummaryWriter(log_dir='./logs')

    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, spectra, labels = data  # 修改以加载光谱信息

            optimizer.zero_grad()
            logits = model(images.to(device), spectra.to(device))  # 将SEM图像和光谱信息输入模型
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "训练 第[{}/{}] 轮 loss:{:.3f}".format(epoch + 1, epochs, loss)

        # 在此处添加验证步骤，评估模型性能

        # 将训练损失写入SummaryWriter
        writer.add_scalar('训练损失', running_loss / train_steps, epoch)

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), save_path)

    writer.close()
    print('训练完成')


if __name__ == '__main__':
    main()
