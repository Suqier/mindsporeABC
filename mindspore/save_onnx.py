# 导出onnx模型
import mindspore as ms
from mindspore import Tensor, export, nn, train
import numpy as np
from model import resnet50

best_ckpt_path = "./resnet50-best.ckpt"

if __name__ == "__main__":
    net = resnet50()
    # 全连接层输入层的大小
    in_channels = net.fc.in_channels
    head = nn.Dense(in_channels, 5)
    # 重置全连接层
    net.fc = head
    # 平均池化层kernel size为7
    avg_pool = nn.AvgPool2d(kernel_size=7)
    # 重置平均池化层
    net.avg_pool = avg_pool
    # 加载模型参数
    param_dict = ms.load_checkpoint(best_ckpt_path)
    ms.load_param_into_net(net, param_dict)

    input = np.ones([128, 3, 224, 224]).astype(np.float32)
    export(net, Tensor(input), file_name='resnet50_onnx', file_format='ONNX')
