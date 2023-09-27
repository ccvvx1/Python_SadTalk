import torch
from torch import nn
from torch.nn import functional as F

# 定义一个用于创建卷积层的自定义模块
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 卷积块包含卷积层和批归一化层
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()  # ReLU激活函数
        self.residual = residual  # 是否使用残差连接

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x  # 如果启用了残差连接，则将输入添加到输出中
        return self.act(out)

# 定义一个音频编码器模块
class AudioEncoder(nn.Module):
    def __init__(self, wav2lip_checkpoint, device):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        # 加载预训练的音频编码器（注释部分是加载wav2lip模型的代码，根据需要取消注释）
        # wav2lip_state_dict = torch.load(wav2lip_checkpoint, map_location=torch.device(device))['state_dict']
        # state_dict = self.audio_encoder.state_dict()
        #
        # for k,v in wav2lip_state_dict.items():
        #     if 'audio_encoder' in k:
        #         state_dict[k.replace('module.audio_encoder.', '')] = v
        # self.audio_encoder.load_state_dict(state_dict)

    def forward(self, audio_sequences):
        # 输入音频序列的维度是 (B, T, 1, 80, 16)，其中 B 表示批量大小，T 表示时间步，1 表示通道数
        B = audio_sequences.size(0)

        # 将时间步的维度展平，将多个时间步的音频特征连接在一起
        audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)

        # 将展平后的音频序列通过音频编码器
        audio_embedding = self.audio_encoder(audio_sequences)  # 输出形状为 (B, 512, 1, 1)
        dim = audio_embedding.shape[1]
        audio_embedding = audio_embedding.reshape((B, -1, dim, 1, 1))

        # 压缩维度，去掉大小为1的维度，得到最终的音频编码表示
        return audio_embedding.squeeze(-1).squeeze(-1)  # 输出形状为 (B, seq_len+1, 512)
