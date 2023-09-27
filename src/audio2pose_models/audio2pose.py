import torch
from torch import nn
from src.audio2pose_models.cvae import CVAE
from src.audio2pose_models.discriminator import PoseSequenceDiscriminator
from src.audio2pose_models.audio_encoder import AudioEncoder

class Audio2Pose(nn.Module):
    def __init__(self, cfg, wav2lip_checkpoint, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.MODEL.CVAE.SEQ_LEN
        self.latent_dim = cfg.MODEL.CVAE.LATENT_SIZE
        self.device = device

        # 创建音频编码器模块
        self.audio_encoder = AudioEncoder(wav2lip_checkpoint, device)
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # 创建条件变分自编码器（CVAE）模型
        self.netG = CVAE(cfg)
        
        # 创建动作序列判别器模型
        self.netD_motion = PoseSequenceDiscriminator(cfg)
        
        
    def forward(self, x):

        batch = {}
        coeff_gt = x['gt'].cuda().squeeze(0)           # 获取真实姿势系数（ground truth） bs frame_len+1 73
        batch['pose_motion_gt'] = coeff_gt[:, 1:, 64:70] - coeff_gt[:, :1, 64:70] # 计算动作变化 bs frame_len 6
        batch['ref'] = coeff_gt[:, 0, 64:70]  # 获取参考姿势系数 bs  6
        batch['class'] = x['class'].squeeze(0).cuda() # 获取类别信息 bs
        indiv_mels= x['indiv_mels'].cuda().squeeze(0) # 获取个体音频特征 bs seq_len+1 80 16

        # 前向传播
        audio_emb_list = []
        audio_emb = self.audio_encoder(indiv_mels[:, 1:, :, :].unsqueeze(2)) # 提取音频特征 bs seq_len 512
        batch['audio_emb'] = audio_emb
        batch = self.netG(batch)

        pose_motion_pred = batch['pose_motion_pred']           # 预测的动作变化 bs frame_len 6
        pose_gt = coeff_gt[:, 1:, 64:70].clone()               # 真实动作变化 bs frame_len 6
        pose_pred = coeff_gt[:, :1, 64:70] + pose_motion_pred  # 预测的姿势系数 bs frame_len 6

        batch['pose_pred'] = pose_pred
        batch['pose_gt'] = pose_gt

        return batch

    def test(self, x):

        batch = {}
        ref = x['ref']                            # 获取参考姿势系数 bs 1 70
        batch['ref'] = x['ref'][:,0,-6:]  
        batch['class'] = x['class']  
        bs = ref.shape[0]
        
        indiv_mels= x['indiv_mels']               # 获取个体音频特征 bs T 1 80 16
        indiv_mels_use = indiv_mels[:, 1:]        # 将参考帧视为第一帧
        num_frames = x['num_frames']
        num_frames = int(num_frames) - 1

        # 计算序列划分
        div = num_frames//self.seq_len
        re = num_frames%self.seq_len
        audio_emb_list = []
        pose_motion_pred_list = [torch.zeros(batch['ref'].unsqueeze(1).shape, dtype=batch['ref'].dtype, 
                                                device=batch['ref'].device)]

        for i in range(div):
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, i*self.seq_len:(i+1)*self.seq_len,:,:,:]) # 提取音频特征 bs seq_len 512
            batch['audio_emb'] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(batch['pose_motion_pred'])  # 列表中包含 bs seq_len 6
        
        if re != 0:
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, -1*self.seq_len:,:,:,:]) # 提取音频特征 bs seq_len  512
            if audio_emb.shape[1] != self.seq_len:
                pad_dim = self.seq_len-audio_emb.shape[1]
                pad_audio_emb = audio_emb[:, :1].repeat(1, pad_dim, 1) 
                audio_emb = torch.cat([pad_audio_emb, audio_emb], 1) 
            batch['audio_emb'] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(batch['pose_motion_pred'][:,-1*re:,:])   
        
        pose_motion_pred = torch.cat(pose_motion_pred_list, dim = 1)
        batch['pose_motion_pred'] = pose_motion_pred

        pose_pred = ref[:, :1, -6:] + pose_motion_pred  # 预测的姿势系数 bs T 6

        batch['pose_pred'] = pose_pred
        return batch
