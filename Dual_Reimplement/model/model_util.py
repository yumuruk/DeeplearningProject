import torch
import torch.nn as nn

def extract_channel_statistics(batch):
    batch_size = batch.shape[0]
    
    sorted_channel_means = []       # 각 이미지의 채널 평균 (정렬된)
    sorted_channel_indices = []     # 평균 정렬 순서에 대한 채널 인덱스

    large_channel_idx = []
    medium_channel_idx = []
    small_channel_idx = []

    large_channel_val = []
    medium_channel_val = []
    small_channel_val = []

    for batch_index in range(batch_size):
        image = batch[batch_index, :, :, :]  # shape: [C, H, W]
        mean = torch.mean(image, (2, 1))     # shape: [C]
        
        mean_sorted, channel_sorted_indices = torch.sort(mean)  # 오름차순 정렬
        sorted_channel_means.append(mean_sorted)
        sorted_channel_indices.append(channel_sorted_indices)

        large_channel_idx.append(channel_sorted_indices[2])
        medium_channel_idx.append(channel_sorted_indices[1])
        small_channel_idx.append(channel_sorted_indices[0])

        large_channel_val.append(image[channel_sorted_indices[2]].unsqueeze(0))
        medium_channel_val.append(image[channel_sorted_indices[1]].unsqueeze(0))
        small_channel_val.append(image[channel_sorted_indices[0]].unsqueeze(0))

    # 리스트 → 텐서 변환
    sorted_channel_means = torch.stack(sorted_channel_means)              # shape: [B, C]
    sorted_channel_indices = torch.stack(sorted_channel_indices)          # shape: [B, C]
    
    large_channel_idx = torch.stack(large_channel_idx)                    # shape: [B]
    medium_channel_idx = torch.stack(medium_channel_idx)
    small_channel_idx = torch.stack(small_channel_idx)

    large_channel_val = torch.stack(large_channel_val)                    # shape: [B, 1, H, W]
    medium_channel_val = torch.stack(medium_channel_val)
    small_channel_val = torch.stack(small_channel_val)
    
    return sorted_channel_means, sorted_channel_indices, large_channel_val, medium_channel_val, small_channel_val, large_channel_idx, medium_channel_idx, small_channel_idx

def replace_channel_by_index(J, J_m, channel_idx):
    batch_size = J.shape[0]
    mapped_J = []
    for batch_index in range(batch_size):
        image = J[batch_index, :, :, :] # batch size 내부에서 특정 이미지 선택
        image[channel_idx[batch_index]] = J_m[batch_index] # 
        mapped_J.append(image)
    mapped_J = torch.stack(mapped_J)  # shape: [B, C, H, W]
    return mapped_J    
    
    
def get_dark_channel(x, patch_size):
    pad_size = (patch_size - 1) // 2
    # Get batch size of input
    H, W = x.size()[2], x.size()[3]
    # Minimum among three channels
    x, _ = x.min(dim=1, keepdim=True)  # (B, 1, H, W) 수식에서 min_c을 의미함
    x = nn.ReflectionPad2d(pad_size)(x)  # (B, 1, H+2p, W+2p) path 연산 하기 위해 가장자리 padding 해줌
    # x = nn.ReflectionPad2d(pad_size)(x)  # 원래 official matlab code에서는 ReplicationPad2d 사용
    x = nn.Unfold(patch_size)(x) # (B, k*k, H*W) 각 픽셀을 기준으로 patch를 펼친 값
    x = x.unsqueeze(1)  # (B, 1, k*k, H*W) 나중에 min 및 index 계산을 위해 준비
    
    # Minimum in (k, k) patch
    index_map = torch.argmin(x, dim=2, keepdim=False) # (B, 1, H*W) 각 patch 내 최소값 위치 (0~H*W)
    dark_map, _ = x.min(dim=2, keepdim=False)  # (B, 1, H*W)
    dark_map = dark_map.view(-1, 1, H, W)

    return dark_map, index_map

# Soft thresholding function for tensor x and threshold lamda 
def softThresh(x, lamda):
    relu = nn.ReLU()
    return torch.sign(x).cuda() * relu(torch.abs(x).cuda() - lamda)