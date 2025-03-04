import torch
torch.manual_seed(0)

# 原始矩阵
x = torch.randn((8, 8), dtype=torch.float16)
print("原始矩阵:")
print(x)
print("形状:", x.shape)

# 提取 4x4 块并展平
_x = x.view(2,4,2,4)
_x = _x.permute(0,2,1,3)
_x = _x.contiguous().view(-1, 4*4)
# _x = x.unfold(0,4,4).unfold(1,4,4)
# _x = _x.contiguous().view(-1, 4*4)

print("\n展平后的矩阵:")
print(_x)
print("形状:", _x.shape)

y = _x.amax(dim=1, keepdim=True).expand_as(_x)
# 将展平后的矩阵还原为原始形状
y = y.view(2, 2, 4, 4)      # 调整为 (2, 2, 4, 4) 的分块结构
y = y.permute(0, 2, 1, 3)  # 调整维度顺序为 (2, 4, 2, 4)
y = y.contiguous().view(8, 8)  # 合并分块为 8x8 矩阵
y = y[:, ::4]
print(y)

z = x.view(-1, 4)


# 将展平后的矩阵还原为原始形状
restored = _x.view(2, 2, 4, 4)      # 调整为 (2, 2, 4, 4) 的分块结构
restored = restored.permute(0, 2, 1, 3)  # 调整维度顺序为 (2, 4, 2, 4)
restored = restored.contiguous().view(8, 8)  # 合并分块为 8x8 矩阵

print("\n还原后的矩阵:")
print(restored)
print("形状:", restored.shape)


# 检查是否恢复正确
print("\n恢复是否正确:", torch.allclose(x, restored, atol=1e-5))