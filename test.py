# import debugpy
# debugpy.listen(('127.0.0.1', 9901))
# print('Waiting for debugger attach')
# debugpy.wait_for_client()

# import torch
# x = torch.zeros(1, 2, 3)
# a = torch.randn(1, 2, 3)
# b = torch.randn(1, 3, 3)

# N, L, D = x.shape

# keep = 1
# patch_mask = torch.rand(N, L, device=x.device)
# patch_mask = torch.argsort(patch_mask, dim=1)
# patch_mask = patch_mask[:, :keep]

# a1 = torch.gather(a, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))
# b1 = torch.gather(b, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D) + 1)

# print(123)
def printT(x):
    for name, value in globals().items():
        if value is x:
            print("{}:\n{}\n{}\n------\n".format(name, x.shape, x))
    
import torch

# 创建一个 2D 张量 input
input = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
B, T, N, N = input.shape
keep = 2
printT(input)

index = torch.tensor([[0,1],[0,2]])
printT(index)
row = index.unsqueeze(1).unsqueeze(-1).repeat(1, T, 1, N)
printT(row)
x = torch.gather(input, dim=2, index=row)
printT(x)


col = index.unsqueeze(1).unsqueeze(1).repeat(1, T, keep, 1)
printT(col)
x = torch.gather(x, dim=3, index=col)
printT(x)

# output = input.index_select(1,index)

# printT(output)

# import torch

# # 创建一个3x3的矩阵
# matrix = torch.tensor([[1, 2, 3],
#                         [4, 5, 6],
#                         [7, 8, 9]])
# printT(matrix)
# # 使用index选择行
# rows = torch.tensor([0, 2])  # 索引为0的行是第1行，索引为2的行是第3行
# selected_rows = matrix.index_select(0, rows)  # 0代表选择第0维，即行

# # 使用gather提取行
# indices = rows.view(-1, 1).expand(-1, matrix.size(1))  # 扩展索引以匹配列数
# selected_rows_gather = matrix.gather(0, indices)

# print("Selected Rows using index_select:\n", selected_rows)
# print("Selected Rows using gather:\n", selected_rows_gather)