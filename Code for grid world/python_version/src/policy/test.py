import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
index = torch.tensor([[0, 1],
                      [1, 2]])

# 写法一：张量方法形式
result1 = x.gather(1, index)  # 在维度1（列方向）上收集

# 写法二：函数形式
result2 = torch.gather(x, 1, index)
x_max = x.max(1)

print("张量方法结果:\n", result1)
print("函数形式结果:\n", result2)
print("结果是否相同:", torch.equal(result1, result2))