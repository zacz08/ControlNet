import re
import matplotlib.pyplot as plt

# 读取日志文件
log_file = 'your_log_file.log'  # 将这个替换为你的实际日志文件名

# 打开并读取日志文件内容
with open(log_file, 'r') as f:
    log_data = f.read()

# 使用正则表达式提取每一行中的 loss 值
loss_pattern = re.compile(r'loss=([0-9.]+)')
losses = loss_pattern.findall(log_data)

# 将loss值转换为浮点数
losses = [float(loss) for loss in losses]

# 计算总共多少个iterations
total_iterations = len(losses)

# 绘制曲线图
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss')

# 添加每142个iteration处的竖线
epoch_interval = 142  # 每142个iteration为一个epoch
for i in range(epoch_interval, total_iterations, epoch_interval):
    plt.axvline(x=i, color='r', linestyle='--', alpha=0.6)

# 设置图表的标签和标题
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss over Iterations')
plt.legend()
plt.grid(True)
plt.show()
