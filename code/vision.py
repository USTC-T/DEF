from model import DEANet
from torchinfo import summary
net = DEANet(base_dim=32)
summary(net, (1, 3, 400, 400))
