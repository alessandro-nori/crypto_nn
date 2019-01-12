import torch
from random import randint

# relu approximation coefficients
a = 0.0012
b = 0.5
c = 52

def generate_data_twolines():
    x1 = randint(0,30)
    x2 = randint(0, 30)
    if x1*m1+b1>=x2 and x1*m2+b2<=x2:
        y = torch.tensor(10, dtype=torch.float32)
        # print(x1, " ", x2, " ", y.item())
    else:
        y = torch.tensor(0, dtype=torch.float32)
    input = torch.tensor([x1, x2], dtype=torch.float32)

    return input, y

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        relu = ReLUApproxim.apply
        h_relu = relu(self.linear1(x))

        # h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

    def get_parameters(self):
        w = []
        w.append(list(self.linear1.parameters()))
        w.append(list(self.linear2.parameters()))
        return w

class ReLUApproxim(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return a*(x**2)+b*x+c

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = 2*c*x+b
        return grad_input

# N is the batch size
# D_in is input dimension
# D_out is output dimension
# H is hidden dimension
N, D_in, H, D_out = 1, 2, 2, 1

model = TwoLayerNet(D_in, H, D_out)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

m1 = 1;
b1 = 0;
m2 = -1;
b2 = 10;

for t in range(20000):
    if (t+1)%10000 == 0:
        print(t+1)

    input, y = generate_data_twolines()
    # input = [[[0,0],[0,1],[1,0],[1,0]]]

    y_pred = model(input)

    loss = loss_fn(y_pred, y)
    # print('epoch: ', t,' loss: ', loss.item(), ' y_pred: ', y_pred.item(), ' y: ', y.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

loss = 0
for i in range(200):
    input, y = generate_data_twolines()
    y_pred = model(input)
    loss += loss_fn(y_pred, y)
    print('loss: ', loss.item(), ' y_pred: ', y_pred.item(), ' y: ', y.item())

print("loss: ", loss.item()/1000)

print('model parameters:')
print(model.get_parameters())
