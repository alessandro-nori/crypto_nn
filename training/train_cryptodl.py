import sys

import matplotlib.pyplot as plt
import numpy as np

import torch

#ReLU approximation coefficients
a = 0.1524
b = 0.5
c = 0.409

class ReLUApproxim(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return a*(x**2)+b*x+c

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = grad_input*(2*a*x + b)
        return grad_input

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        relu = ReLUApproxim.apply
        h_relu = relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred

    def get_parameters(self):
        w = []
        w.append(list(self.linear1.parameters()))
        w.append(list(self.linear2.parameters()))
        return w

    def save_parameters(self, filename):
        # layer0
        params = list(self.linear1.parameters())
        w1 = params[0][0].detach().numpy()
        w2 = params[0][1].detach().numpy()
        b1 = params[1][0].item()
        b2 = params[1][0].item()

        #layer1
        params = list(self.linear2.parameters())
        w3 = params[0][0].detach().numpy()
        b3 = params[1][0].detach().item()

        f = open(filename, 'w')
        f.write(str(w1[0]) + ' ' + str(w1[1]) + ' ' + str(b1) + '\n')
        f.write(str(w2[0]) + ' ' + str(w2[1]) + ' ' + str(b2) + '\n')
        f.write(str(w3[0]) + ' ' + str(w3[1]) + ' ' + str(b3) + '\n')
        f.close()


def load_data(filename):
    try:
        f = open(filename, "r")
    except FileNotFoundError:
        print("File not found!!!")
        exit()
    
    lines = f.readlines()
    X = []
    t = []

    for l in lines:
        x1, x2, y = [int(s) for s in l.strip().split(" ")]
        
        x = torch.tensor([x1, x2], dtype=torch.float32)
        y = torch.tensor([y], dtype=torch.float32)

        X.append(x)
        t.append(y)
    
    return X, t


def test(model, X, t):
    correct = 0

    for x, target in zip(X, t):
            pred = model(x)

            pred = 0 if pred.item() < 0.5 else 1
            if pred == target.item():
                correct+=1
    
    return correct*100/len(t)

def plot_predictions(model, X):
    # lines parameters
    m1 = -0.1
    b1 = 7
    m2 = 0.5
    b2 = 9

    predictions = []
    inputs = []

    for x in X:
        inputs.append(x.numpy())
        pred = model(x)
        pred = 'r' if pred.item() < 0.5 else 'b'
        predictions.append(pred)

    inputs = np.array(inputs)
    predictions = np.array(predictions)

    axis0 = np.linspace(0,20,200)
    line1 = axis0*m1+b1
    line2 = axis0*m2+b2
    line2[line2<0] = 0

    ax = plt.axes()
    ax.set(xlim=(0, 20), ylim=(0, 20), xlabel='x1', ylabel='x2')
    ax.plot(axis0, line1, 'r')
    ax.plot(axis0, line2, 'r')
    ax.scatter(inputs[:,0], inputs[:,1], c=predictions)
    axisx = np.linspace(0,20,200)
    ax.fill_between(axisx, 0, 20,
                    color='blue', alpha=0.2)
    ax.fill_between(axis0, line2, line1,
                    color='red', alpha=0.2)

    plt.show()


def main():

    if len(sys.argv) > 1:
        # output file name
        fout = sys.argv[1]
    else:
        fout = 'weights.txt'

    # D_in is input dimension
    # D_out is output dimension
    # H is hidden dimension
    D_in, H, D_out = 2, 2, 1

    model = TwoLayerNet(D_in, H, D_out)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X, t = load_data('train.data')
    X_test, t_test = load_data('test.data')
    print('Training and Test data loaded')
    accuracy = 0.00

    print('Training phase...')
    epoch = 0
    while accuracy < 95.0:
        for x, target in zip(X, t):
            pred = model(x)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        accuracy = test(model, X_test, t_test)
        epoch+=1
        print('Epoch', epoch, 'accuracy:', accuracy)


    model.save_parameters(fout)

    plot_predictions(model, X_test)

    


if __name__ == '__main__':
    main()