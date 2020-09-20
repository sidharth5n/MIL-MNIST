import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):

    def __init__(self, L, D, K):
        super(GatedAttention, self).__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.fc = nn.Linear(D, K)

    def forward(self, x):
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.fc(A_V * A_U) #NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim = 1) #KxN
        M = torch.mm(A, x).squeeze(0)  # KxL
        return M

class Attention(nn.Module):

    def __init__(self, L, D, K):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(L, D)
        self.non_linearity = nn.Tanh()
        self.fc2 = nn.Linear(D, K)

    def forward(self, x):
        A = self.fc1(x)
        A = self.non_linearity(A)
        A = self.fc2(A)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim = 1)
        M = torch.mm(A, x).squeeze(0)
        return M

class NoisyAnd(nn.Module):

    def __init__(self, a = 10, dims = [0]):
        super(NoisyAnd, self).__init__()
        # slope of the activation
        self.a = a
        # adaptable soft threshold
        self.b = nn.Parameter(torch.tensor(0.01))
        self.dims = dims
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean = torch.mean(x, self.dims, False)
        res = (self.sigmoid(self.a * (mean - self.b)) - self.sigmoid(-self.a * self.b)) / (
              self.sigmoid(self.a * (1 - self.b)) - self.sigmoid(-self.a * self.b))
        return res

class NoisyOr(nn.Module):

    def __init__(self, dims = 0):
        super(NoisyOr, self).__init__()
        self.dims = dims

    def forward(self, x):
        x = 1 - torch.prod(1 - x, dim = self.dims)
        return x

class ISR(nn.Module):

    def __init__(self, dims = 0):
        super(ISR, self).__init__()
        self.dims = dims

    def forward(self, x):
        w = x / (1 - x)
        y = 1 + torch.sum(w, dim = self.dims)
        z = torch.sum(w / y, dim = self.dims)
        return z

class GeneralizedMean(nn.Module):

    def __init__(self, dims = 0):
        super(GeneralizedMean, self).__init__()
        # self.r = nn.Parameter(torch.tensor(2.5))
        self.r = 2.5 #1, 2.5, 5
        self.dims = dims

    def forward(self, x):
        y = torch.pow(torch.abs(x), self.r)
        y = torch.mean(y, dim = self.dims)
        y = torch.pow(y, 1/self.r)
        return y

class LSE(nn.Module):

    def __init__(self, dims = 0):
        super(LSE, self).__init__()
        # self.r = nn.Parameter(torch.tensor(2.5))
        self.r = 2.5 #1, 2.5, 5
        self.dims = dims

    def forward(self, x):
        y = torch.exp(self.r * x)
        y = torch.mean(y, dim = self.dims)
        y = torch.log(y + 1e-10) / self.r
        return y

class MIL(nn.Module):

    def __init__(self, aggregation = 'attention'):
        super(MIL, self).__init__()
        L = 500
        D = 128
        K = 1

        self.feature_extractor1 = nn.Sequential(nn.Conv2d(1, 20, kernel_size = 5),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2, stride = 2),
                                                nn.Conv2d(20, 50, kernel_size = 5),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2, stride = 2),)

        self.feature_extractor2 = nn.Sequential(nn.Linear(50 * 4 * 4, L),
                                                nn.ReLU())

        if aggregation == 'attention':
            print('Attention MIL')
            self.aggregation = Attention(L, D, K)
        elif aggregation == 'gated attention':
            print('Gated attention MIL')
            self.aggregation = GatedAttention(L, D, K)
        elif aggregation == 'noisy and':
            print('Noisy AND MIL')
            self.aggregation = NoisyAnd()
        elif aggregation == 'noisy or':
            print('Noisy OR MIL')
            self.aggregation = NoisyOr()
        elif aggregation == 'isr':
            print('ISR MIL')
            self.aggregation = ISR()
        elif aggregation == 'generalized mean':
            print('Generalized mean MIL')
            self.aggregation = GeneralizedMean()
        else:
            print('LSE MIL')
            self.aggregation = LSE()

        self.classifier = nn.Sequential(nn.Linear(L * K, 1),
                                        nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)
        # feature extraction
        H = self.feature_extractor1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor2(H)
        # aggregation
        M = self.aggregation(H)
        # classification
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat
