from torch import nn
from torch.nn import functional as F

################################################
#         Dense only (few layers)              #
################################################

class Net_0(nn.Module):
      def __init__(self, hidden_units, img_size=(14,14), batch_normalization=False):
        super().__init__()
        self.img_vect_length = img_size[0]*img_size[1]
        self.fc1 = nn.Linear(self.img_vect_length, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.fc3 = nn.Linear(hidden_units, 10)

        self.batch_normalization = batch_normalization

      def forward(self, x):
        out = F.relu(self.fc1(x.view(-1, self.img_vect_length)))
        if self.batch_normalization:
          out = self.bn1(out)
        out = self.fc2(out)
        if self.batch_normalization:
          out = self.bn2(out)
        out = self.fc3(out)
        return out

################################################
#      Conv + dense (few layers)               #
################################################

class Net_1(nn.Module):

    def __init__(self, hidden_units, batch_normalization=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.bn2   = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64, hidden_units)
        self.bn3 = nn.BatchNorm1d(hidden_units)
        self.fc2 = nn.Linear(hidden_units, 10)

        self.batch_normalization = batch_normalization

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=3))
        if self.batch_normalization:
          out = self.bn1(out)
        out = F.relu(F.max_pool2d(self.conv2(out), kernel_size=2, stride=2))
        if self.batch_normalization:
          out = self.bn2(out)
        
        out = F.relu(self.fc1(out.view(-1, 64)))
        if self.batch_normalization:
          out = self.bn3(out)
        out = self.fc2(out)
        return out
        # return F.log_softmax(out, dim=1)


################################################
#        Conv + dense (many layers)            #
################################################

class Net_2(nn.Module):

    def __init__(self, hidden_units, batch_normalization=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.bn3   = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64, hidden_units)
        self.bn4   = nn.BatchNorm1d(hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn5   = nn.BatchNorm1d(hidden_units)
        self.fc3 = nn.Linear(hidden_units, 10)

        self.batch_normalization = batch_normalization

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
        if self.batch_normalization:
          out = self.bn1(out)
        out = F.relu(F.max_pool2d(self.conv2(out), kernel_size=2, stride=3))
        if self.batch_normalization:
          out = self.bn2(out)
        out = F.relu(F.max_pool2d(self.conv3(out), kernel_size=2, stride=2))
        if self.batch_normalization:
          out = self.bn3(out)

        out = F.relu(self.fc1(out.view(-1, 64)))
        if self.batch_normalization:
          out = self.bn4(out)
        out = self.fc2(out)
        if self.batch_normalization:
          out = self.bn5(out)
        out = self.fc3(out)
        return out