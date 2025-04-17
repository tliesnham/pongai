import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv_output_size = self._get_conv_output((3, 84, 84))
        self.fc = nn.Linear(self.conv_output_size, num_classes)
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv_layers(dummy_input)
            return output.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x