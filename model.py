import torch
import torch.nn as nn
import torch.nn.functional as F

class FirstModel(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.Flatten(x)
        x = self.l1(x)
        x = F.ReLU(x)
        x = self.l2(x)
        x = self.ReLU(x)
        logits = self.l3(x)
        return logits
    
if __name__ == "__main__":
    first_model = FirstModel(28*28, 50, 10)
    print(first_model)
        
        
        
        
        
