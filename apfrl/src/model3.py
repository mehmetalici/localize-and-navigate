import torch
import torch.nn as nn
class PyTorchMlp(nn.Module):  

  def __init__(self, n_inputs=132, n_actions=3):
      nn.Module.__init__(self)

      self.fc1 = nn.Linear(n_inputs, 512)
      self.fc2 = nn.Linear(512, 512)    
      self.fc3 = nn.Linear(512, 512)  
      self.mu_head = nn.Linear(512, n_actions)  
      #self.log_std_head = nn.Linear(512, n_actions)
      self.activ_fn = nn.ReLU()
      self.out_activ = nn.Tanh()

  def forward(self, x):
      x = self.activ_fn(self.fc1(x))
      x = self.activ_fn(self.fc2(x))
      x = self.activ_fn(self.fc3(x))
      mu_x = self.out_activ(self.mu_head(x))
      #std_x = self.out_activ(self.log_std_head(x))
      return mu_x
