from torch import nn
import torch
from app import logger

torch.set_default_dtype(torch.float)
	
class VariableNet(nn.Module):

	def __init__(self, input_size, num_layers, embedding_size):
		if not type(num_layers) == int or num_layers < 1:
			raise ValueError("num_layers must be an integer greater than 0")
		if not type(embedding_size) == int or embedding_size < 1:
			raise ValueError("embedding_size must be an integer greater than 0")
		super(VariableNet, self).__init__()
		self.act = nn.SiLU()
		
		self.layers = nn.Sequential()
		self.layers.add_module("fc0", nn.Linear(input_size, embedding_size))
		for i in range(1, num_layers+1):
			self.layers.add_module(f"act{i}", self.act)
			self.layers.add_module(f"fc{i+1}", nn.Linear(embedding_size, embedding_size))
			
		
	def forward_once(self, x) -> torch.Tensor:
		x = self.layers(x)
		x = torch.nn.functional.normalize(x, p=2, dim=1)
		return x
	
	def forward(self, triplet) -> tuple:
		ancor, positive, negative = triplet[0], triplet[1], triplet[2]
		output1 = self.forward_once(ancor)
		output2 = self.forward_once(positive)
		output3 = self.forward_once(negative)
		return output1, output2, output3
