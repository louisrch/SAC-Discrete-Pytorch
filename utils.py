import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import math
import torch
# model imports
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
import numpy as np
from PIL import Image
import argparse



QUERIES = {
    "CartPole-v1" : "What is in this picture ? The goal of the agent is to keep the pole upright. Is the pole upright in this picture ? If not, edit this picture, while preserving the proportions, such that the pole is upright. If it is already upright, do not do anything." 
}

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def save_goal_image(env):
	# whatever this is, fix later, too lazy rn ngl
	path = "goal.png"
	img = env.render()
	img = Image.fromarray(img)
	img.save(path)
	return path

# TODO : modularize
def get_goal_embedding(mode, env_name = "CartPole-v1", sys_path_to_goal= None):
	if mode == "image":
		img = Image.open(sys_path_to_goal)
		return get_image_embedding(img)
	elif mode == "text":
		query = QUERIES[env_name]
		return get_text_embedding(clip.tokenize([query]).to(device))



def get_goal_embedding(env, query = "a cartpole standing upright"):
    embedding = get_text_embedding(clip.tokenize([query]).to(device))
    return embedding



def get_text_embedding(tokens, model):
	with torch.no_grad():
		return model.encode_text(tokens)

def get_current_state_embedding(env):
	image = env.render()
	return get_image_embedding(image)


def get_image_embedding(image, model):
	"""
	encodes the image using the model
	image : input image
	model : encoding model
	credit : https://github.com/openai/CLIP
	"""
	with torch.no_grad():
		features = model.forward_image(model.get_fast_preprocessing(image))
		return features


def build_net(layer_shape, hid_activation, output_activation):
	'''build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = hid_activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Double_Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Double_Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]

		self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		q1 = self.Q1(s)
		q2 = self.Q2(s)
		return q1,q2


class Policy_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Policy_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]
		self.P = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		logits = self.P(s)
		probs = F.softmax(logits, dim=1)
		return probs


class ReplayBuffer(object):
	def __init__(self, state_dim, dvc, max_size=int(1e6)):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, 1),dtype=torch.long,device=self.dvc)
		self.r = torch.zeros((max_size, 1),dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1),dtype=torch.bool,device=self.dvc)
	
	def addAll(self, s_array, a_array,  r_array, s_next_array, dw_array):
		#print(len(s_array), len(a_array), len(r_array), len(s_next_array))
		idx = torch.arange(self.ptr, (self.ptr+len(a_array)) %self.max_size)
		self.s[idx] = torch.from_numpy(np.stack(s_array, axis=0)).to(self.dvc)
		self.a[idx] = torch.tensor(a_array).unsqueeze(-1).to(self.dvc)
		self.r[idx] = r_array.clone().detach().float().to(self.dvc)
		self.s_next[idx] = torch.from_numpy(np.stack(s_next_array, axis=0)).to(self.dvc)
		self.dw[idx] = torch.tensor(dw_array).unsqueeze(-1).to(self.dvc)
		self.ptr = (self.ptr + len(a_array)) % self.max_size
		self.size = min(self.size + len(idx), self.max_size)
		

	def add(self, s, a, r, s_next, dw):
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = a
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]


def evaluate_policy(env, agent, turns = 3):
	total_scores = 0
	for j in range(turns):
		s, info = env.reset()
		done = False
		while not done:
			# Take deterministic actions at test time
			a = agent.select_action(s, deterministic=True)
			s_next, r, dw, tr, info = env.step(a)
			done = (dw or tr)
			total_scores += r
			s = s_next
	return int(total_scores/turns)


#You can just ignore 'str2bool'. Is not related to the RL.
def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	

def compute_distance(a, b, dist_type = "euclidean"):
	"""
	distance between a and b is bounded between 0 and infty
	"""
	if dist_type == "euclidean":
		return torch._euclidean_dist(a,b)
	elif dist_type == "cosine":
		sim = torch.cosine_similarity(a,b)
		return (1 - sim) / (1+ sim + 1e-6)


def compute_reward(a, b, dist_type = "euclidean"):
	"""
	bijection from [0, infty) to [0,1)
	"""
	return torch.exp(-compute_distance(a,b,dist_type=dist_type))

def compute_rewards(rgb_imgs, goal, model):
	with torch.inference_mode():
		embeddings = model.forward_image(rgb_imgs)
		#print(embeddings.size(), goal.size(), rgb_imgs.size())
		rewards = compute_reward(embeddings, goal)
		# L2 norm squared
		return rewards


def dump_infos_to_replay_buffer(states, actions, depictions, dws, goal, agent, model):
	"""
	states : list of N+1 1xS numpy arrays
	actions : list of N actions
	depictions : list of N+1 depiction of states, 224x224x3 numpy arrays
	dws : list of episode completions
	goal : embedding of the goal image
	agent : agent at hand
	model : embedding model
	"""
	preprocessed_depictions = model.get_fast_preprocessing(np.stack(depictions, axis=0))
	rewards = compute_rewards(preprocessed_depictions, goal)
	next_states = states[1:]
	states = states[:-1]
	agent.replay_buffer.addAll(states, actions, rewards, next_states, dws)


class Model():
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		if self.model == "CLIP":
			model, _ = clip.load("ViT-B/32", device = self.dvc)
			self.model = model
			self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.dvc).view(1,3,1,1)
			self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.dvc).view(1,3,1,1)
		elif self.model == "DINOV2":
			model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
			self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dvc).view(1,3,1,1)
			self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dvc).view(1,3,1,1)
		self.img_size = 224

	def get_fast_preprocessing(self, img_np : np.ndarray, input_height = 224, input_width = 224):
		""""
		img_np : N x H x W x 3 RGB image
		output : N x 3 x input_height x input_width image
		"""
		img_tensor = torch.from_numpy(img_np)
		if img_tensor.ndim == 3:
			img_tensor = torch.unsqueeze(img_tensor, 0)
		img_tensor = img_tensor.permute(0,3,1,2).to(device, non_blocking=True).float().div_(255.)
		img_tensor = F.interpolate(img_tensor, size=(input_height, input_width), mode='bilinear', align_corners=False)
		img_tensor = (img_tensor - self.mean).div(self.std)
		return img_tensor

	def forward_image(self, image):
		if self.model == "CLIP":
			return self.encode_image(image)
		elif self.model == "DINOV2":
			return self.forward(image)


