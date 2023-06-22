import os
import json
from core.data_processor import DataLoad
from core.model import Deblur
from core.early_stopping import EarlyStopping
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import math

unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
	image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
	image = image.squeeze(0)  # remove the fake batch dimension
	image = unloader(image)
	plt.imshow(image)
	if title is not None:
		plt.title(title)
	plt.pause(0.001)  # pause a bit so that plots are updated

def tensor_to_np(tensor):
	img = tensor.mul(255).byte()
	img = img.cpu().numpy().transpose((1, 2, 0))
	return img

def calculate(image1, image2):
	# 灰度直方图算法
	# 计算单通道的直方图的相似值
	hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
	hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
	# 计算直方图的重合度
	degree = 0
	for i in range(len(hist1)):
		if hist1[i] != hist2[i]:
			degree = degree + \
				(1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
		else:
			degree = degree + 1
	degree = degree / len(hist1)
	return degree

def classify_hist_with_split(image1, image2):
	sub_image1 = cv2.split(image1)
	sub_image2 = cv2.split(image2)
	sub_data = 0
	for im1, im2 in zip(sub_image1, sub_image2):
		sub_data += calculate(im1, im2)
	sub_data = sub_data / 3
	return sub_data

def main():
	# config
	configs = json.load(open('config.json', 'r'))
	if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

	early_stopping = EarlyStopping(configs['model']['save_dir'], 4)

	# data
	data = DataLoad(configs['data']['dirpath'], configs['data']['train_test_split'])
	train_data_size = len(data.data_train)
	print(train_data_size)
	test_data_size = len(data.data_test)
	print(test_data_size)
	train_dataloader = data.get_train_dataloader()
	test_dataloader = data.get_test_dataloader()

	# model
	deblur = Deblur()
	if torch.cuda.is_available():
		print('GPU')
		deblur = deblur.cuda()

	# loss_function
	loss_fn = torch.nn.SmoothL1Loss()
	# loss_fn = torch.nn.MSELoss()
	if torch.cuda.is_available():
		print('GPU')
		loss_fn = loss_fn.cuda()

	# learning rate and optimizer
	learning_rate = configs['training']['learning_rate']
	optimizer = torch.optim.SGD(deblur.parameters(), lr=learning_rate)

	total_train_step = 0

	total_test_step = 0

	epoch = configs['training']['epochs']

	# add tensorboard
	with SummaryWriter("./logs_train") as writer:
		for i in range(epoch):
			print("------- epoch {} -------".format(i + 1))

			# start training
			deblur.train()
			for data in train_dataloader:
				imgs, targets = data
				if torch.cuda.is_available():
					imgs = imgs.cuda()
					targets = targets.cuda()
				# print(imgs.size())  # [1, 3, 256, 256]
				outputs = deblur(imgs)
				loss = loss_fn(outputs, targets)

				# optimize model
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				total_train_step += 1

				total_train_accuracy = 0
				for n in range(len(outputs)):
					accuracy = float(classify_hist_with_split(tensor_to_np(outputs[n]), tensor_to_np(targets[n])))
					total_train_accuracy += accuracy

				if total_train_step % 10 == 0:
					train_accuracy = total_train_accuracy / len(outputs)
					writer.add_scalar("train_loss", loss.item(), total_train_step)
					writer.add_scalar("train_accuracy", train_accuracy, total_train_step)
					if total_train_step % 100 == 0:
						print("train：{}, Loss: {}, Accuracy: {}".format(total_train_step, loss.item(), train_accuracy))

			# start testing
			deblur.eval()
			total_test_loss = 0
			total_accuracy = 0
			with torch.no_grad():
				for data in test_dataloader:
					imgs, targets = data
					if torch.cuda.is_available():
						imgs = imgs.cuda()
						targets = targets.cuda()
					outputs = deblur(imgs)
					loss = loss_fn(outputs, targets)
					total_test_loss += loss.item()
					for n in range(len(outputs)):
						accuracy = float(classify_hist_with_split(tensor_to_np(outputs[n]), tensor_to_np(targets[n])))
						total_accuracy += accuracy

			print("total test loss: {}".format(total_test_loss))
			print("total test accuracy: {}".format(total_accuracy / test_data_size))
			writer.add_scalar("test_loss", total_test_loss, total_test_step)
			writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
			total_test_step += 1

			# torch.save(deblur, f"{configs['model']['save_dir']}\\deblur_{i}.pth")
			# print("model saved")

			early_stopping(math.floor(total_test_loss * 1000), deblur)
			if early_stopping.early_stop:
				print("Early stopping")
				break


if __name__ == '__main__':
	main()
