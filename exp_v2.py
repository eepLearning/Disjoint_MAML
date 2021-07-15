import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim
import os
from test_refact import maml_exp

import sys
import argparse



def get_arguments():
	def str2bool(vv):
		l = []
		for v in vv:
			if isinstance(v, bool):
				l.append(v)
			if v.lower() in ('yes', 'true', 't', 'y', '1'):
				l.append(True)
			elif v.lower() in ('no', 'false', 'f', 'n', '0'):
				l.append(False)
			else:
				raise argparse.ArgumentTypeError('Boolean value expected.')
		return l
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--client_grid', help='default) client_grid = [5,10,20,50,100]',nargs ="+", default = 2 , dest='client_grid',type = int)
	parser.add_argument('--num_iterations', help='default) 10', default=10, dest='num_iterations',type = int)
	parser.add_argument('--is_disjoint_option', help='default) [True,False]', nargs ="*",default= False , dest='is_disjoint_option',type= str)
	parser.add_argument('--file_name', help='default) "EXP_0" ', default="EXP_0", dest='file_name',type = str)
	parser.add_argument('--experiment', help='default) False if True: Test Performance will be recorded', default=False, dest='experiment', type=str)
	parser.add_argument('--scope', help='default) 1 , to minimize total train class to half then set scope to 0.5 ', default=1, dest='scope', type=float)
	parser.add_argument('--gpu_number', help='default) 1', default=1, dest='GPU_NUM',type = int)
	parser.add_argument('--case_control', help='default) False',default= False , dest='case_control',type = str)

	
	
	#그냥 type bool로 하면 파서가 못 읽는 거 같다.

	client_grid = parser.parse_args().client_grid
	num_iterations = parser.parse_args().num_iterations
	is_disjoint_option= parser.parse_args().is_disjoint_option
	is_disjoint_option = str2bool(is_disjoint_option)
	#is_disjoint_option = [True , False] #아직 파서로 받는 거 해결못함. #해결됨
	file_name= parser.parse_args().file_name
	experiment= parser.parse_args().experiment
	scope = parser.parse_args().scope
	#print(experiment)
	#experiment =str2bool(experimnet)[0]
	GPU_NUM= parser.parse_args().GPU_NUM
	#add case_control
	case_control = dict()
	case_control["control"] = bool(parser.parse_args().case_control)
	case_control["case"] = [10,20,30]
	return client_grid, num_iterations, is_disjoint_option, file_name, experiment, scope, GPU_NUM,case_control


def main(client_grid,num_iterations,is_disjoint_option,file_name,experiment,scope, GPU_NUM,**case_control  ):
	print("client_grid :",client_grid)
	print("num_iterations :",num_iterations)
	print("is_disjoint_option :",is_disjoint_option)
	print("file_name :",file_name)
	print("experiment :",experiment)
	print("scope :",scope)
	print("GPU_NUM :",GPU_NUM)
	print("case_control :",case_control["control"])
	if case_control["control"] == True:
		print("1 Disjoint pool have : ",case_control["case"])
		total_train_class =1100
		control_break_condition = 5
		total_train_class = int(total_train_class * scope)
		if total_train_class/max(case_control["case"]) < control_break_condition:
			raise ('Invalid Numbers of Classes Assign to Pool')
		client_grid = [int(total_train_class  / classes ) for classes in case_control["case"]]
		print("(Revised) client_grid :", client_grid)
		
		
	import timeit
	idx = 0
	st = timeit.default_timer()  # 시작 시간 체크
	log_dir = "./log_dir/" + file_name
	for is_disjoint in is_disjoint_option:
		for client in client_grid:
			idx = idx + 1
			print(">>>>>>>>>EXP :",idx,"Proceed ")
			maml_exp(ways=5,
						shots=1,
						meta_lr=0.003,
						fast_lr=0.5,
						meta_batch_size=client,
						adaptation_steps=1,
						num_iterations=num_iterations,
						is_disjoint=is_disjoint,
						GPU_NUM=GPU_NUM,
						seed=42,
						file_name=file_name,
						log_dir=log_dir,
						experiment=experiment,
						scope = scope)
	
	te = timeit.default_timer()  # 종료 시간 체크
	print("%f초 걸렸습니다." % (te - st))
	

	

if __name__ == '__main__':
	client_grid, num_iterations, is_disjoint_option, file_name, experiment,scope, GPU_NUM, case_control = get_arguments()
	
	main(client_grid, num_iterations, is_disjoint_option, file_name, experiment,scope, GPU_NUM,**case_control)
