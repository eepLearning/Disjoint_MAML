import numpy as np
import torch
from torch.autograd import grad
from learn2learn.utils import clone_module, update_module
from torch import nn, optim


def maml_update(model, lr, grads=None):
	"""
	[[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)

	**Description**

	Performs a MAML update on model using grads and lr.
	The function re-routes the Python object, thus avoiding in-place
	operations.

	NOTE: The model itself is updated in-place (no deepcopy), but the
			parameters' tensors are not.

	**Arguments**

	* **model** (Module) - The model to update.
	* **lr** (float) - The learning rate used to update the model.
	* **grads** (list, *optional*, default=None) - A list of gradients for each parameter
		 of the model. If None, will use the gradients in .grad attributes.

	**Example**
	~~~python
	maml = l2l.algorithms.MAML(Model(), lr=0.1)
	model = maml.clone() # The next two lines essentially implement model.adapt(loss)
	grads = autograd.grad(loss, model.parameters(), create_graph=True)
	maml_update(model, lr=0.1, grads)
	~~~
	"""
	if grads is not None:
		params = list(model.parameters())
		if not len(grads) == len(list(params)):
			msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
			msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
			print(msg)
		for p, g in zip(params, grads):
			if g is not None:
				p.update = - lr * g
	return update_module(model)


def accuracy(predictions, targets):
	predictions = predictions.argmax(dim=1).view(targets.shape)
	return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
	data, labels = batch
	data, labels = data.to(device), labels.to(device)
	
	# Separate data into adaptation/evalutation sets
	adaptation_indices = np.zeros(data.size(0), dtype=bool)
	adaptation_indices[np.arange(shots * ways) * 2] = True
	evaluation_indices = torch.from_numpy(~adaptation_indices)
	adaptation_indices = torch.from_numpy(adaptation_indices)
	# print("evaluation_indices",evaluation_indices)
	# print("adaptation_indices", adaptation_indices)
	
	adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
	evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
	
	# Adapt the model #support loss
	for step in range(adaptation_steps):
		train_error = loss(learner(adaptation_data), adaptation_labels)
		learner.adapt(train_error)  # update
	
	# Evaluate the adapted model
	predictions = learner(evaluation_data)
	# query loss
	valid_error = loss(predictions, evaluation_labels)
	valid_accuracy = accuracy(predictions, evaluation_labels)
	return valid_error, valid_accuracy


# Adapt the model #support loss


def fake_adopt_debug2(batch,
							 learner,
							 loss,
							 adaptation_steps,
							 shots,
							 ways,
							 device,
							 error_dict,
							 error_data,
							 task):
	data, labels = batch
	data, labels = data.to(device), labels.to(device)
	
	# Separate data into adaptation/evalutation sets
	adaptation_indices = np.zeros(data.size(0), dtype=bool)
	adaptation_indices[np.arange(shots * ways) * 2] = True
	evaluation_indices = torch.from_numpy(~adaptation_indices)
	adaptation_indices = torch.from_numpy(adaptation_indices)
	# print("evaluation_indices",evaluation_indices)
	# print("adaptation_indices", adaptation_indices)
	
	adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
	evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
	loss2 = nn.CrossEntropyLoss(reduction='none')

	# Adapt the model #support loss
	for step in range(adaptation_steps):
		train_error = loss2(learner(adaptation_data), adaptation_labels)
		# learner.adapt(train_error) #update
	mean_seperate_error = torch.mean(train_error)
	grads = grad(mean_seperate_error, learner.parameters(), create_graph=True)
	updates = [-learner.lr * g for g in grads]
	update_module(learner, updates=updates)
	# Evaluate the adapted model
	predictions = learner(evaluation_data)
	# query loss
	valid_error = loss(predictions, evaluation_labels)
	valid_accuracy = accuracy(predictions, evaluation_labels)
	return valid_error, valid_accuracy,{"2":[3]},{"2":[3]}


def fake_adopt_3_before(batch,
							 learner,
							 loss,
							 adaptation_steps,
							 shots,
							 ways,
							 device,
							 error_dict,
							 error_data,
							 task,iteration):
	data, labels = batch
	data, labels = data.to(device), labels.to(device)
	
	adaptation_indices = np.zeros(data.size(0), dtype=bool)
	adaptation_indices[np.arange(shots * ways) * 2] = True
	evaluation_indices = torch.from_numpy(~adaptation_indices)
	adaptation_indices = torch.from_numpy(adaptation_indices)

	
	adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
	evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
	
	# Adapt the model #support loss
	if iteration % 49 == 0:
		loss2 = nn.CrossEntropyLoss(reduction='none')
		for step in range(adaptation_steps):
			individual_loss = loss2(learner(adaptation_data), adaptation_labels)
			for il in individual_loss:
				grads = grad(il, learner.parameters(),retain_graph= True) #이거 안하면 끝나고 free되서 오류남
				updates = [-learner.lr * g for g in grads]
				error_dict[task].append(updates)
			error_data[task] = evaluation_data, evaluation_labels
			#train_error = torch.mean(individual_loss)
			#learner.adapt(train_error)
		valid_error = torch.tensor([0])
		valid_accuracy = torch.tensor([0])
	
	else:
		for step in range(adaptation_steps):
			train_error = loss(learner(adaptation_data), adaptation_labels)
			learner.adapt(train_error)
		predictions = learner(evaluation_data)
		valid_error = loss(predictions, evaluation_labels)
		valid_accuracy = accuracy(predictions, evaluation_labels)


	return valid_error, valid_accuracy, error_dict, error_data


def fake_adopt_3_now(learner, fake_grads, loss, error_data, task):
	for updates in fake_grads:
		update_module(learner, updates=updates)
	query_data, query_label = error_data[task]
	
	predictions = learner(query_data)
	# query loss
	
	evaluation_error = loss(predictions, query_label)
	
	return evaluation_error

def fake_adopt_before(batch, learner, loss, adaptation_steps, shots, ways, device, error_dict, error_data, task):
	datas, labels = batch
	datas, labels = datas.to(device), labels.to(device)
	
	# Separate data into adaptation/evalutation sets
	adaptation_indices = np.zeros(datas.size(0), dtype=bool)
	adaptation_indices[np.arange(shots * ways) * 2] = True
	evaluation_indices = torch.from_numpy(~adaptation_indices)
	adaptation_indices = torch.from_numpy(adaptation_indices)
	# print("evaluation_indices",evaluation_indices)
	# print("adaptation_indices", adaptation_indices)
	
	adaptation_data, adaptation_labels = datas[adaptation_indices], labels[adaptation_indices]
	evaluation_data, evaluation_labels = datas[evaluation_indices], labels[evaluation_indices]
	
	# Adapt the model
	train_error = 0
	print("adaptation_labels)", adaptation_labels)
	for step in range(adaptation_steps):
		for (one_class_data, one_class_label) in zip(adaptation_data, adaptation_labels):
			print("one_class_label: ", one_class_label)
			one_class_data = one_class_data.unsqueeze(0)
			one_class_label = one_class_label.unsqueeze(0)
			print("one_class_label:(unsquzee) ", one_class_label)
			one_class_loss = loss(learner(one_class_data), one_class_label)
			grads = grad(one_class_loss / 5, learner.parameters(), allow_unused=False)
			error_dict[task].append(grads)
			train_error += one_class_loss
		# print("one class label loss :",one_class_loss)
		
		# print("mean train error :",train_error/5)
		original_error = loss(learner(adaptation_data), adaptation_labels)
		# print("original train error : ",original_error)
		# print("@@@@@@@@@@@@@@@@@@@debug loss")
		# fine-tune
		# learner.adapt(train_error)
		for g in error_dict[task]:
			learner = maml_update(learner, learner.lr, g)
	
	# Evaluate the adapted model
	error_data[task] = evaluation_data, evaluation_labels
	predictions = learner(evaluation_data)
	# query loss
	
	evaluation_error = loss(predictions, evaluation_labels)
	evaluation_accuracy = accuracy(predictions, evaluation_labels)
	return evaluation_error, evaluation_accuracy, error_dict, error_data


def fake_adopt_now(learner, fake_grads, loss, error_data, task):
	for g in fake_grads:
		learner = maml_update(learner, learner.lr, g)
	query_data, query_label = error_data[task]
	
	predictions = learner(query_data)
	# query loss
	
	evaluation_error = loss(predictions, query_label)
	
	return evaluation_error


def fake_adopt_debug(batch, learner, loss, adaptation_steps, shots, ways, device, error_dict, error_data, task):
	datas, labels = batch
	datas, labels = datas.to(device), labels.to(device)
	
	# Separate data into adaptation/evalutation sets
	adaptation_indices = np.zeros(datas.size(0), dtype=bool)
	adaptation_indices[np.arange(shots * ways) * 2] = True
	evaluation_indices = torch.from_numpy(~adaptation_indices)
	adaptation_indices = torch.from_numpy(adaptation_indices)
	# print("evaluation_indices",evaluation_indices)
	# print("adaptation_indices", adaptation_indices)
	
	adaptation_data, adaptation_labels = datas[adaptation_indices], labels[adaptation_indices]
	evaluation_data, evaluation_labels = datas[evaluation_indices], labels[evaluation_indices]
	
	# Adapt the model
	train_error = []
	# print("adaptation_labels)", adaptation_labels)
	for step in range(adaptation_steps):
		for (one_class_data, one_class_label) in zip(adaptation_data, adaptation_labels):
			# print("one_class_label: ", one_class_label)
			
			# print("one_class_label:(unsquzee) ", one_class_label)
			# 주석처리
			one_class_data = one_class_data.unsqueeze(0)
			one_class_label = one_class_label.unsqueeze(0)
			one_class_loss = loss(learner(one_class_data), one_class_label)
			grads = grad(one_class_loss / 5, learner.parameters(), create_graph=True)
			updates = [-learner.lr * g for g in grads]
			error_dict[task].append(updates)
			train_error.append(one_class_loss)
	# print("one class label loss :",one_class_loss)
	
	# print("mean train error :",train_error/5)
	
	# original_error = loss(learner(adaptation_data), adaptation_labels)
	# print("original train error : ",original_error)
	# print("@@@@@@@@@@@@@@@@@@@debug loss")
	# fine-tune
	# learner.adapt(train_error)
	# 1차 시도
	# for g in error_dict[task]:
	# learner = maml_update(learner, learner.lr, g)
	# 2차 시도
	# for u in error_dict[task]:
	# update_module(learner,updates = u)
	# 3차 시도
	# grads = grad(train_error, learner.parameters(),  create_graph=True)
	# updates = [-learner.lr * g for g in grads]
	# update_module(learner, updates=updates)
	# 4차 시도
	# grads = grad(original_error, learner.parameters(),  create_graph=True)
	# updates = [-learner.lr * g for g in grads]
	# update_module(learner, updates=updates)
	# 5차 시도
	# mean_error = torch.mean(torch.stack(train_error))
	# grads = grad(mean_error, learner.parameters(),  create_graph=True)
	# updates = [-learner.lr * g for g in grads]
	# update_module(learner, updates=updates)
	# 6차 시도
	# mean_error = torch.mean(torch.stack(train_error))
	# grads = grad(mean_error, learner.parameters(),  create_graph=True)
	# updates = [-learner.lr * g for g in grads]
	# update_module(learner, updates=updates)
	
	# Evaluate the adapted model
	error_data[task] = evaluation_data, evaluation_labels
	predictions = learner(evaluation_data)
	# query loss
	
	evaluation_error = loss(predictions, evaluation_labels)
	evaluation_accuracy = accuracy(predictions, evaluation_labels)
	return evaluation_error, evaluation_accuracy, error_dict, error_data


def evaluate(test_iteration, maml, task_information):
	tasksets, meta_batch_size, loss, adaptation_steps, shots, ways, device = task_information
	test_error = []
	test_accuracy = []
	
	for i in range(test_iteration):
		meta_test_error = 0.0
		meta_test_accuracy = 0.0
		
		# Compute meta-testing loss
		learner = maml.clone()
		batch = tasksets.test.sample()
		# print("batch",len(batch))
		evaluation_error, evaluation_accuracy = fast_adapt(batch,
																			learner,
																			loss,
																			adaptation_steps,
																			shots,
																			ways,
																			device)
		meta_test_error += evaluation_error.item()
		meta_test_accuracy += evaluation_accuracy.item()
		test_error.append(meta_test_error)
		test_accuracy.append(meta_test_accuracy)
	# print('Meta Test Error', meta_test_error / meta_batch_size)
	# print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)
	test_error_mean = np.mean(test_error)
	test_accuracy_mean = np.mean(test_accuracy)
	test_error_std = np.std(test_error)
	test_accuracy_std = np.std(test_accuracy)
	print('Meta Test Error(Iteration Record)', test_error_mean)
	print('Meta Test Accuracy(Iteration Record)', test_accuracy_mean)
	return test_error_mean, test_error_std, test_accuracy_mean, test_accuracy_std


#############fake adopt 4
def fake_adopt_4_before(batch,
								learner,
								loss,
								adaptation_steps,
								shots,
								ways,
								device,
								error_dict,
								error_data,
								task, iteration):
	data, labels = batch
	data, labels = data.to(device), labels.to(device)
	
	adaptation_indices = np.zeros(data.size(0), dtype=bool)
	adaptation_indices[np.arange(shots * ways) * 2] = True
	evaluation_indices = torch.from_numpy(~adaptation_indices)
	adaptation_indices = torch.from_numpy(adaptation_indices)
	
	adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
	evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
	
	# Adapt the model #support loss
	if iteration % 9 == 0:
		loss2 = nn.CrossEntropyLoss(reduction='none')
		for step in range(adaptation_steps):
			individual_loss = loss2(learner(adaptation_data), adaptation_labels)
			for il in individual_loss:
				#grads = grad(il, learner.parameters(), retain_graph=True)  # 이거 안하면 끝나고 free되서 오류남
				#updates = [-learner.lr * g for g in grads]
				error_dict[task].append(il)
			error_data[task] = evaluation_data, evaluation_labels
		# train_error = torch.mean(individual_loss)
		# learner.adapt(train_error)
		valid_error = torch.tensor([0])
		valid_accuracy = torch.tensor([0])
	
	else:
		for step in range(adaptation_steps):
			train_error = loss(learner(adaptation_data), adaptation_labels)
			learner.adapt(train_error)
		predictions = learner(evaluation_data)
		valid_error = loss(predictions, evaluation_labels)
		valid_accuracy = accuracy(predictions, evaluation_labels)
	
	return valid_error, valid_accuracy, error_dict, error_data


def fake_adopt_4_now(learner, fake_grads, loss, error_data, task):
	#for  in fake_grads:
		#update_module(learner, updates=updates)
	print(fake_grads)
	train_error = torch.mean( torch.stack(fake_grads) )
	#train_error = torch.cat(fake_grads, 0)
	print(train_error)
	#learner.adapt(train_error)
	grads = grad(train_error, learner.parameters())
	updates = [-learner.lr * g for g in grads]
	update_module(learner, updates=updates)
	
	query_data, query_label = error_data[task]
	
	predictions = learner(query_data)
	# query loss
	
	evaluation_error = loss(predictions, query_label)
	
	return evaluation_error