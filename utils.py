import numpy as np
import torch

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    #print("evaluation_indices",evaluation_indices)
    #print("adaptation_indices", adaptation_indices)

    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model #support loss
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error) #update

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    #query loss
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

# Adapt the model #support loss



def fake_adopt_before(batch, learner, loss, adaptation_steps, shots, ways, device,error_dict,error_data,task):
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
	for step in range(adaptation_steps):
		for (one_class_data, one_class_label) in zip(adaptation_data, adaptation_labels):
			one_class_data = one_class_data.unsqueeze(0)
			one_class_label = one_class_label.unsqueeze(0)
			one_class_loss = loss(learner(one_class_data), one_class_label)
			error_dict[task] = one_class_loss
			train_error += one_class_loss
		# fine-tune
		learner.adapt(train_error)
	
	# Evaluate the adapted model
	error_data[task] = evaluation_data, evaluation_labels
	predictions = learner(evaluation_data)
	# query loss
	
	evaluation_error = loss(predictions, evaluation_labels)
	evaluation_accuracy = accuracy(predictions, evaluation_labels)
	return evaluation_error,evaluation_accuracy,error_dict,error_data


def fake_adopt_now( learner, fake_loss,loss, error_data, task):

	learner.adapt(fake_loss)
	query_data, query_label = error_data[task]
	
	predictions = learner(query_data)
	# query loss
	
	evaluation_error = loss(predictions, query_label)
	
	return evaluation_error


def evaluate(test_iteration,maml,task_information):
	tasksets,meta_batch_size,loss,adaptation_steps,shots,ways,device = task_information
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
		test_error.append(meta_test_error )
		test_accuracy.append(meta_test_accuracy )
	# print('Meta Test Error', meta_test_error / meta_batch_size)
	# print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)
	test_error_mean = np.mean(test_error)
	test_accuracy_mean = np.mean(test_accuracy)
	test_error_std = np.std(test_error)
	test_accuracy_std = np.std(test_accuracy)
	print('Meta Test Error(Iteration Record)', test_error_mean)
	print('Meta Test Accuracy(Iteration Record)', test_accuracy_mean)
	return test_error_mean,test_error_std,test_accuracy_mean,test_accuracy_std