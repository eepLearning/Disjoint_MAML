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

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def evaluate(test_iteration,maml,task_information):
	tasksets,meta_batch_size,loss,adaptation_steps,shots,ways,device = task_information
	test_error = []
	test_accuracy = []
	
	for i in range(test_iteration):
		
		meta_test_error = 0.0
		meta_test_accuracy = 0.0
		
		for task in range(meta_batch_size):
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
		test_error.append(meta_test_error / meta_batch_size)
		test_accuracy.append(meta_test_accuracy / meta_batch_size)
	# print('Meta Test Error', meta_test_error / meta_batch_size)
	# print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)
	test_error_mean = np.mean(test_error)
	test_accuracy_mean = np.mean(test_accuracy)
	test_error_std = np.std(test_error)
	test_accuracy_std = np.std(test_accuracy)
	print('Meta Test Error(Mean)', test_error_mean)
	print('Meta Test Accuracy(Mean)', test_accuracy_mean)
	return test_error_mean,test_error_std,test_accuracy_mean,test_accuracy_std