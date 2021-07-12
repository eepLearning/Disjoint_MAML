import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim
import os
print("pwd :",os.getcwd())
from utils import evaluate
##일단 writer 꺼둔다
import pickle
import gzip
import os
#####args
from collections import defaultdict


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
   
   # Adapt the model
   for step in range(adaptation_steps):
      train_error = loss(learner(adaptation_data), adaptation_labels)
      learner.adapt(train_error)
   
   # Evaluate the adapted model
   predictions = learner(evaluation_data)
   valid_error = loss(predictions, evaluation_labels)
   valid_accuracy = accuracy(predictions, evaluation_labels)
   return valid_error, valid_accuracy

#print("local 수정 이제 자동으로 안되나?")

def maml_exp(ways=5,
             shots=1,
             meta_lr=0.003,
             fast_lr=0.5,
             meta_batch_size=20,
             adaptation_steps=1,
             num_iterations=30,
             is_disjoint = True,
             GPU_NUM = 1,
             seed=42,
             file_name = "exp",
             log_dir ="./log_dir/default",
             experiment = False,
             scope = 1):


   #set experiment_config
   
   #valid 측정 할거냐 안할거냐, 할거면 몇 번에 한 번 할거냐#이건 일단 빼지 말아보자
   #test 측정 할거냐 안할거냐 , 할거면 몇 번에 한 번 할거냐
   ##experiment :
   #1.Test record True / False
   #2.scope
   if experiment == False:
      test_record = False
   
   else:
      test_record = True
   
   
   #set certain gpu id
  
   device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
   torch.cuda.manual_seed(seed)
   print ('Current cuda device ', torch.cuda.current_device())
   
   
   
   ######tesnsorboard
   #from torch.utils.tensorboard import SummaryWriter
   #log_dir = "./log_dir/0247"
   #writer = SummaryWriter(log_dir)
   
   ####csv

   result = defaultdict(list)
   
   import timeit
   
   start_time = timeit.default_timer()  # 시작 시간 체크
   
   
   ###set seed###
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
 
   
   #set up disjoint
   train_class_origin = 1100
   train_class =int(train_class_origin*scope)
   print("Total Train Classes : ",train_class_origin,"===>",train_class,"(scope : ",scope,")")
   if is_disjoint == True:
      disjoint_setting = []
      client_number = meta_batch_size
      unit = int(train_class / client_number)
      for i in range(client_number):
         tmp  =  list(range(unit*i,unit*(i+1)))
         disjoint_setting.append(tmp)
   else:
      disjoint_setting = []
   
   
   # Load train/validation/test tasksets using the benchmark interface
   tasksets,client = l2l.vision.benchmarks.get_tasksets('omniglot',
                                               train_ways=ways, #How many class
                                               train_samples=2*shots, #왜 2를 곱하지?
                                               test_ways=ways,
                                               test_samples=2*shots,
                                               num_tasks=-1,
                                               evaluation_tasks=10000,
                                               is_disjoint = disjoint_setting,
                                               root='~/data/task10000',
                                               scope = scope
                                               
   )
   
   # Create model
   model = l2l.vision.models.OmniglotCNN()
   
   #GPU Parallel
   #model = torch.nn.DataParallel(model)
   model.to(device)
   
   
   
   maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
   para = maml.parameters()
   
   opt = optim.Adam(maml.parameters(), meta_lr)
   loss = nn.CrossEntropyLoss(reduction='mean')
   
   best_accuracy = 0
   
   #meta_train
   task_information = (tasksets, meta_batch_size, loss, adaptation_steps, shots, ways, device)
   
   #아 여기에 오류가 있엇다.(이거때문에 BASE가 저장이 안됨";)
   if len(disjoint_setting) == 0:
      #disjoint_setting = []
      method = "Base"
   else:
      method = "Disjoint"

   for iteration in range(num_iterations):
     opt.zero_grad()
     meta_train_error = 0.0
     meta_train_accuracy = 0.0
     meta_valid_error = 0.0
     meta_valid_accuracy = 0.0
     for task in range(meta_batch_size):
        
         #if task == 1:
            #break
         # Compute meta-training loss
         learner = maml.clone()
         if len(disjoint_setting) == 0:
            batch = tasksets.train.sample()
         else:
            batch = client[task].sample()
            #print(len(batch[0]))
         
         #print(len(batch[0]))
   
         #print(len(batch[1]))
         #print(batch[1])
         
        
         evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                             learner,
                                                             loss,
                                                             adaptation_steps,
                                                             shots,
                                                             ways,
                                                             device)
   
         #outer update가 이부분인듯?
         evaluation_error.backward()
         meta_train_error += evaluation_error.item()
         meta_train_accuracy += evaluation_accuracy.item()
   
         # Compute meta-validation loss
         learner = maml.clone()
         batch = tasksets.validation.sample()
         evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                             learner,
                                                             loss,
                                                             adaptation_steps,
                                                             shots,
                                                             ways,
                                                             device)
         meta_valid_error += evaluation_error.item()
         meta_valid_accuracy += evaluation_accuracy.item()
    
     # Print some metrics
     error_train = meta_train_error / meta_batch_size
     accuracy_train = meta_train_accuracy / meta_batch_size
     error_valid = meta_valid_error / meta_batch_size
     accuracy_valid = meta_valid_accuracy / meta_batch_size
   
     #writer.add_scalar('Loss/Train', error_train, iteration)
     #writer.add_scalar('Acc/Train', accuracy_train, iteration)
     #writer.add_scalar('Loss/Valid', error_valid, iteration)
     #writer.add_scalar('Acc/Valid', accuracy_valid, iteration)
     
     result["iteration"].append(iteration)
     result["train_loss"].append(error_train)
     result["valid_loss"].append(error_valid)
     result["train_acc"].append(accuracy_train)
     result["valid_acc"].append(accuracy_valid)
   
     if iteration % 10 == 0:
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', error_train)
        print('Meta Train Accuracy', accuracy_train)
        print('Meta Valid Error', error_valid)
        print('Meta Valid Accuracy', accuracy_valid)
     if iteration % 100 ==0:
        valid_accuracy_best = (meta_valid_accuracy / meta_batch_size)
        if (valid_accuracy_best  > best_accuracy) and (valid_accuracy_best <1):
           best_accuracy = valid_accuracy_best
           best_learner = maml.clone()
           best_iteration = iteration
     if test_record == True:
        if iteration % 1000 == 0:
           test_error_mean, test_error_std, test_accuracy_mean, test_accuracy_std = evaluate(5000, maml,task_information)
           result["test(mean)"].append({'mean': test_accuracy_mean, 'std': test_accuracy_std, 'iteration': iteration,
                                        'client': meta_batch_size, 'method': method})
 
           
   
     # Average the accumulated gradients and optimize
     for p in maml.parameters():
         p.grad.data.mul_(1.0 / meta_batch_size)
     opt.step() #여기서 maml 파라미터 업데이트가 일어난다.

   
   #writer.add_scalar('Test',{'Acc(Mean)':test_accuracy_mean, 'Acc(Mean)(std)':test_accuracy_std  } , 0)
   #final_acc
   test_error_mean, test_error_std, test_accuracy_mean, test_accuracy_std = evaluate(5000,maml,task_information)
   
   
   result["test(mean)"].append({'mean':test_accuracy_mean,'std':test_accuracy_std,'iteration':iteration,
                                'client':meta_batch_size,'method':method })
   
   print('\n')
   print(method," : Client ",meta_batch_size)
   print('Meta Test Error(Mean)', test_error_mean)
   print('Meta Test Accuracy(Mean)', test_accuracy_mean)
   
   #for best model at valid dataset

   test_error_mean, test_error_std, test_accuracy_mean, test_accuracy_std = evaluate(5000, best_learner, task_information)
   print("Best Iteration on Validation : Iteration",best_iteration)
   print('Meta Test Error(Best)', test_error_mean)
   print('Meta Test Accuracy(Best)', test_accuracy_mean)
   #writer.add_scalars('Test',{'Acc(Best)':test_accuracy_mean, 'Acc(Best)(std)':test_accuracy_std  } , 0)
   result["test(best)"].append({'mean':test_accuracy_mean,'std':test_accuracy_std,'iteration':iteration,
                                'client':meta_batch_size,'method':method,'best_iteration':best_iteration,"best_accuracy":best_accuracy })
   #writer.close()

   if not os.path.exists(log_dir):
      os.makedirs(log_dir)

   terminate_time = timeit.default_timer() # 종료 시간 체크
   result["time"] = (terminate_time - start_time)
   with gzip.open(log_dir+'/{}_{}_{}_{}.pickle'.format(file_name,method,meta_batch_size,str(scope)), 'wb') as f:
      pickle.dump(result, f)
   
   terminate_time = timeit.default_timer()  # 종료 시간 체크
   
   print("%f초 걸렸습니다." % (terminate_time - start_time))


#일단 고정은 됬다.
#Meta Test Error 3.1889143958687782
#Meta Test Accuracy 0.26875000447034836