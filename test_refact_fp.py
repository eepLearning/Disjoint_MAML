import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim
import os
import time

print("pwd :", os.getcwd())
from utils import *
##일단 writer 꺼둔다
import pickle
import gzip
import os
#####args
from collections import defaultdict

from tensorboardX import SummaryWriter  # asdf
import os  # asdf
import datetime  # asdf


def launch_tensor_board(log_path, port, host):  # asdf
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True


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


# print("local 수정 이제 자동으로 안되나?")

def maml_exp(ways=5,
             shots=1,
             meta_lr=0.003,
             fast_lr=0.5,
             meta_batch_size=20,
             adaptation_steps=1,
             num_iterations=30,
             is_disjoint=True,
             GPU_NUM=1,
             seed=42,
             file_name="exp",
             log_dir="./log_dir/default",
             experiment=False,
             scope=1,
             data="omniglot",
             fix_batch_size=True,
             fraction=1,
             fp = 0,
             commend = "commend not found"):
    # set experiment_config
    

    # valid 측정 할거냐 안할거냐, 할거면 몇 번에 한 번 할거냐#이건 일단 빼지 말아보자
    # test 측정 할거냐 안할거냐 , 할거면 몇 번에 한 번 할거냐
    ##experiment :
    # 1.Test record True / False
    # 2.scope
    if experiment == False:
        test_record = False

    else:
        test_record = True

    # set certain gpu id
    exp_name = file_name

    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed(seed)
    print('Current cuda device ', torch.cuda.current_device())
    file_name = data + "_" + "Client=" + str(meta_batch_size) + "_is_disjoint=" + str(is_disjoint) + "_scope=" + str(scope)+"_fraction=" + str(fraction)+"_shots=" + str(shots)+"_fake_episode="+str(fp)+"_"
    log_path = os.path.join('./log/{}'.format(exp_name) )
    time.sleep(random.randrange(1, 6, 1)/10)
    log_path = os.path.join(log_path, file_name + str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) +"_GPU="+str(GPU_NUM))
    writer = SummaryWriter(logdir=log_path)  # asdf

    ######tesnsorboard
    # from torch.utils.tensorboard import SummaryWriter
    # log_dir = "./log_dir/0247"
    # writer = SummaryWriter(log_dir)

    ####csv

    result = defaultdict(list)

    import timeit

    start_time = timeit.default_timer()  # 시작 시간 체크

    ###set seed###
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # in case of mini-imagent :
    # The dataset is divided in 3 splits of
    # 64 training, 16 validation, and 20 testing classes
    # set up disjoint
    if data == "omniglot":
        train_class_origin = 1100
    elif data == "mini-imagenet":
        train_class_origin = 64
    elif data == "tiered-imagenet":
        train_class_origin = 351
    train_class = int(train_class_origin * scope)
    print("Total Train Classes : ", train_class_origin, "===>", train_class, "(scope : ", scope, ")")
    if is_disjoint == True:
        disjoint_setting = []
        client_number = meta_batch_size
        unit = int(train_class / client_number)
        for i in range(client_number):
            tmp = list(range(unit * i, unit * (i + 1)))
            disjoint_setting.append(tmp)
        #### ~exp-v3: client = 4, disjoint_setting = [ [1,2,3] , [4,5,6] , [7,8,9], [10,11,12] ]
        #### from exp-v4: client = 4 but batch_size = 32 so sample 32 clinet in 4 client randomly
        ##exp_v4 / test_refact_v4 revise
        ###Disjoint(NEW)
        if fix_batch_size:
            meta_batch_size_client = meta_batch_size
            meta_batch_size = 32
    else:
        if fix_batch_size:
            meta_batch_size_client = meta_batch_size
            meta_batch_size = 32

        disjoint_setting = []
        meta_batch_size_client = meta_batch_size
        meta_batch_size = 32

    # Load train/validation/test tasksets using the benchmark interface
    tasksets, client = l2l.vision.benchmarks.get_tasksets(data,
                                                          train_ways=ways,  # How many class
                                                          train_samples=2 * shots,  # 왜 2를 곱하지?
                                                          test_ways=ways,
                                                          test_samples=2 * shots,
                                                          num_tasks=-1,
                                                          evaluation_tasks=10000,
                                                          is_disjoint=disjoint_setting,
                                                          root='~/data/{}_task10000'.format(data),
                                                          scope=scope)
    client_original = client

    ###exp_v4

    # Create model
    if data == "omniglot":
        model = l2l.vision.models.OmniglotCNN()
    elif data == "mini-imagenet":
        model = l2l.vision.models.MiniImagenetCNN(ways)
    elif data == "tiered-imagenet":
        model = l2l.vision.models.MiniImagenetCNN(ways)

    # GPU Parallel
    # model = torch.nn.DataParallel(model)
    model.to(device)

    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    best_accuracy = 0
    best_loss = 10
    best_iteration = 0

    # meta_train
    task_information = (tasksets, meta_batch_size, loss, adaptation_steps, shots, ways, device)

    # 아 여기에 오류가 있엇다.(이거때문에 BASE가 저장이 안됨";)
    if len(disjoint_setting) == 0:
        # disjoint_setting = []
        method = "Base"
    else:
        method = "Disjoint"
    client_list = range(len(disjoint_setting))  # len(disjoint_setting)
    client_list = list(client_list)

    for iteration in range(num_iterations+1):

        # nomal fraction and disjoint_setting
        # iteration마다 다르게 sampling되도록 (문제는 , 지금 len(djisjoint) =! 32이고 그냥 client 수 라는거)

        # disjoint setting  #의문: 근데 이 변수가 client랑 동일하게 적용될지는 모르겠다.
        # 근데 어차피 disjoint setting이 client 뽑을려고 만든 거니까 무시해도 괜찮지 않을까?
        if (fix_batch_size) & (method != "Base"):
            random.shuffle(client_list)
            client_list_new = client_list[: int(len(client_list) * fraction)]
            if fraction < 1:
                print("Total Train Client : ", meta_batch_size_client, "===>", len(client_list_new), "(fraction : ",
                      fraction,
                      ")")
                print("Client Index :", client_list_new)
                
            # 에러가 나면 여기서 난다.
            # disjoint_setting_new = [disjoint_setting[random.choice(client_list_new)] for i in range(32)]
            # disjoint_setting = disjoint_setting_new

            client_new = [client_original[random.choice(client_list_new)] for i in range(32)]
            client = client_new
            

    # @save time
        if iteration == 0:
            iteration = iteration + 1
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        error_dict = defaultdict(list)
        error_data = defaultdict(list)
        #여기까지는 iteration 안에 있다.
        split_meta_batch_size = int(meta_batch_size/2)
    
    

        for task in range(meta_batch_size):

            # if task == 1:
            # break
            # Compute meta-training loss
            learner = maml.clone()
            if len(disjoint_setting) == 0:
                batch = tasksets.train.sample()
            else:
                batch = client[task].sample()
                
 
            #normal base
            #print("fp",fp)
            
            if fp == 0:
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   loss,
                                                                   adaptation_steps,
                                                                   shots,
                                                                   ways,
                                                                   device)
                
                evaluation_error.backward()
                
            elif fp == 1:
                evaluation_error, evaluation_accuracy,error_dict,error_data = fake_adopt_1_before(batch,
                                                                                       learner,
                                                                                       loss,
                                                                                       adaptation_steps,
                                                                                       shots,
                                                                                       ways,
                                                                                       device,
                                                                                       error_dict,
                                                                                       error_data,
                                                                                       task)
                evaluation_error.backward()
               
            elif fp == 2:
                evaluation_error, evaluation_accuracy,error_dict,error_data = fake_adopt_debug2(batch,
                                                                                       learner,
                                                                                       loss,
                                                                                       adaptation_steps,
                                                                                       shots,
                                                                                       ways,
                                                                                       device,
                                                                                       error_dict,
                                                                                       error_data,
                                                                                       task)
                evaluation_error.backward()
            elif fp == 3:
               evaluation_error, evaluation_accuracy, error_dict, error_data = fake_adopt_3_before(batch,
                                                                                                 learner,
                                                                                                 loss,
                                                                                                 adaptation_steps,
                                                                                                 shots,
                                                                                                 ways,
                                                                                                 device,
                                                                                                 error_dict,
                                                                                                 error_data,
                                                                                                 task,iteration)
               if iteration % 49 !=0:
                  evaluation_error.backward()

            elif fp == 4:
               evaluation_error, evaluation_accuracy, error_dict, error_data = fake_adopt_4_before(batch,
                                                                                                   learner,
                                                                                                   loss,
                                                                                                   adaptation_steps,
                                                                                                   shots,
                                                                                                   ways,
                                                                                                   device,
                                                                                                   error_dict,
                                                                                                   error_data,
                                                                                                   task, iteration)
               if iteration % 9 != 0:
                  evaluation_error.backward()
            # each iteration, 18 batch go original, 18 batch go fake
            elif fp == 5:
               evaluation_error, evaluation_accuracy, error_dict, error_data = fake_adopt_5_before(batch,
                                                                                                   learner,
                                                                                                   loss,
                                                                                                   adaptation_steps,
                                                                                                   shots,
                                                                                                   ways,
                                                                                                   device,
                                                                                                   error_dict,
                                                                                                   error_data,
                                                                                                   task, iteration,
                                                                                                   split_meta_batch_size)
               if task < split_meta_batch_size:
                  evaluation_error.backward()

            # each iteration, 18 batch go original, 18 batch go fake
            # but not dynamic change
            elif fp == 6:
               evaluation_error, evaluation_accuracy, error_dict, error_data = fake_adopt_6_before(batch,
                                                                                                   learner,
                                                                                                   loss,
                                                                                                   adaptation_steps,
                                                                                                   shots,
                                                                                                   ways,
                                                                                                   device,
                                                                                                   error_dict,
                                                                                                   error_data,
                                                                                                   task, iteration,
                                                                                                   split_meta_batch_size)
               if task < split_meta_batch_size:
                  evaluation_error.backward()
                  
            ###08.10 Professor' lab meeting : => query class =support class
            # fake라 하더라도 구성은 똑같아야 한다.
            elif fp == 7:
               evaluation_error, evaluation_accuracy, error_dict, error_data = fake_adopt_7_before(batch,
                                                                                                   learner,
                                                                                                   loss,
                                                                                                   adaptation_steps,
                                                                                                   shots,
                                                                                                   ways,
                                                                                                   device,
                                                                                                   error_dict,
                                                                                                   error_data,
                                                                                                   task, iteration,
                                                                                                   split_meta_batch_size)
               if task < split_meta_batch_size:
                  evaluation_error.backward()
                  
            ###08.10 Professor' lab meeting : => query class =support class
            # fake라 하더라도 구성은 똑같아야 한다.
            elif fp == 8:
               if task < split_meta_batch_size:
                  evaluation_error, evaluation_accuracy, error_dict, error_data = fake_adopt_8_before(batch,
                                                                                                      learner,
                                                                                                      loss,
                                                                                                      adaptation_steps,
                                                                                                      shots,
                                                                                                      ways,
                                                                                                      device,
                                                                                                      error_dict,
                                                                                                      error_data,
                                                                                                      task, iteration,
                                                                                                      split_meta_batch_size)
                  evaluation_error.backward()
               else:
                  evaluation_error = torch.tensor([0])
                  evaluation_accuracy = torch.tensor([0])


            ###08.11 Professor' lab meeting(공동)
            # [1] 비복원추출 : 같은 class를 다르다고 말하지 않도록
            # [2] support grad / query loss : index가 같도록
            elif fp == 9:
               evaluation_error, evaluation_accuracy, error_dict, error_data = fake_adopt_9_before(batch,
                                                                                                   learner,
                                                                                                   loss,
                                                                                                   adaptation_steps,
                                                                                                   shots,
                                                                                                   ways,
                                                                                                   device,
                                                                                                   error_dict,
                                                                                                   error_data,
                                                                                                   task, iteration,
                                                                                                   split_meta_batch_size)
               if task < split_meta_batch_size:
                  evaluation_error.backward()

            ###08.10 Professor' lab meeting : => query class =support class
            # fake라 하더라도 구성은 똑같아야 한다.
            #디버깅상 오류가 있었네!
            
            elif fp == 10:
               if task < split_meta_batch_size:
                  evaluation_error, evaluation_accuracy, error_dict, error_data = fake_adopt_10_before(batch,
                                                                                                      learner,
                                                                                                      loss,
                                                                                                      adaptation_steps,
                                                                                                      shots,
                                                                                                      ways,
                                                                                                      device,
                                                                                                      error_dict,
                                                                                                      error_data,
                                                                                                      task, iteration,
                                                                                                      split_meta_batch_size)
                  evaluation_error.backward()
               else:
                  evaluation_error = torch.tensor([0])
                  evaluation_accuracy = torch.tensor([0])

            elif fp == 11:
               evaluation_error, evaluation_accuracy, error_dict, error_data = fake_adopt_11_before(batch,
                                                                                                   learner,
                                                                                                   loss,
                                                                                                   adaptation_steps,
                                                                                                   shots,
                                                                                                   ways,
                                                                                                   device,
                                                                                                   error_dict,
                                                                                                   error_data,
                                                                                                   task, iteration,
                                                                                                   split_meta_batch_size)
               if task < split_meta_batch_size:
                  evaluation_error.backward()


            # outer update가 이부분인듯?
            #evaluation_error.backward()
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
        #print("수정 되기는 하는겨?")

        # Print some metrics
        error_train = meta_train_error / meta_batch_size
        accuracy_train = meta_train_accuracy / meta_batch_size
        error_valid = meta_valid_error / meta_batch_size
        accuracy_valid = meta_valid_accuracy / meta_batch_size

        writer.add_scalar('Error_train', error_train, iteration)
        writer.add_scalar('Accuracy_train', accuracy_train, iteration)
        writer.add_scalar('Error_valid', error_valid, iteration)
        writer.add_scalar('Accuracy_valid', accuracy_valid, iteration)

        result["iteration"].append(iteration)
        result["train_loss"].append(error_train)
        result["valid_loss"].append(error_valid)
        result["train_acc"].append(accuracy_train)
        result["valid_acc"].append(accuracy_valid)

        #if iteration % 10 == 0:
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', error_train)
        print('Meta Train Accuracy', accuracy_train)
        print('Meta Valid(before) Error', error_valid)
        print('Meta Valid(before) Accuracy', accuracy_valid)
        

        # Average the accumulated gradients and optimize

        print("fp :",fp)

        if fp in [0,2]:
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()  # 여기서 maml 파라미터 업데이트가 일어난다.
            
        elif fp == 1 :
           all_grad = []
           for g_list in error_dict.values():
              all_grad.extend(g_list)
             
           for task in range(meta_batch_size):
              learner = maml.clone()
              fake_grads = random.choices(all_grad,k=5 )
              
              evaluation_error = fake_adopt_1_now( learner, fake_grads,loss, error_data, task)
              evaluation_error.backward() #이게 제대로 되는 것인지가 좀 애매하네....
   
  
           for p in maml.parameters():
              p.grad.data.mul_(1.0 / meta_batch_size)
           opt.step()
        elif fp == 3:
           if iteration % 49 == 0:
              all_grad = [] #사실 all_updates
              for g_list in error_dict.values():
                 all_grad.extend(g_list)
              for task in range(meta_batch_size):
                 learner = maml.clone()
                 fake_grads = random.choices(all_grad, k=5) # 5 updates
   
                 evaluation_error = fake_adopt_3_now(learner, fake_grads, loss, error_data, task)
                 evaluation_error.backward()  # 이게 제대로 되는 것인지가 좀 애매하네....
   
                 #print("Fake batch - outer loss done")
              for p in maml.parameters():
                 p.grad.data.mul_(1.0 / meta_batch_size)
              opt.step()
           else:
              for p in maml.parameters():
                 p.grad.data.mul_(1.0 / meta_batch_size)
              opt.step()  # 여기서 maml 파라미터 업데이트가 일어난다.


        elif fp == 4:
           if iteration % 9 == 0:
              all_grad = []  # 사실 all_updates
              for g_list in error_dict.values():
                 all_grad.extend(g_list)
              for task in range(meta_batch_size):
                 learner = maml.clone()
                 fake_grads = random.choices(all_grad, k=5)  # 5 updates
         
                 evaluation_error = fake_adopt_4_now(learner, fake_grads, loss, error_data, task)
                 evaluation_error.backward()  # 이게 제대로 되는 것인지가 좀 애매하네....

              # print("Fake batch - outer loss done")
              for p in maml.parameters():
                 p.grad.data.mul_(1.0 / meta_batch_size)
              opt.step()
           else:
              for p in maml.parameters():
                 p.grad.data.mul_(1.0 / meta_batch_size)
              opt.step()  # 여기서 maml 파라미터 업데이트가 일어난다.

        elif fp == 5:
           all_grad = []  # 사실 all_updates
           for g_list in error_dict.values():
              all_grad.extend(g_list)
           for task in range(split_meta_batch_size,meta_batch_size):
              learner = maml.clone()
              fake_grads = random.choices(all_grad, k=5)  # 5 updates
      
              evaluation_error = fake_adopt_5_now(learner, fake_grads, loss, error_data, task)
              evaluation_error.backward()  # 이게 제대로 되는 것인지가 좀 애매하네....
      
              # print("Fake batch - outer loss done")
           for p in maml.parameters():
              p.grad.data.mul_(1.0 / meta_batch_size)
           opt.step()


        elif fp == 6:
           all_grad = []  # 사실 all_updates
           for g_list in error_dict.values():
              all_grad.extend(g_list)
           for task in range(split_meta_batch_size,meta_batch_size):
              learner = maml.clone()
              original_grads = error_dict[task]
              fake_grads = random.choices(all_grad, k=2)  # only 2 updates
              original_grads[3:] = fake_grads
      
              evaluation_error = fake_adopt_6_now(learner, original_grads, loss, error_data, task)
              evaluation_error.backward()  # 이게 제대로 되는 것인지가 좀 애매하네....
      
              # print("Fake batch - outer loss done")
           for p in maml.parameters():
              p.grad.data.mul_(1.0 / meta_batch_size)
           opt.step()


        elif fp == 7:
           all_grad = []  # 사실 all_updates
           for g_list in error_dict.values():
              all_grad.extend(g_list)
           #제대로 되었는지 확인
           assert len(all_grad) == ways * split_meta_batch_size
           for task in range(split_meta_batch_size, meta_batch_size):
              learner = maml.clone()
              #original_grads = error_dict[task]
              
              fake_grads_index = random.choices(range(len(all_grad)), k=5)  # only 2 updates
              fake_grads = [all_grad[idx] for idx in fake_grads_index]
              label_index = [f % 5 for f in fake_grads_index]  # label index
              client_index = [(f // 5)+split_meta_batch_size for f in fake_grads_index]  # client index
      
              evaluation_error = fake_adopt_7_now(learner,fake_grads,  loss, error_data, task,label_index,client_index )
              evaluation_error.backward()  # 이게 제대로 되는 것인지가 좀 애매하네....
 
         
      
              # print("Fake batch - outer loss done")
           for p in maml.parameters():
              p.grad.data.mul_(1.0 / meta_batch_size)
           opt.step()

        elif fp == 8:
           all_grad = []  # 사실 all_updates
           for g_list in error_dict.values():
              all_grad.extend(g_list)
           # 제대로 되었는지 확인
           assert len(all_grad) == ways * split_meta_batch_size
           for task in range(split_meta_batch_size, meta_batch_size):
              learner = maml.clone()
              # original_grads = error_dict[task]
      
              fake_grads_index = random.choices(range(len(all_grad)), k=5)  # only 2 updates
              fake_grads = [all_grad[idx] for idx in fake_grads_index]
              label_index = [f % 5 for f in fake_grads_index]  # label index
              client_index = [(f // 5)  for f in fake_grads_index]  # client index
      
              evaluation_error = fake_adopt_8_now(learner, fake_grads, loss, error_data, task, label_index,
                                                  client_index)
              evaluation_error.backward()  # 이게 제대로 되는 것인지가 좀 애매하네....
      
              # print("Fake batch - outer loss done")
           for p in maml.parameters():
              p.grad.data.mul_(1.0 / meta_batch_size)
           opt.step()

#############08.11.
        # 리스트에 담는 방식으로 , 없으면 추가하도록
        # 추가하는 순서는 0,1,2,3,4 => 이건 고정시켜도 될듯

        elif fp == 9:
           all_grad = []  # 사실 all_updates
           for g_list in error_dict.values():
              all_grad.extend(g_list)
           # 제대로 되었는지 확인
           assert len(all_grad) == ways * split_meta_batch_size
           for task in range(split_meta_batch_size, meta_batch_size):
              learner = maml.clone()
              # original_grads = error_dict[task]
              fake_grads =[]
              dis_grads =[]
              fake_grads_index = []
              for meta_label in range(ways):
                 while len(fake_grads) != (meta_label+1):
                    index = random.choice([ (i*ways + meta_label) for i in range(split_meta_batch_size) ])
                    #index = 1
                    fake_grad = all_grad[index]
                    head_grad = fake_grad[-1]
                    if any([(head_grad == dis_).all() for dis_ in dis_grads]):
                       continue
                    else:
                       dis_grads.append(head_grad)
                       fake_grads.append(fake_grad)
                       fake_grads_index.append(index)
                       
              assert len(fake_grads) == ways
              assert len(fake_grads_index) == ways
      
              label_index = [f % 5 for f in fake_grads_index]  # label index
              assert len(set(label_index)) == ways
              client_index = [(f // 5) + split_meta_batch_size for f in fake_grads_index]  # client index
      
              evaluation_error = fake_adopt_9_now(learner, fake_grads, loss, error_data, task, label_index,
                                                  client_index)
              evaluation_error.backward()  # 이게 제대로 되는 것인지가 좀 애매하네....
      
              # print("Fake batch - outer loss done")
           for p in maml.parameters():
              p.grad.data.mul_(1.0 / meta_batch_size)
           opt.step()

        elif fp == 10:
           all_grad = []  # 사실 all_updates
           for g_list in error_dict.values():
              all_grad.extend(g_list)
           # 제대로 되었는지 확인
           assert len(all_grad) == ways * split_meta_batch_size
           for task in range(split_meta_batch_size, meta_batch_size):
              learner = maml.clone()
              # original_grads = error_dict[task]
              fake_grads = []
              dis_grads = []
              fake_grads_index = []
              for meta_label in range(ways):
                 while len(fake_grads) != (meta_label + 1):
                    index = random.choice([(i * ways + meta_label) for i in range(split_meta_batch_size)])
                    # index = 1
                    fake_grad = all_grad[index]
                    head_grad = fake_grad[-1]
                    if any([(head_grad == dis_).all() for dis_ in dis_grads]):
                       continue
                    else:
                       dis_grads.append(head_grad)
                       fake_grads.append(fake_grad)
                       fake_grads_index.append(index)

              assert len(fake_grads) == ways
              assert len(fake_grads_index) == ways

              label_index = [f % 5 for f in fake_grads_index]  # label index
              assert len(set(label_index)) == ways #검증 0,1,2,3,4가 하나씩만 들어가야 되니까
              client_index = [(f // 5) for f in fake_grads_index]  # client index
      
              evaluation_error = fake_adopt_10_now(learner, fake_grads, loss, error_data, task, label_index,
                                                  client_index)
              evaluation_error.backward()  # 이게 제대로 되는 것인지가 좀 애매하네....
      
              # print("Fake batch - outer loss done")
           for p in maml.parameters():
              p.grad.data.mul_(1.0 / meta_batch_size)
           opt.step()

           #############08.11.
           # 리스트에 담는 방식으로 , 없으면 추가하도록
           # 추가하는 순서는 0,1,2,3,4 => 이건 고정시켜도 될듯

        elif fp == 11:
           all_grad = []  # 사실 all_updates
           for g_list in error_dict.values():
              all_grad.extend(g_list)
           # 제대로 되었는지 확인
           assert len(all_grad) == ways * split_meta_batch_size
           for task in range(split_meta_batch_size, meta_batch_size):
              task_order = task - split_meta_batch_size
              learner = maml.clone()
              # original_grads = error_dict[task]
              fake_grads = all_grad[task_order*5:task_order*5+5]

              fake_grads_index = list(range(task_order*5,task_order*5+5))
   
   
              assert len(fake_grads) == ways
              assert len(fake_grads_index) == ways
   
              label_index = [f % 5 for f in fake_grads_index]  # label index
              assert len(set(label_index)) == ways
              client_index = [(f // 5) + split_meta_batch_size for f in fake_grads_index]  # client index
   
              evaluation_error = fake_adopt_11_now(learner, fake_grads, loss, error_data, task, label_index,
                                                  client_index)
              evaluation_error.backward()  # 이게 제대로 되는 것인지가 좀 애매하네....
   
              # print("Fake batch - outer loss done")
           for p in maml.parameters():
              p.grad.data.mul_(1.0 / meta_batch_size)
           opt.step()
        '''
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for fp debug
        for i in range(32):
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
        error_valid = meta_valid_error / meta_batch_size
        accuracy_valid = meta_valid_accuracy / meta_batch_size
        print('Meta Valid(after) Error', error_valid)
        print('Meta Valid(after) Accuracy', accuracy_valid)
        '''
        
    
        
        ##for monitoring
        if iteration % 1000 == 0:
            valid_error_mean, valid_error_std, valid_accuracy_mean, valid_accuracy_std = evaluate(1000,
                                                                                                  maml,
                                                                                                  task_information)
            #valid acc말고 valid loss로 간다.
            writer.add_scalar('valid_accuracy_mean_1000',valid_accuracy_mean, iteration)
            writer.add_scalar('valid_loss_mean_1000', valid_error_mean, iteration)
            
            #valid_accuracy_best = valid_accuracy_mean
            
            #if (valid_accuracy_best > best_accuracy) and (valid_accuracy_best < 1):
            if (valid_error_mean < best_loss):
                #best_accuracy = valid_accuracy_best
                best_loss = valid_error_mean
                best_learner = maml.clone()
                best_iteration = iteration
                
        if test_record == True:
            if iteration % 1000 == 0:
                test_error_mean, test_error_std, test_accuracy_mean, test_accuracy_std = evaluate(5000,
                                                                                                  maml,
                                                                                                  task_information)
                writer.add_scalar('test_accuracy_mean_5000', test_accuracy_mean, iteration)
                writer.add_scalar('test_accuracy_std_5000', test_accuracy_std, iteration)
                result["test(record)"].append(
                    {'mean': test_accuracy_mean, 'std': test_accuracy_std, 'iteration': iteration,
                     'client': meta_batch_size, 'method': method})

    # writer.add_scalar('Test',{'Acc(Mean)':test_accuracy_mean, 'Acc(Mean)(std)':test_accuracy_std  } , 0)
    # final_acc

    # @save time
    # 모든 iteration이 끝난 후
    test_error_mean, test_error_std, test_accuracy_mean, test_accuracy_std = evaluate(5000,
                                                                                      maml, task_information)
    result["test(mean)"].append({'mean': test_accuracy_mean, 'std': test_accuracy_std, 'iteration': iteration,
                                 'client': meta_batch_size, 'method': method})
  
    print('\n')
    print(method, " : Client ", meta_batch_size)
    print('Meta Test Error(Mean)', test_error_mean)
    print('Meta Test Accuracy(Mean)', test_accuracy_mean)

    # for best model at valid dataset

    # @save time
    # for there is no best model yet
    try:
        best_test_error_mean, best_test_error_std, best_test_accuracy_mean, best_test_accuracy_std = evaluate(5000,
                                                                                          best_learner,
                                                                                          task_information)
    except:
        best_test_error_mean, best_test_error_std, best_test_accuracy_mean, best_test_accuracy_std= evaluate(5000,
                                                                                          maml, task_information)
    print("Best Iteration on Validation : Iteration", best_iteration)
    print('Meta Test Error(Best)', best_test_error_mean)
    print('Meta Test Accuracy(Best)', best_test_accuracy_mean)
    # writer.add_scalars('Test',{'Acc(Best)':test_accuracy_mean, 'Acc(Best)(std)':test_accuracy_std  } , 0)
    
    result["test(best)"].append({'mean': best_test_accuracy_mean, 'std': best_test_accuracy_std, 'iteration': iteration,
                                 'client': meta_batch_size, 'method': method, 'best_iteration': best_iteration,
                                 "best_accuracy": best_accuracy})
    result["commend"] = commend
    writer.add_hparams(
        {'Method': method, 'shots': shots, 'ways': ways, 'meta_lr': meta_lr, 'Client': len(disjoint_setting), 'fast_lr': fast_lr,
         'meta_batch_size': meta_batch_size,
         'num_iterations': num_iterations, 'is_disjoint': is_disjoint, 'scope': scope, 'data': data,
         'fraction': fraction, 'best_iteraetion':best_iteration, 'commend':commend},
        {'hparam/accuracy_mean': test_accuracy_mean, 'hparam/accuracy_mstd': test_accuracy_std,
         'hparam/accuracy_mean(best_learner)': best_test_accuracy_mean,
         'hparam/accuracy_mstd(best_learner)': best_test_accuracy_std})
    
    # writer.close()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    terminate_time = timeit.default_timer()  # 종료 시간 체크
    result["time"] = (terminate_time - start_time)
    with gzip.open(log_dir + '/{}_{}_client{}_scope{}_fraction{}_fp{}_{}_shot{}.pickle'.format(exp_name, method, meta_batch_size_client,
                                                                           str(scope),str(fraction), fp,data,shots), 'wb') as f:
        pickle.dump(result, f)

    terminate_time = timeit.default_timer()  # 종료 시간 체크

    print("%f초 걸렸습니다." % (terminate_time - start_time))
    writer.flush()
    writer.close()
# 일단 고정은 됬다.
# Meta Test Error 3.1889143958687782
# Meta Test Accuracy 0.26875000447034836
