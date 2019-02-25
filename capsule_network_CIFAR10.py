"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""
import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from residual_caps_net import CapsuleNet
from capsule import softmax, augmentation, CapsuleLoss

BATCH_SIZE = 4
NUM_CLASSES = 10
NUM_EPOCHS = 400
NUM_ROUTING_ITERATIONS = 3
RECONSTRUCTION_REGULARIZER = 0.0005


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    #from torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomSaver
    #from torchvision.utils import make_grid
    from torchvision.datasets import CIFAR10
    from tqdm import tqdm
    import torchnet as tnt

    model = CapsuleNet(NUM_CLASSES = NUM_CLASSES, NUM_ITERATIONS = NUM_ROUTING_ITERATIONS)
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    model.cuda()
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    
    

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    #train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    #train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    #test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    #test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    #confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix','columnnames': list(range(NUM_CLASSES)),'rownames': list(range(NUM_CLASSES))})
                                                     
    #ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
    #reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    capsule_loss = CapsuleLoss(RECONSTRUCTION_REGULARIZER = RECONSTRUCTION_REGULARIZER)


    def get_iterator(mode):
        dataset = CIFAR10(root='./data', download=True, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)


    def processor(sample):
        data, labels, training = sample
        data = data.permute(0,3,1,2)
        
        data = augmentation(data.float() / 255.0)
        labels = torch.LongTensor(labels)

        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes


    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()


    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].item())


    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])


    def on_end_epoch(state):
        #print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))
        global train_loss
        global train_accuracy
        global test_loss
        global test_accuracy
        
        train_loss.append(meter_loss.value()[0])
        train_accuracy.append(meter_accuracy.value()[0])

        #train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        #train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

        reset_meters()

        engine.test(processor, get_iterator(False))
        #test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        #test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        #confusion_logger.log(confusion_meter.value())

        #print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))
        
        test_loss.append(meter_loss.value()[0])
        test_accuracy.append(meter_accuracy.value()[0])

        if state['epoch']%5 == 0 or state['epoch'] == NUM_EPOCHS:
            torch.save(model.state_dict(), 'epochs/epoch_%d_Feb_13_recon_0_1.pt' % state['epoch'])

        # Reconstruction visualization.

        test_sample = next(iter(get_iterator(False)))

        ground_truth = (test_sample[0].float() / 255.0)
        ground_truth = ground_truth.permute(0,3,1,2)
        _, reconstructions = model(Variable(ground_truth).cuda())
        reconstruction = reconstructions.cpu().view_as(ground_truth).data
        


        #ground_truth_logger.log(make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
        #reconstruction_logger.log(make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
        

        result_dict = {
                           'test_loss': np.asarray(test_loss), 
                           'test_accuracy': np.asarray(test_accuracy),
                           'train_loss': np.asarray(train_loss),
                           'train_accuracy': np.asarray(train_accuracy),
                           'confusion_matrix': np.asarray(confusion_meter.value()),
                           'num_epochs' : NUM_EPOCHS, 'batch_size':BATCH_SIZE
                      }
        
        np.save("experiments\\result_dict_base_caps_E400_Feb13_Recon_0_1.npy", result_dict)
        
        
        if state['epoch'] == NUM_EPOCHS:
            torch.save(ground_truth, "experiments\\ground_truth_Feb13_Recon_0_1.pt")
            torch.save(reconstruction, "experiments\\reconstruction_Feb13_Recon_0_1.pt")


    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
