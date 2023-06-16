#import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

from time import time

import numpy as np
import torch

import config as c

import model
import monitoring
import train
import test


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run_train():
    torch.multiprocessing.freeze_support()

    monitoring.restart()
    if c.device is "cuda":
        print(torch.cuda.get_device_name(0))        
        c.log(torch.cuda.get_device_name(0), c.logfile)
    running_loss = np.Inf
    try:
        train_losses = []
        val_losses = []
        monitoring.print_config()
        t_start = time()
        for i_epoch in range(-c.pre_low_lr, c.n_epochs):

            if i_epoch < 0:
                for param_group in model.optim_e.param_groups:
                    param_group['lr'] = c.lr_init * 1e-1
                for param_group in model.optim_s.param_groups:
                    param_group['lr'] = c.lr_init * 1e-1
                for param_group in model.optim_n.param_groups:
                    param_group['lr'] = c.lr_init * 1e-1

            epoch_train_losses = train.train_epoch()
            epoch_val_losses  = train.train_epoch(eval=True)

            val_loss_cum = (epoch_val_losses[0]+epoch_val_losses[1]+epoch_val_losses[2])/3
            if  val_loss_cum < running_loss:
                print("Best loss: ", running_loss)
                print("New loss: ", val_loss_cum)
                model.save(c.filename_out)
                running_loss = val_loss_cum
                print("Model saved.")

            lrs = (get_lr(model.optim_e), get_lr(model.optim_s), get_lr(model.optim_n))
            monitoring.show_loss(lrs, np.concatenate([epoch_train_losses, epoch_val_losses]))
            train_losses.append(epoch_train_losses)
            val_losses.append(epoch_val_losses)
            model.scheduler_step(val_loss_cum)

        monitoring.plot_loss(train_losses, val_losses, logscale=True)
        
    except:
        model.save(c.filename_out + '_ABORT')
        raise

    finally:
        print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))        
        c.log("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.), c.logfile)
        model.save(c.filename_out)

def run_test():
    torch.multiprocessing.freeze_support()

    if c.device is "cuda":
        print(torch.cuda.get_device_name(0))        
        c.log(torch.cuda.get_device_name(0), c.logfile)

    try:
        t_start = time()
        test_losses  = test.inference(c.test_path)
        print("Test results: ", test_losses)
        c.log("Test results: %f" % test_losses, c.logfile)

    finally:
        print("\n\nTesting took %f minutes\n\n" % ((time()-t_start)/60.))        
        c.log("\n\nTesting took %f minutes\n\n" % ((time()-t_start)/60.), c.logfile)

# TODO: add evaluation
if __name__ == '__main__':
    run_train()
    run_test()

