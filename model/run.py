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

def run():
    torch.multiprocessing.freeze_support()

    monitoring.restart()

    running_loss = np.Inf
    try:
        monitoring.print_config()
        t_start = time()
        for i_epoch in range(-c.pre_low_lr, c.n_epochs):

            if i_epoch < 0:
                for param_group in model.optim.param_groups:
                    param_group['lr'] = c.lr_init * 1e-1

            train_losses = train.train_epoch(i_epoch)
            test_losses  = train.train_epoch(i_epoch, test=True)

            if test_losses[0] < running_loss:
                model.save(c.filename_out)
                running_loss = test_losses[0]

            monitoring.show_loss(np.concatenate([train_losses, test_losses]))
            model.scheduler_step()

    except:
        model.save(c.filename_out + '_ABORT')
        raise

    finally:
        print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))        
        c.fileout.write("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))
        model.save(c.filename_out)
        c.fileout.close()


# TODO: add evaluation

if __name__ == '__main__':
    run()

