import sys
sys.path.insert(0, '/home/mifsutbenet/fuels')
import os
import numpy as np
import matplotlib.pyplot as plt
from fuels.datasets import init_dataloaders, init_adapt_dataloaders


def assert_case(name,params,i):
        
        params = params.numpy()

        w0s = params[:,0]
        alphas = params[:,1]
        wfs = params[:,2]
        f0s = params[:,3]
      
        if name == 'pendulum-ideal':
            return 'w0 =' + str(w0s[i//2])
        
        elif name == 'pendulum-damped':
             if alphas[i//2] < w0s[i//2]:
                 return 'Underdamped'
             elif alphas[i//2] == w0s[i//2]:
                 return 'Critically damped'
             elif alphas[i//2] > w0s[i//2]:
                 return 'Overdamped'
            
        elif name == 'pendulum-driven':
            if wfs[i//2] == w0s[i//2]:
                return 'Resonance'
            elif wfs[i//2] < w0s[i//2]:
                return 'Sub-resonance'
            elif wfs[i//2] > w0s[i//2]:
                return 'Super-resonance'
            
        else:
            return 'w0 =' + str(w0s[i//2]) + ' alpha =' + str(alphas[i//2]) + ' wf =' + str(wfs[i//2]) + ' f0 =' + str(f0s[i//2])
        


if __name__ == '__main__':  
        path = '/home/mifsutbenet/fuels/fuels/datasets/tests/data'
        # dataset_names = ['pendulum-ideal', 'pendulum-damped', 'pendulum-driven', 'pendulum-damped_driven', 'pendulum-chaotic']
        dataset_names = ['pendulum']



        for name in dataset_names:
                
                dataloader_train, dataloader_test, params = init_dataloaders(dataset_name=name,
                                                                             buffer_filepath= os.path.join(path, name), 
                                                                             batch_size_train=1, 
                                                                             batch_size_val=1)
                
                dataloader_train_ada, dataloader_test_ada, params = init_adapt_dataloaders(dataset_name=name,
                                                                             buffer_filepath= os.path.join(path, name), 
                                                                             batch_size_train=1, 
                                                                             batch_size_val=1)
                
                
                for data in dataloader_train:
                
                    trjs = data['states'].shape[0]

                    fig, ax = plt.subplots(trjs, 1, figsize=(15,8*trjs) )
                    fig.suptitle('Train',fontsize=50)
                    fig.subplots_adjust(top=0.95)


                    for j, tr in enumerate(data['states']):

                            ax[j].plot(tr[0,:].numpy())
                            ax[j].set_title(r"$\omega^2_0 = {:.2f}, $".format(data["w02"][j]) + 
                                            r" $\alpha = {:.2f}, $".format(data["alpha"][j])  +
                                            r" $\omega_f = {:.2f}, $".format(data["wf"][j])   +
                                            r" $f_0 = {:.2f}$".format(data["f0"][j]), fontsize = 25)
                            ax[j].set_xlabel('Time')
                            ax[j].set_ylabel('Position')

                            
                    
                    fig.savefig(os.path.join(path,'Images/'+ name +'_train.png'))




                for data2 in dataloader_train_ada:
                
                    trjs = data2['states'].shape[0]

                    fig, ax = plt.subplots(trjs, 1, figsize=(15,7*trjs) )
                    fig.suptitle('Train adapt',fontsize=50)
                    fig.subplots_adjust(top=0.9)


                    for j, tr in enumerate(data2['states']):

                            ax[j].plot(tr[0,:].numpy())
                            ax[j].set_title(r"$\omega^2_0 = {:.2f}, $".format(data2["w02"][j]) + 
                                            r" $\alpha = {:.2f}, $".format(data2["alpha"][j])  +
                                            r" $\omega_f = {:.2f}, $".format(data2["wf"][j])   +
                                            r" $f_0 = {:.2f}$".format(data2["f0"][j]), fontsize = 25)
                            ax[j].set_xlabel('Time')
                            ax[j].set_ylabel('Position')

                            
                    
                    fig.savefig(os.path.join(path,'Images/'+ name +'_train_ada.png'))