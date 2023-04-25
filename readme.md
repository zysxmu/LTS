## Code for Exploiting the Partly Scratch-off Lottery Ticket for Quantization-Aware Training

### Training for CIFAR10/100

#### fp model

At first, train a fp model. For example, train a fp model for CIFAR100, please use:

`python main_fixed.py 
--data pathToCifar
--visible_gpus '0' 
--multiprocessing_distributed False
--dist_url 'tcp://127.0.0.1:23117'
--workers 4 --arch 'resnet20_quant'
--batch_size 128  
--epochs 400 
--lr_m 0.1 
--lr_q 0.0001 
--log_dir "./results/"
--gpu 0 
--datasetsname 'cifar100/10' 
--lr_scheduler step 
--decay_schedule 100-200-300`

#### baseline quantized model

For example, if you want to train a 2-bit r20, use:

`python main.py 
--data pathToCifar
--visible_gpus '0' 
--multiprocessing_distributed False 
--dist_url 'tcp://127.0.0.1:23117' 
--workers 4  
--arch 'resnet20_quant' 
--batch_size 128  
--epochs 400 
--lr_m 0.1 
--lr_q 0.0001 
--log_dir "./results/"  
--gpu 0 
--datasetsname'cifar100' 
--lr_scheduler step 
--decay_schedule 100-200-300 
--fp_path pathTofpModel
--bit 2`

#### LTS quantized model

For example, if you want to train a 2-bit r20, use:

`python main_fixed.py 
--data pathToCifar
--visible_gpus '0' 
--multiprocessing_distributed False 
--dist_url 'tcp://127.0.0.1:23117' 
--workers 4  
--arch 'resnet20_quant' 
--batch_size 128  
--epochs 400 
--lr_m 0.1 
--lr_q 0.0001 
--log_dir "./results/"  
--gpu 0 
--datasetsname'cifar100' 
--lr_scheduler step 
--decay_schedule 100-200-300 
--fp_path pathTofpModel 
--warmup_epoch 80 
--fixed_mode linear-growth  
--distance_ema 0.99 
--bit 2`

### Training for ImageNet

#### baseline quantized model

For example, if you want to train a 2-bit r18, use:

`python main.py 
--data pathToImagenet
--visible_gpus '0,1,2,3' 
--multiprocessing_distributed True 
--dist_url 'tcp://127.0.0.1:23117' 
--workers 16  
--arch 'resnet18_quant' 
--batch_size 256  
--epochs 100 
--lr_m 0.1 
--lr_q 0.0001 
--log_dir "./results/"  
--datasetsname'ImageNet' 
--lr_scheduler step 
--decay_schedule 30-60-90 
--bit 2`

#### LTS quantized model

For example, if you want to train a 2-bit r18, use:


`python main.py 
--data pathToImagenet
--visible_gpus '0,1,2,3' 
--multiprocessing_distributed True 
--dist_url 'tcp://127.0.0.1:23117' 
--workers 16  
--arch 'resnet18_quant' 
--batch_size 256  
--epochs 100 
--lr_m 0.1 
--lr_q 0.0001 
--log_dir "./results/"  
--datasetsname'ImageNet' 
--lr_scheduler step 
--decay_schedule 30-60-90 
--warmup_epoch 80 
--fixed_mode linear-growth 
--distance_ema 0.99 
--bit 2`

### Trained model
Coming soon......

## Acknowledgments
Code is implemented based on [PalQuant](https://github.com/huqinghao/PalQuant/)
