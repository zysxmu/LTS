python main.py \
--data /dev/shm/memory_data/ImageNet --visible_gpus '0,1,2,3' \
--multiprocessing_distributed True --dist_url 'tcp://127.0.0.1:23126' \
--workers 16  --arch 'resnet18_quant' --batch_size 256  \
--epochs 100 --lr_m 0.1 --lr_q 0.0001 --log_dir "./results/"  \
--datasetsname 'ImageNet' --lr_scheduler step \
--decay_schedule 30-60-90 \
--bit 2

python main_fixed.py \
--data /dev/shm/memory_data/ImageNet --visible_gpus '0,1,2,3' \
--multiprocessing_distributed True --dist_url 'tcp://127.0.0.1:23108' \
--workers 16  --arch 'resnet18_quant' --batch_size 256  \
--epochs 100 --lr_m 0.1 --lr_q 0.0001 --log_dir "./results/"  \
--datasetsname 'ImageNet' --lr_scheduler step \
--decay_schedule 30-60-90 \
--fixed_mode 'linear-growth' --distance_ema 0.9999 --warmup_epoch 20 --bit 2

python main.py --data /media/disk/ --visible_gpus '1' \
--multiprocessing_distributed False --dist_url 'tcp://127.0.0.1:23116' \
--workers 4  --arch 'resnet20_quant' --batch_size 128  --epochs 400 \
--lr_m 0.1 --lr_q 0.0001 --log_dir "./results/"  --gpu 0 --datasetsname 'cifar10' \
--lr_scheduler step --decay_schedule 100-200-300 \
--fp_path /media/disk/zys/LTS-save/cifar_fp/220316_26_21resnet20_quantcifar10_fp/model_best.pth.tar \
--warmup_epoch 80 --fixed_mode linear-growth --distance_ema 0.99 --bit 2

python main.py --data /media/disk/ --visible_gpus '2' \
--multiprocessing_distributed False --dist_url 'tcp://127.0.0.1:23117' \
--workers 4  --arch 'resnet20_quant' --batch_size 128  --epochs 400 \
--lr_m 0.1 --lr_q 0.0001 --log_dir "./results/"  --gpu 0 --datasetsname 'cifar100' \
--lr_scheduler step --decay_schedule 100-200-300 \
--fp_path /media/disk/zys/LTS-save/cifar_fp/220316_26_21resnet20_quantcifar100_fp/model_best.pth.tar \
--warmup_epoch 80 --fixed_mode linear-growth --distance_ema 0.99 --bit 2


python main.py --data /media/disk/ --visible_gpus '2' \
--multiprocessing_distributed False --dist_url 'tcp://127.0.0.1:23114' \
--workers 4  --arch 'resnet20_quant' --batch_size 128  --epochs 400 \
--lr_m 0.1 --lr_q 0.0001 --log_dir "./results/"  --gpu 0 --datasetsname 'cifar100' \
--lr_scheduler step --decay_schedule 100-200-300 \
--fp_path /media/disk/zys/LTS-save/cifar_fp/220316_26_21resnet20_quantcifar100_fp/model_best.pth.tar --bit 2