python main_fixed.py --data /media/disk/ --visible_gpus '1' --multiprocessing_distributed False --dist_url 'tcp://127.0.0.1:23118' --workers 4  --arch 'resnet20_quant' --batch_size 128  --epochs 400 --lr_m 0.1 --lr_q 0.0001 --log_dir "./results/Ablation_linear_em"  --gpu 0 --datasetsname 'cifar100' --lr_scheduler step --decay_schedule 100-200-300 --fp_path 220316_26_21resnet20_quantcifar100_fp/model_best.pth.tar --warmup_epoch 0 --fixed_mode linear-growth --distance_ema 0.99 --bit 2

python main_fixed.py --data /media/disk/ --visible_gpus '1' --multiprocessing_distributed False --dist_url 'tcp://127.0.0.1:23118' --workers 4  --arch 'resnet20_quant' --batch_size 128  --epochs 400 --lr_m 0.1 --lr_q 0.0001 --log_dir "./results/Ablation_linear_em"  --gpu 0 --datasetsname 'cifar100' --lr_scheduler step --decay_schedule 100-200-300 --fp_path 220316_26_21resnet20_quantcifar100_fp/model_best.pth.tar --warmup_epoch 40 --fixed_mode linear-growth --distance_ema 0.99 --bit 2

python main_fixed.py --data /media/disk/ --visible_gpus '1' --multiprocessing_distributed False --dist_url 'tcp://127.0.0.1:23118' --workers 4  --arch 'resnet20_quant' --batch_size 128  --epochs 400 --lr_m 0.1 --lr_q 0.0001 --log_dir "./results/Ablation_linear_em"  --gpu 0 --datasetsname 'cifar100' --lr_scheduler step --decay_schedule 100-200-300 --fp_path 220316_26_21resnet20_quantcifar100_fp/model_best.pth.tar --warmup_epoch 120 --fixed_mode linear-growth --distance_ema 0.99 --bit 2


python main_fixed.py --data /media/disk/ --visible_gpus '1' --multiprocessing_distributed False --dist_url 'tcp://127.0.0.1:23118' --workers 4  --arch 'resnet20_quant' --batch_size 128  --epochs 400 --lr_m 0.1 --lr_q 0.0001 --log_dir "./results/Ablation_linear_em"  --gpu 0 --datasetsname 'cifar100' --lr_scheduler step --decay_schedule 100-200-300 --fp_path 220316_26_21resnet20_quantcifar100_fp/model_best.pth.tar --warmup_epoch 200 --fixed_mode linear-growth --distance_ema 0.99 --bit 2


python main_fixed.py --data /media/disk/ --visible_gpus '1' --multiprocessing_distributed False --dist_url 'tcp://127.0.0.1:23118' --workers 4  --arch 'resnet20_quant' --batch_size 128  --epochs 400 --lr_m 0.1 --lr_q 0.0001 --log_dir "./results/Ablation_linear_em"  --gpu 0 --datasetsname 'cifar100' --lr_scheduler step --decay_schedule 100-200-300 --fp_path 220316_26_21resnet20_quantcifar100_fp/model_best.pth.tar --warmup_epoch 280 --fixed_mode linear-growth --distance_ema 0.99 --bit 2


python main_fixed.py --data /media/disk/ --visible_gpus '1' --multiprocessing_distributed False --dist_url 'tcp://127.0.0.1:23118' --workers 4  --arch 'resnet20_quant' --batch_size 128  --epochs 400 --lr_m 0.1 --lr_q 0.0001 --log_dir "./results/Ablation_linear_em"  --gpu 0 --datasetsname 'cifar100' --lr_scheduler step --decay_schedule 100-200-300 --fp_path 220316_26_21resnet20_quantcifar100_fp/model_best.pth.tar --warmup_epoch 360 --fixed_mode linear-growth --distance_ema 0.99 --bit 2

