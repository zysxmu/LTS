## frozen
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch \
                                          --master_port 12347 \
                                          --nproc_per_node=4 train_ImageNet.py \
                                          --arch 'mobilenetv1_w1_quant' \
                                          --epochs 100 \
                                          --bit_weight 4 \
                                          --bit_act 4 \
                                          --optimizer 'SGD' \
                                          --weight_decay 5e-4 \
                                          --lr 1e-2 \
                                          --log_dir './results/mobilenetv1_w1_quant-W4A4-e100' \
                                          --datapath './DATASET/ImageNet/' \
                                          --fixed True \
                                          --fixed_rate 0.1 \
                                          --fixed_mode 'linear-growth' \
                                          --warmup_epoch 16 \
                                          --distance_ema 0.9999 \
                                          --scheduler step \
                                          --batch_size 256
## unfrozen
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
                                          --master_port 12327 \
                                          --nproc_per_node=4 train_ImageNet.py \
                                          --arch 'resnet18_quant' \
                                          --epochs 100 \
                                          --bit_weight 4 \
                                          --bit_act 4 \
                                          --optimizer 'SGD' \
                                          --weight_decay 5e-4 \
                                          --lr 1e-2 \
                                          --log_dir './results/resnet18_quant-W4A4-baseline' \
                                          --datapath './DATASET/ImageNet/' \
                                          --fixed False \
                                          --scheduler step \
                                          --batch_size 256


