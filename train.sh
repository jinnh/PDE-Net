python train.py --cuda --gpu "0" --dataset "CAVE" --upscale_factor 4 --model_name "template" --nEpochs 50

python train.py --cuda --gpu "0" --dataset "CAVE" --upscale_factor 4 --model_name "pde-net" --nEpochs 100 --resume checkpoints/CAVE_x4/template_4_epoch_50.pth