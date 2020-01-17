export CUDA_VISIBLE_DEVICES=6,7

python -u main.py           /home/zhanzheng/places365_standard \
							--resume /home/zhanzheng/Automatic-Image-Colorization/checkpoints/model_best.pth.tar \
							--batch-size 128 \
							-j 16 \
							--optmzr adam \
							--lr 0.001 \
                            --epochs 100 &&
echo "Congratus! Finished *color* training!"

