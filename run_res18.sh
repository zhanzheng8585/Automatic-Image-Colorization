export CUDA_VISIBLE_DEVICES=1,2,3,4

python -u main.py           /home/zhanzheng/places365_standard \
							--resume /home/zhanzheng/Automatic-Image-Colorization/checkpoints/model_best.pth.tar \
							--batch-size 256 \
							-j 16 \
							--optmzr adam \
							--lr 0.00001 \
                            --epochs 50 &&
echo "Congratus! Finished *color* training!"

