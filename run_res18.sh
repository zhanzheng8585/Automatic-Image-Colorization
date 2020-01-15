export CUDA_VISIBLE_DEVICES=6

python -u main.py          /home/zhanzheng/places365_standard \
                           --pretrained pretrained/resnet_gray_weights.pth.tar &&
echo "Congratus! Finished *c8pattern* admm training!"

