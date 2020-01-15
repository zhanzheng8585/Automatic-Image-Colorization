export CUDA_VISIBLE_DEVICES=6

python -u main.py          /home/zhanzheng/places365_standard \
                           --resume pretrained/checkpoint-trained.pth.tar &&
echo "Congratus! Finished *color* training!"

