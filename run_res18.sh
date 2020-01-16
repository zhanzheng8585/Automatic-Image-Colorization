export CUDA_VISIBLE_DEVICES=6,7

python -u main.py           /home/zhanzheng/places365_standard \
							--batch-size 64 \
							-j 16 \
							--lr 0.0001 \
                            --epochs 100 &&
echo "Congratus! Finished *color* training!"

