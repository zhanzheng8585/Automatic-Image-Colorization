export CUDA_VISIBLE_DEVICES=6,7

python -u main.py           /home/zhanzheng/places365_standard \
							--batch-size 128 \
							-j 16 \
							--optmzr sgd \
							--lr 0.0001 \
							--warmup \
					   		--warmup-epochs 2 \
                            --epochs 50 &&
echo "Congratus! Finished *color* training!"

