export CUDA_VISIBLE_DEVICES=6,7

python -u main.py           /home/zhanzheng/places365_standard \
							--batch-size 128 \
							-j 16 \
							--optmzr sgd \
							--lr 0.001 \
							--warmup \
					   		--warmup-epochs 4 \
							--no-tricks \
                            --epochs 100 &&
echo "Congratus! Finished *color* training!"

