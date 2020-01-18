export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u main.py           /home/zhanzheng/places365_standard \
							--batch-size 256 \
							-j 16 \
							--optmzr sgd \
							--lr 0.0001 \
							--warmup \
					   		--warmup-epochs 2 \
                            --epochs 50 &&
echo "Congratus! Finished *color* training!"

