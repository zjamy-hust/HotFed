mode=1

python -u fedxai.py --epochs 20 --num_users 16 \
                    --local_ep 50 --gpu 1 --iid 1 --mode $mode \
                    --dataset 'cifar10' --num_classes 10 \
                    > ./res/mode_${mode}.log 2>&1