class = 10
image_shape = 28
--class_num 10 

--image_shape 28 28 1 --stages 4 --channel_count 78 --graph_model ws --graph_param 32 4 0.75 --dropout_rate 0.2 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --train_set_size 50000 --val_set_size 10000 --batch_size 100 --epochs 100 --checkpoint_dir ./checkpoint --checkpoint_name randwire_mnist --train_record_dir ./dataset/mnist/train.tfrecord --val_record_dir ./dataset/mnist/test.tfrecord

python train.py 