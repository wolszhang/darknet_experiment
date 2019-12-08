Class=10
Channel=78
Dropout=0.2
Learning_rate=0.1
Momentum=0.9
Weight_decay=0.0001
Train_sz=50000
Val_sz=10000
Batch_size=100
Epochs=100
Checkpoint_dir="./checkpoint"	
Checkpoint_name="randwire_cifar10"
Train_dir="./dataset/cifar10/train.tfrecord"
Tal_dir="./dataset/cifar10/test.tfrecord"
# Train with ER
python.exe train.py --class_num $Class --image_shape 32 32 3 --channel_count $Channel --graph_model ws --graph_param 32 4 0.75 --dropout_rate $Dropout --learning_rate $Learning_rate --momentum $Momentum --weight_decay ${Weight_decay} --train_set_size $Train_sz --val_set_size $Val_sz --batch_size $Batch_size --epochs $Epochs --checkpoint_dir ${Checkpoint_dir} --checkpoint_name ${Checkpoint_name} --train_record_dir $Train_dir --val_record_dir $Val_dir 

