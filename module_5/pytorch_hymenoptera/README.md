
python train.py -–datapath ../input/hymenoptera_data/hymenoptera_data/ -–save_to ../checkpoints  -–model resnet18 –-pretrained True --aug False

python eval.py –-datapath ../input/hymenoptera_data/val/ --model resnet18 --pretrained True --aug False