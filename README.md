# Note

1. The paired_data.json file ( https://huggingface.co/datasets/KurisuTL/paired_data ) should be downloaded and saved in the root directory.

2. After training, the program will generate .pt embedding files in the "embeddings" folder


# Training

Here are two parameters to set batch_size and epoch to train LM embedding.

For example, for generating embeddings with batch_size=32, num_epochs=25:

    python emb_train.py --batch_size 32 --num_epochs 25



# Environment(from original project)

Python=3.8.10 

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install pyg -c pyg

conda install -c dglteam/label/cu118 dgl # for RevGAT

pip install transformer

pip install optuna # for hp search

pip install deepspeed # recommend using deepspeed if you want to finetune LM by your self

