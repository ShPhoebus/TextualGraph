# TextualGraph

For generating embeddings with batch_size=16, num_epochs=15:

  python emb_train_16.py

with batch_size=64, num_epochs=15:

  python emb_train_64.py

with batch_size=256, num_epochs=15:

  python emb_train_256.py


# Environment

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c dglteam/label/cu118 dgl # for RevGAT
pip install transformer
pip install optuna # for hp search
pip install deepspeed # recommend using deepspeed if you want to finetune LM by your self
