#!/usr/bin/env bash

#rsync -avr  --progress /data/dataset/data_inpaint dgl@server2:~/
##rsync -avr  --progress /data/dataset/data_inpaint dgl@server3:~/
#rsync -avr  --progress /data/dataset/data_inpaint dgl@server4:~/
#
rsync -avr  --progress ~/iter_denoise dgl@server2:~/
rsync -avr  --progress ~/iter_denoise dgl@server3:~/
rsync -avr  --progress ~/iter_denoise dgl@server4:~/

#rsync -avr  --progress ~/iter_denoise/model.py dgl@server2:~/iter_denoise
#rsync -avr  --progress ~/iter_denoise/model.py dgl@server3:~/iter_denoise
#rsync -avr  --progress ~/iter_denoise/model.py dgl@server4:~/iter_denoise
#
#rsync -avr  --progress ~/iter_denoise/mypackage/ dgl@server2:~/iter_denoise
#rsync -avr  --progress ~/iter_denoise/mypackage dgl@server3:~/iter_denoise
#rsync -avr  --progress ~/iter_denoise/mypackage dgl@server4:~/iter_denoise
#
#
#rsync -avr  --progress ~/iter_denoise/dist_main.py dgl@server2:~/iter_denoise/
#rsync -avr  --progress ~/iter_denoise/dist_main.py dgl@server3:~/iter_denoise/
#rsync -avr  --progress ~/iter_denoise/dist_main.py dgl@server4:~/iter_denoise/
#rsync -avr  --progress ~/iter_denoise/run.sh dgl@server2:~/iter_denoise/
#rsync -avr  --progress ~/iter_denoise/run.sh dgl@server3:~/iter_denoise/
#rsync -avr  --progress ~/iter_denoise/run.sh dgl@server4:~/iter_denoise/
