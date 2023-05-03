#!/bin/bash

python train_mlm.py \
-b 'dlicari/Italian-Legal-BERT' \
--ntrain 900000 \
--nval 200 \
--ckptrate 50000 \
--valrate 1000 \
--batch 32 \
--chunk 200 \
--lr '2e-5' \
--mlmprob 0.15 \
--seed 0 \
--data '/home/rpozzi/temp_archive/archives/sentenze_pulite_json/' \
--user insides-lab-unimib-wandb \
--project giustizia_mlm \
--name mlm_sentenze
# --wholeword
# --verbose
# --droplast
