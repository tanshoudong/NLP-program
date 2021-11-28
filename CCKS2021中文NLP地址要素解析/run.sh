rm -rf out
python3 pretrain_bert.py
mkdir cv_tmp

python3 main.py \
--mode='stack' \
--train_batch_size=128 \
--pseudo_filter

rm -rf cv_tmp
mkdir cv_tmp
rm -rf out

python3 main.py \
--mode='stack' \
--train_batch_size=128 \
--attack_train='fgm' \
--pseudo_filter

rm -rf cv_tmp
mkdir cv_tmp
rm -rf out

python3 main.py \
--mode='stack' \
--train_batch_size=256 \
--attack_train='fgm' \
--pseudo_filter

rm -rf cv_tmp
mkdir cv_tmp
rm -rf out

python3 main.py \
--mode='stack' \
--train_batch_size=128 \
--attack_train='fgm' \
--pseudo_filter

rm -rf cv_tmp
mkdir cv_tmp
rm -rf out

python3 main.py \
--mode='stack' \
--train_batch_size=256 \
--train_epochs=8 \
--swa_start=6 \
--cv_infer
