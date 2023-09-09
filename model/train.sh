# save model path
rm -rf /embedding/v-xingwuchen/ts_data/TinyStories/tmp_models
mkdir -p /embedding/v-xingwuchen/ts_data/TinyStories/tmp_models
cd $(dirname "$0")

python model.py --vocab-size 50257 --n-positions 2048 --model-folder /embedding/v-xingwuchen/ts_data/TinyStories/tmp_models\
                    --n-embd 64 --n-layer 8 --n-head 16 --model-name TinyStories-1M | tee ../log/TinyStories-1M.log

# python model.py --vocab-size 50257 --n-positions 2048 --model-folder /embedding/v-xingwuchen/ts_data/TinyStories/tmp_models\
#                     --n-embd 128 --n-layer 8 --n-head 16 --model-name TinyStories-3M

# python model.py --vocab-size 50257 --n-positions 2048 --model-folder /embedding/v-xingwuchen/ts_data/TinyStories/tmp_models\
#                     --n-embd 256 --n-layer 8 --n-head 16 --model-name TinyStories-8M | tee ../log/TinyStories-8M.log

# python model.py --vocab-size 50257 --n-positions 2048 --model-folder /embedding/v-xingwuchen/ts_data/TinyStories/tmp_models\
#                     --n-embd 512 --n-layer 8 --n-head 16 --model-name TinyStories-28M \
                    # | tee ../log/TinyStories-28M.log

# python model.py --vocab-size 50257 --n-positions 2048 --model-folder /embedding/v-xingwuchen/ts_data/TinyStories/tmp_models\
#                     --n-embd 768 --n-layer 8 --n-head 16 --model-name TinyStories-33M

# python model.py --vocab-size 50257 --n-positions 2048 --model-folder /embedding/v-xingwuchen/ts_data/TinyStories/tmp_models\
#                     --n-embd 1024 --n-layer 1 --n-head 16 --model-name TinyStories-1Layer-21M

# python model.py --vocab-size 50257 --n-positions 2048 --model-folder /embedding/v-xingwuchen/ts_data/TinyStories/tmp_models\
#                     --n-embd 1024 --n-layer 2 --n-head 16 --model-name TinyStories-2Layers-33M

# du -sh /embedding/v-xingwuchen/ts_data/TinyStories/tmp_models/*

# rm -rf /embedding/v-xingwuchen/ts_data/TinyStories/tmp_models