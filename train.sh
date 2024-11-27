MODEL_DIR="gs://sultan-t5x/kenlm_30GB_M3Vocab_base"
python3 t5x/train.py   --gin_search_paths=clinc_t5   --gin_file="pretrain.gin"   --gin.MODEL_DIR=\"${MODEL_DIR}\"
