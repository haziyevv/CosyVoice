#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=5
stop_stage=5

data_url=www.openslr.org/resources/60
data_dir=/workspace/CosyVoice-mine-2/cosyvoice-dataset/samples
pretrained_model_dir=/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B

train_folders=(
  Autumn
  Honey
  Phoenix
  Swiss
  Achernar
  Aoede
  Autonoe
  Despina
  Erinome
  Kore
  Leda
  Pulcherrima
  Sulafat
  Vindemiatrix
  Zephyr
)

test_folder_name=(
  Autumn-test
  Honey-test
  Phoenix-test
  Swiss-test
  Achernar-test
  Aoede-test
  Autonoe-test
  Despina-test
  Erinome-test
  Kore-test
  Leda-test
  Pulcherrima-test
  Sulafat-test
  Vindemiatrix-test
  Zephyr-test
)

data_folder_name=data

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in "${test_folder_name[@]}"; do
    mkdir -p $data_folder_name/$x
    python local/prepare_data.py --src_dir $data_dir/$x --des_dir $data_folder_name/$x
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for x in "${test_folder_name[@]}"; do
    tools/extract_embedding.py --dir $data_folder_name/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in "${test_folder_name[@]}"; do
    tools/extract_speech_token.py --dir $data_folder_name/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v3.onnx
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in "${test_folder_name[@]}"; do
    mkdir -p $data_folder_name/$x/parquet
    echo $x
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir $data_folder_name/$x \
      --des_dir $data_folder_name/$x/parquet
  done
fi


# train llm
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Run train. We only support llm traning for now"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cat $data_folder_name/{Achernar,Aoede,Autonoe,Despina,Erinome,Kore,Leda,Pulcherrima,Sulafat,Vindemiatrix,Zephyr}/parquet/data.list > $data_folder_name/train.data.list
  cat $data_folder_name/{Achernar-test,Aoede-test,Autonoe-test,Despina-test,Erinome-test,Kore-test,Leda-test,Pulcherrima-test,Sulafat-test,Vindemiatrix-test,Zephyr-test}/parquet/data.list > $data_folder_name/dev.data.list
  # NOTE will update llm/hift training later
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice3.yaml \
      --train_data $data_folder_name/train.data.list \
      --cv_data $data_folder_name/dev.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice3/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice3/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer \
      --freeze_tokens
  done
fi

# average model
average_num=2
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  for model in llm; do
    decode_checkpoint=`pwd`/exp/cosyvoice3/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice3/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi
