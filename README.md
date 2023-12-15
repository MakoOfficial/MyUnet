# Run
```bash
python pretrain.py \
        --use_canny False \
        --data_path ../maske/train/ \
        --save_pth  ./checkpoint/checkpoint_ori_200.pth \
        --batch_size 20 \
        --epochs 200 \
        --save_ckpt_freq 200 \
        --input_size 224 \
        --lr 1e-4 \
        --step_size 50 \
        --gamma 0.7
```

# local run ori
```bash
python pretrain.py
        --use_canny False
        --data_path ../archive/masked_crop/train/1
        --save_path  ./checkpoint/checkpoint_ori_200.pth
        --batch_size 8
        --epochs 200
        --save_ckpt_freq 200
        --input_size 512
        --lr 1e-4
        --step_size 50
        --gamma 0.7
```

# local run canny
```bash
python pretrain.py
        --use_canny True
        --data_path ../archive/masked_crop/canny/train
        --save_path  ./checkpoint/checkpoint_canny_200.pth
        --batch_size 8
        --epochs 200
        --save_ckpt_freq 200
        --input_size 512
        --lr 1e-4
        --step_size 50
        --gamma 0.7
```

# server run ori
```bash
python pretrain.py \
        --use_canny False \
        --data_path ../masked/train/1 \
        --save_path  ./checkpoint/checkpoint_ori_200.pth \
        --batch_size 8 \
        --epochs 200 \
        --save_ckpt_freq 200 \
        --input_size 512 \
        --lr 1e-4 \
        --step_size 50 \
        --gamma 0.7
```

# server run canny
```bash
python pretrain.py \
        --use_canny True \
        --data_path ../canny \
        --save_path  ./checkpoint/checkpoint_canny_200.pth \
        --batch_size 8 \
        --epochs 200 \
        --save_ckpt_freq 200 \
        --input_size 512 \
        --lr 1e-4 \
        --step_size 50 \
        --gamma 0.7
```

# if input_size = 512, on 12G memory, the batch is no more than 8 

# run demo ori
```bash
python demo.py
    --chkpt checkpoint\checkpoint_ori_200.pth
    --data_path ../archive/masked_crop/val/a
    --save_path ./output/masked_ori.png
    --input_size 512
```

# run demo canny
```bash
python demo.py
    --chkpt checkpoint\checkpoint_canny_200.pth
    --data_path ../archive/masked_crop/canny/val
    --save_path ./output/masked_canny.png
    --input_size 512
```

# play
```bash
python demo.py
    --chkpt checkpoint\checkpoint_ori_200.pth
    --data_path ../archive/masked_crop/canny/val
    --save_path ./output/ori2canny.png
    --input_size 512
```