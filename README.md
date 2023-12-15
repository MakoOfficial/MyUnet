# Run
```bash
python pretrain.py \
        --use_canny 0 \
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
        --use_canny 0
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
        --use_canny 1
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
        --use_canny 0 \
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
        --use_canny 1 \
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
    --chkpt checkpoint/shortcutthree/checkpoint_ori_200.pth
    --data_path ../archive/masked_crop/val/a
    --save_path ./output/shortcutthree/masked_ori.png
    --input_size 512
```

# run demo canny
```bash
python demo.py
    --chkpt checkpoint/shortcutthree/checkpoint_canny_200.pth
    --data_path ../archive/masked_crop/canny/val
    --save_path ./output/shortcutthree/masked_canny.png
    --input_size 512
```

# ori2canny
```bash
python demo.py
    --chkpt checkpoint/shortcutthree/checkpoint_ori_200.pth
    --data_path ../archive/masked_crop/canny/val
    --save_path ./output/shortcutthree/ori2canny.png
    --input_size 512
```

# canny2ori
```bash
python demo.py
    --chkpt checkpoint/shortcut/checkpoint_canny_200.pth
    --data_path ../archive/masked_crop/val/a
    --save_path ./output/shortcut/canny2ori.png
    --input_size 512
```