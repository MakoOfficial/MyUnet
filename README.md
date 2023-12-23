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
        --data_path ../archive/masked_1K_train/ori
        --save_path  ./checkpoint/masked_1K_ori_200.pth
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
        --data_path ../archive/masked_1K_train/canny
        --save_path  ./checkpoint/masked_1K_canny_200.pth
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
        --data_path ../masked_1K_train/ori \
        --save_path  ./checkpoint/masked_1K_ori_200.pth \
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
        --data_path ../masked_1K_train/canny \
        --save_path  ./checkpoint/masked_1K_canny_200.pth \
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
    --chkpt ./checkpoint/masked_1K/masked_1K_ori_200.pth
    --data_path ../archive/masked_crop/val/a
    --save_path ./output/masked_1K/masked_ori.png
    --input_size 512
```

# run demo canny
```bash
python demo.py
    --chkpt checkpoint/masked_1K/masked_1K_canny_200.pth
    --data_path ../archive/masked_crop/canny/val
    --save_path ./output/masked_1K/masked_canny.png
    --input_size 512
```

# ori2canny
```bash
python demo.py
    --chkpt ./checkpoint/masked_1K/masked_1K_ori_200.pth
    --data_path ../archive/masked_crop/canny/val
    --save_path ./output/masked_1K/ori2canny.png
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

# run class.py
```bash
python class.py \
    --batch_size 80 \
    --epochs 100 \
    --save_ckpt_freq 50 \
    --ori_ckpt_path ./checkpoint/masked_1K/masked_1K_ori_200.pth \
    --canny_ckpt_path ./checkpoint/masked_1K/masked_1K_canny_200.pth \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --step_size 30 \
    --gamma  0.5
```

# run class2.py
```bash
python class2.py \
    --batch_size 20 \
    --epochs 100 \
    --save_ckpt_freq 50 \
    --ori_ckpt_path ./checkpoint/masked_1K/masked_1K_ori_200.pth \
    --canny_ckpt_path ./checkpoint/masked_1K/masked_1K_canny_200.pth \
    --lr 1e-3 \
    --weight_decay 1e-2 \
    --step_size 10 \
    --gamma  0.7
```

# run K-fold.py
```bash
python K-fold.py \
    --batch_size 32 \
    --epochs 100 \
    --save_ckpt_freq 50 \
    --ori_ckpt_path ./checkpoint/masked_1K/masked_1K_ori_200.pth \
    --canny_ckpt_path ./checkpoint/masked_1K/masked_1K_canny_200.pth \
    --lr 1e-3 \
    --weight_decay 1e-2 \
    --step_size 10 \
    --gamma  0.7 \
    --ori_train_path ../masked_1K/ori \
    --canny_train_path ../masked_1K/canny 
```


python onlyUseOri.py \
    --batch_size 80 \
    --epochs 200 \
    --save_ckpt_freq 50 \
    --ori_ckpt_path ./checkpoint/masked_1K/masked_1K_ori_200.pth \
    --canny_ckpt_path ./checkpoint/masked_1K/masked_1K_canny_200.pth \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --step_size 30 \
    --gamma  0.5

python onlyUseCanny.py \
    --batch_size 80 \
    --epochs 100 \
    --save_ckpt_freq 50 \
    --ori_ckpt_path ./checkpoint/masked_1K/masked_1K_ori_200.pth \
    --canny_ckpt_path ./checkpoint/masked_1K/masked_1K_canny_200.pth \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --step_size 30 \
    --gamma  0.5