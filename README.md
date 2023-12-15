# Run
```bash
python pretain.py \
        --data_path ../maske/train/ \
        --save_pth  ./output/checkpoint_ori_200.pth \
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
python pretain.py \
        --data_path ../archive/masked_crop/train/1 \
        --save_path  ./output/checkpoint_ori_200.pth \
        --batch_size 20 \
        --epochs 200 \
        --save_ckpt_freq 200 \
        --input_size 512 \
        --lr 1e-4 \
        --step_size 50 \
        --gamma 0.7
```

# local run canny
```bash
python pretain.py \
        --data_path ../archive/masked_crop/canny/train \
        --save_path  ./output/checkpoint_canny_200.pth \
        --batch_size 20 \
        --epochs 200 \
        --save_ckpt_freq 200 \
        --input_size 512 \
        --lr 1e-4 \
        --step_size 50 \
        --gamma 0.7
```