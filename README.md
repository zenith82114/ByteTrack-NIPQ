
# ByteTrack-NIPQ

**ByteTrack + NIPQ + KETI Pruning**

Please refer to [the original repository](https://github.com/ifzhang/ByteTrack) for setup, dataset preparation and other details.


## Original Papers

> [**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**](https://arxiv.org/abs/2110.06864)
>
> Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang
>
> *[arXiv 2110.06864](https://arxiv.org/abs/2110.06864)*

> [**NIPQ: Noise proxy-based Integrated Pseudo-Quantization**](https://arxiv.org/abs/2206.00820)
>
> Juncheol Shin, Junhyuk So, Sein Park, Seungyeop Kang, Sungjoo Yoo, Eunhyeok Park
>
> *[arXiv 2206.00820](https://arxiv.org/abs/2206.00820)*


## Training

I only implemented and tested with the `yolox_x_ablation` exp.

* **ByteTrack + NIPQ**

```shell
cd <ByteTrack_HOME>
OMP_NUM_THREADS=8 python tools/train.py -f exps/example/mot/yolox_x_ablation.py -d 4 -b 16 --fp16 -o -c pretrained/bytetrack_ablation.pth.tar --nipq --target 8
```

* **ByteTrack + Pruning**

```shell
cd <ByteTrack_HOME>
OMP_NUM_THREADS=8 python tools/train.py -f exps/example/mot/yolox_x_ablation.py -d 4 -b 8 -o -c pretrained/bytetrack_ablation.pth.tar --pruning --prune_ratio 0.1
```

If both `--nipq` and `--pruning` are set, the pretrained weight is pruned first and then fine-tuned with NIPQ.

**NOTE:** if `--pruning` is set I had to disable `--fp16` and halve the batch size.


## Tracking

* **Evaluation on MOT17 half val**

```shell
cd <ByteTrack_HOME>
python3 tools/track.py -f exps/example/mot/yolox_x_ablation.py -c YOLOX_outputs/yolox_x_ablation/best_ckpt.pth.tar -b 1 -d 1 --fp16 --fuse
```

**NOTE:** `--nipq` and/or `--pruning` and other related arguments should be given the same as in the training command for that checkpoint.
