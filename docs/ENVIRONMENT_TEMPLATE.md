# ENVIRONMENT TEMPLATE

用途：记录当前可复现实验环境（已按当前容器补全，可继续手动追加）。
更新时间：2026-04-05（UTC）

## 1) 基础环境

- `conda env name`: `<not set>`（base）
- `python version`: `3.12.3`
- `platform/os`: `Linux x86_64 (Ubuntu kernel 5.4.0-132-generic)`
- `hostname(optional)`: `autodl-container-17b111a152-76a7669c`

## 2) 深度学习依赖

- `torch version`: `2.8.0+cu128`
- `torchvision version`: `0.23.0+cu128`
- `cuda available (torch.cuda.is_available)`: `True`
- `cuda runtime version`:
  - `torch.version.cuda`: `12.8`
  - `nvidia-smi CUDA Version`: `13.0`
- `cuDNN version`: `91002`

## 3) 拓扑相关依赖

- `cripser version`: `0.0.25`
- `numpy version`: `1.26.4`
- `scipy version`: `1.17.0`

## 4) 关键库（可选）

- `segmentation_models_pytorch version`: `0.5.0`
- `pyyaml version`: `6.0.3`
- `tqdm version`: `4.66.2`

## 5) GPU / CUDA 信息

- `nvidia-smi (GPU model)`: `NVIDIA GeForce RTX 3080`
- `nvidia-smi (driver version)`: `580.105.08`
- `nvidia-smi (CUDA version)`: `13.0`
- `显存容量`: `10240 MiB`

## 6) 快速采集命令（可选）

```bash
python -V
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('torch cuda', torch.version.cuda)"
python -c "import cripser; print('cripser', getattr(cripser, '__version__', 'unknown'))"
nvidia-smi
```

## 7) 备注

- 本模板用于“环境快照”；每次关键实验前建议复制一份并填写。
- 当前容器未暴露 `CONDA_DEFAULT_ENV`，如后续切换环境建议手工补上环境名。
- 若依赖升级（尤其 `torch`、`cripser`、CUDA），建议新增一条变更记录。
