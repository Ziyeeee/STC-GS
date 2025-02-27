<h2 align="center">High-Dynamic Radar Sequence Prediction for<br>Weather Nowcasting Using Spatiotemporal Coherent Gaussian Representation</h2>
<p align="center">
    <a href="https://ziyeeee.github.io/">Ziye Wang</a>
    ·
    <a href="https://github.com/IranQin">Yiran Qin</a>
    ·
    Lin Zeng
    ·
    <a href="http://www.zhangruimao.site">Ruimao Zhang</a>
</p>
<h3 align="center"><a href="https://arxiv.org/abs/2502.14895">Paper</a> | <a href="https://ziyeeee.github.io/stcgs.github.io/">Project Page</a> | <a href="https://huggingface.co/datasets/Ziyeeee/3D-NEXRAD">Dataset</a> </h3>
<h3>Code is being gradually released.</h3>

## Installation

Our code relies on Python 3.10 and CUDA 12.4, but it should work with CUDA >= 11.8 as well.

1. Clone STC-GS.
```
git clone https://github.com/Ziyeeee/STC-GS.git --recursive
cd STC-GS
```

2. Create the environment, here we show an example using conda.
```
conda create -n stcgs python=3.10
conda activate stcgs
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

3. Compile the cuda kernel.
```
pip install submodules/diff-gaussian-rasterization-radar
```

## Datasets

Please refer to [3D-NEXRAD](https://huggingface.co/datasets/Ziyeeee/3D-NEXRAD) for more details.

1. Extract from `tars.gz.*`.
```
cat nexrad-[YYYY].tar.gz.* | tar -zxv - -C [your_dataset_dir]/
```

2. Split the dataset.

```
python utils/preprocess.py --path [your_dataset_path]
```

## Running

### Re-representation

```
python mp_represent.py --num_processes [your_cpu_cnt] --hdf_path [your_dataset_path]
```


### Prediction



## Citation

```
@inproceedings{wang2025highdynamic,
    title={High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian Representation},
    author={Ziye Wang and Yiran Qin and Lin Zeng and Ruimao Zhang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=Cjz9Xhm7sI}
}
```
