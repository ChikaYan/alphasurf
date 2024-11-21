
# αSurf: Implicit Surface Reconstruction for Semi-Transparent and Thin Objects with Decoupled Geometry and Opacity



[Project Page](https://alphasurf.netlify.app/), [Data](https://drive.google.com/file/d/10OvhfGj9P4t5esdy9NTriBWm-jKE2n8w/view)


## Setup

First create the virtualenv; we recommend using conda:
```sh
conda env create -f environment.yml
conda activate alphasurf
```

Then install pytorch with CUDA support via:

```sh
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```


Then install the C++/CUDA extension at root by simply running

```
pip install .
```
In the repo root directory.

If your CUDA toolkit is older than 11, then you will need to install CUB as follows:
`conda install -c bottler nvidiacub`.
Since CUDA 11, CUB is shipped with the toolkit.

## Train and eval

An example script for training Plenoxels + our method is avaliable at train_eval.sh

```
bash train_eval.sh
```


Citation:
```
@inproceedings{alphasurf,
    title={AlphaSurf: Implicit Surface Reconstruction for Semi-Transparent and Thin Objects with Decoupled Geometry and Opacity},
    author={Tianhao Walter Wu and Hanxue Liang and Fangcheng Zhong and Gernot Riegler and Shimon Vainer and Jiankang Deng and Cengiz Oztireli}
    journal={3DV},
    year={2025}
}
```





