<div align="center">


<h2>
Texture Vector-Quantization and Reconstruction Aware Prediction for Generative Super-Resolution (ICLR 2026)
</h2>

[Qifan Li](),  [Jiale Zou](https://github.com/Sean2CS),  [Jinhua Zhang](https://nuanbaobao.github.io/),  [Wei Long](https://scholar.google.com/citations?user=CsVTBJoAAAAJ), [Xingyu Zhou](https://scholar.google.com/citations?user=dgO3CyMAAAAJ&hl=zh-CN&oi=sra),  [Shuhang Gu](https://scholar.google.com/citations?user=-kSTt40AAAAJ)

[![arXiv](https://img.shields.io/badge/arXiv-2509.23774-b31b1b.svg)](https://arxiv.org/pdf/2509.23774)
[![GitHub Stars](https://img.shields.io/github/stars/LabShuHangGU/TVQ-RAP?style=social)](https://github.com/LabShuHangGU/TVQ-RAP)

</div>
<img src="assert/pipeline1.png" style="border-radius: 8px">

⭐If you like this work, please help star this repo. Thanks!🤗
 

## Performance
<p align="center">
    <img src="assert/vis.png" style="border-radius: 5px"
    width="100%">
</p>



## Dependencies and Installation
```
# git clone this repository
git clone https://github.com/CVL-UESTC/TVQ-RAP.git
cd TVQ-RAP

# create new anaconda env
conda create -n TVQRAP python -y
conda activate TVQRAP

# install python dependencies
pip install -r requirements.txt
```

## Inference

#### Download Pre-trained Models
Download the pretrained SR model from [Releases](https://huggingface.co/CVLUESTC/TVQRAP/blob/main/tvqrap_sr_stage3_model.pth) and place it in the `trained_weights` folder.

#### Quick Inference
You can place any testing images in the `test_images` folder. Then specify the corresponding data path in `options/TVQRAP_test.yml`.

Then, you can get the SR outputs by running the following command:
```
bash infer.sh
```

#### Reproducing Tab 1,2 in our paper

Download and the testing data ([ImageNet-Test](https://github.com/zsyOAOA/ResShift/tree/journal) + [RealSR](https://github.com/csjcai/RealSR) + [RealSet65](https://github.com/zsyOAOA/ResShift/tree/journal)).

Run inference on each dataset one by one to obtain the SR results.

After generating the SR images, compute the evaluation metrics with:
```
#non-reference-metrics
python test-non-reference-metrics.py

#non-reference-metrics
python test-reference-metrics.py
```

## Training

### Preparing Dataset
Download training dataset: [ImageNet](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).

Download and the testing data [ImageNet-Test](https://github.com/zsyOAOA/ResShift/tree/journal).

Complete training/testing data path in the configuration file in `option/xxx.yml`.

### Preparing Pretrained Weights
Pretrained weights of all stgaes can be found in the [Hugggingface](https://huggingface.co/CVLUESTC/TVQRAP). You can choose to train any individual stage based on our released weights or all stages by yourself.

#### Stage I - Tokenizer
Training tokeizer (4 x 24GB GPUs):
```
bash train1.sh
```

#### Stage II - Predictor (cross-entropy loss pretraining)
Specify the path to the pretrained Stage I weights (either the ones we provide or those trained by yourself) in the corresponding field of `options/TVQRAP_stage2.yml`.
Then, training Predictor using cross-entropy loss (2 x 24GB GPUs):
```
bash train2.sh
```

#### Stage III - Predictor (RAP finetuning)
Specify the path to the pretrained Stage II weights (either the ones we provide or those trained by yourself) in the corresponding field of `options/TVQRAP_stage3.yml`.
Then, training Predictor using cross-entropy loss (2 x 24GB GPUs):
```
bash train3.sh
```


## <a name="cite"></a> 🥰 Citation

Please cite us if our work is useful for your research.

```
@article{li2025texture,
  title={Texture Vector-Quantization and Reconstruction Aware Prediction for Generative Super-Resolution},
  author={Li, Qifan and Zou, Jiale and Zhang, Jinhua and Long, Wei and Zhou, Xingyu and Gu, Shuhang},
  journal={arXiv preprint arXiv:2509.23774},
  year={2025}
}
```

## Acknowledgement

This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [CodeFormer](https://github.com/sczhou/CodeFormer).

## Contact

If you have any questions, feel free to approach me at qifanli.lqf@gmail.com 
