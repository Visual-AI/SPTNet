# SPTNet: An Efficient Alternative Framework for Generalized Category Discovery with Spatial Prompt Tuning


<p align="center">
    <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Wen_Parametric_Classification_for_Generalized_Category_Discovery_A_Baseline_Study_ICCV_2023_paper.html"><img src="https://img.shields.io/badge/ICLR%202024-8A2BE2"></a>
<!--     <a href="https://arxiv.org/abs/2211.11727"><img src="https://img.shields.io/badge/arXiv-2401.11727-b31b1b"></a> -->
</p>
<p align="center">
	SPTNet: An Efficient Alternative Framework for Generalized Category Discovery with Spatial Prompt Tuning (ICLR 2024)<br>
  By
  <a href="https://whj363636.github.io/">Hongjun Wang</a>, 
  <a href="https://sgvaze.github.io/">Sagar Vaze</a>, and 
  <a href="https://www.kaihan.org/">Kai Han</a>.
</p>

![teaser](assets/teaser.jpg)



## Running 🛠️

First, you need to clone the SPTNet repository from GitHub. Open your terminal and run the following command:

```
git clone https://github.com/whj363636/SPTNet.git
cd SPTNet
```

We recommend setting up a conda environment for the project:

```bash
conda create --name=spt python=3.9
conda activate spt
pip install requirements.txt
```

### Config

Set paths to datasets and desired log directories in ```config.py```


### Datasets

We use fine-grained benchmarks in this paper, including:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6)

We also use generic object recognition datasets, including:

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet-100/1K](https://image-net.org/download.php)


### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```


## Results
Our results:



## Citing this work

If you find this repo useful for your research, please consider citing our paper:

```

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
