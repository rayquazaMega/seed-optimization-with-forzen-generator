# Seed Optimization with Frozen Generator

Official PyTorch implementation of the
paper Seed Optimization with Frozen Generator for Superior Zero-shot Low-light Image Enhancement in IEEE TCSVT 2024.

[**Paper**](./docs/paper.pdf)


## Dependencies and Installation


1. Create conda environment

```
conda create --name drp python=3.6
conda activate drp
```

2. Clone repo

```
git clone https://github.com/rayquaza/xxx.git
```
3. Install dependencies

```
cd seed-optimization-with-forzen-generator
pip install -r requirements.txt
```

## Run

Specify the input path ```input_path```, the output directory ```output_dir```, and other hyper-parameters. Then run

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --input_path path_to_input_image.png --output_dir path_to_output_dir
```

## Citation

If you find our work useful in your research or publication, please cite it:

```
@ARTICLE{gu2024seedoptimze,
  author={Gu, Yuxuan and Jin, Yi and Wang, Ben and Wei, Zhixiang and Ma, Xiaoxiao and Wang, Haoxuan and Ling, Pengyang and Chen, Huaian and Chen, Enhong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Seed Optimization with Frozen Generator for Superior Zero-shot Low-light Image Enhancement}, 
  year={2024},
  pages={Early Access},
  doi={10.1109/TCSVT.2024.3454763}}
```

## Further comments

The code is heavily borrowed from [discrepant-untrained-nn-priors](https://github.com/sherrycattt/discrepant-untrained-nn-priors).

The code is provided as-is for academic use only and without any guarantees. Please
contact [the author](mailto:x01914176@gmail.com) to report any bugs. 
