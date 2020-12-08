# Meta-reweighting4DSRC
The implementation of ["Meta-Learning for Neural Relation Classification with Distant Supervision"](https://arxiv.org/pdf/2010.13544.pdf)

## Getting Started
### Required
- python >= 3.5
- pytorch >= 1.3.0


### Data
Considering the copyright of Wiki-KBP and TACRED, we release the intermediate file of our dataset. You can download from [here](https://drive.google.com/file/d/1oEhUvQUHi0yJD-2BM2ufE9QPTY4atxZS/view?usp=sharing) and put the kbp_extend_data file in the project.

### Running
The last parameter represents the index of five reference datasets.

```
python meta_reweighting.py 0
```


## Citation
Please consider citing the following paper if you find our codes helpful. Thank you!

```
@inproceedings{li2020meta,
  title={Meta-Learning for Neural Relation Classification with Distant Supervision},
  author={Li, Zhenzhen and Nie, Jian-Yun and Wang, Benyou and Du, Pan and Zhang, Yuhan and Zou, Lixin and Li, Dongsheng},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={815--824},
  year={2020}
}
```
