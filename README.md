# C^2ROPE: Causal Continuous Rotary Positional Encoding for 3D Large Multimodal-Models Reasoning

[![arXiv](https://img.shields.io/badge/Arxiv-2601.20520-b31b1b.svg?logo=arXiv)](https://arxiv.org) <a href="https://arxiv.org"></a> 


> **[C^2ROPE: Causal Continuous Rotary Positional Encoding for 3D Large Multimodal-Models Reasoning](https://arxiv.org)**<br>
> ICRA 2026<br>
> Koon-Ting Yip*, Qiyan Zhao, Wenhao Yu, Xiaofeng Zhang, Jianming Ji, Yanyong Zhang, Ka-Veng Yuen<br>



## 🛠️ Install
```
git clone https://github.com/ErikZ719/C2RoPE.git
cd C2RoPE
conda create -n C2RoPE python=3.10 -y
conda activate C2RoPE
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install -e .
```


## Citation
```bibtex
@inproceedings{yip2026c,
  title={C\^{}2ROPE: Causal Continuous Rotary Positional Encoding for 3D Large Multimodal-Models Reasoning},
  author={Yip, Koon-Ting and Zhao, Qiyan and Yu, Wenhao and Zhang, Xiaofeng and Ji, Jianming and Zhang, Yanyong and Yuen, Ka-Veng},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```
## Acknowledgement

This repo is built on [LLaDA-3D](https://github.com/ZCMax/LLaVA-3D). Many thanks for their efforts. The use of our code should also follow the original licenses.
