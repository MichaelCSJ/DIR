
## A Real-world Display Inverse Rendering Dataset (ICCV 2025)

<!-- <p align="center">
  <img src="./.images/teaser2.jpg" width="800px">
</p> -->

**Author:** Seokjun Choi, Hoon-Gyu Chung, Yujin Jeon, Giljoo Nam, Seung-Hwan Baek

**Conference:** IEEE/CVF International Conference on Computer Vision (ICCV), 2025

**Project page:** https://michaelcsj.github.io/DIR/

**Abstract:**
Inverse rendering aims to reconstruct geometry and reflectance from captured images. Display-camera imaging systems offer unique advantages for this task: each pixel can easily function as a programmable point light source, and the polarized light emitted by LCD displays facilitates diffuse-specular separation. Despite these benefits, there is currently no public real-world dataset captured using display-camera systems, unlike other setups such as light stages. This absence hinders the development and evaluation of display-based inverse rendering methods. In this paper, we introduce the first real-world dataset for display-based inverse rendering. To achieve this, we construct and calibrate an imaging system comprising an LCD display and stereo polarization cameras. We then capture a diverse set of objects with diverse geometry and reflectance under one-light-at-a-time (OLAT) display patterns. We also provide high-quality ground-truth geometry. Our dataset enables the synthesis of captured images under arbitrary display patterns and different noise levels. Using this dataset, we evaluate the performance of existing photometric stereo and inverse rendering methods, and provide a simple, yet effective baseline for display inverse rendering, outperforming state-of-the-art inverse rendering methods.

**Why Display Inverse Rendering?**
(TBD)

## 🚀 Steps to Get Started
### Step 1: Install Dependencies
Follow instructions below.

### Step 2: Prepare Dataset
[Datase Link](TBD)
<!-- <p align="center">
  <img src="./.images/input.png" width="400px">
  <img src="./.images/tools.png" width="400px">
</p> -->

### Step 3: Normal and BRDF Maps Recovery
Run main.py to recover the surface normal map and/or BRDF parameter maps (base color, roughness, metallic).
<!-- <p align="center">
  <img src="./.images/sample.png" width="800px">
</p> -->

### Step 4: Novel Relighting (Optional, TBD)
Run relighting.py to render images under novel directional lightings based on recovered normal map and BRDF parameter maps.
<!-- <p align="center">
  <img src="./.images/output.gif" width="400px">
</p> -->

## Required Dependencies
To successfully run the universal photometric stereo network, ensure that your system has the following dependencies installed:

- Python 3
- PyTorch
- OpenCV (cv2)

## Dataset Preparation
To run the universal photometric stereo network, you need shading images and an optional binary object mask. The object should be illuminated under arbitrary lighting sources, but shading variations should be sufficient (weak shading variations may result in poor results).

Organize your test data as follows (prefix "L" and suffix ".data" can be modified in main.py). You can process multiple datasets (A, B, ...) simultaneously, which is convenient if you are evaluating the method on the DiLiGenT benchmark.

```
YOUR_DATA_PATH
├── A
│   ├── mask.png
│   ├── imgfile1
│   ├── imgfile2
│   └── ...
├── B
│   ├── mask.png
.   ├── imgfile1
.   ├── imgfile2
    └── ...
```

## Lighting Patterns
(TBD)

## Running the Test
To run the test, execute `main.py` with the following command:

```
python train.py --name SESSION_NAME
```

or

```
python train.py --name SESSION_NAME --light_N NUM_LIGHTINGS --initial_light_pattern YOUR_DISPLAY_PATTERNS --num_basis_BRDFs NUM_BASIS
```
학습이 끝나면, \resurts\SESSION 폴더 안에 'YYYYMMDD_HHMMSS' 디렉토리에  오브젝트별 OLAT 렌더링 결과, 학습된 파라미터들이 저장된다.

You can also use the provided code (`relighting.py`) for relighting the object under novel directional lights based on the recovered attributes. Follow the instructions displayed at the end of the prompt to use it. It should look like this.

To output .avi video:
```
 python relighting.py --datadir ./YOUR_SESSION_NAME/results/OBJECT_NAME --format avi
```

## License

This project is licensed under the MIT License with a non-commercial clause. This means that you are free to use, copy, modify, and distribute the software, but you are not allowed to use it for commercial purposes. 

Copyright (c) [2025] [Seokjun Choi]

## Citation

If you find this repository useful, please consider citing this paper:

(TBD)