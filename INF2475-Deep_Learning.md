# Geometry-Aware High-Frequency Feature Extraction from Monocular Images

**Marcello Braga Nascif Filho**

Literature Review for INF2475 – Introduction to Deep Learning
Advisor: Prof. Paulo Ivson

Rio de Janeiro
April 2026

---

## Approval

Approved by the Examination Committee.

- Prof. Paulo Ivson — Advisor, Departamento de Informática – PUC-Rio

Rio de Janeiro, April 30th, 2026

---

## Bibliographic Data

> All rights reserved.
>
> Marcello Braga Nascif Filho
>
> **Geometry-Aware High-Frequency Feature Extraction from Monocular Images** / Marcello Braga Nascif Filho; advisor: Paulo Ivson. – Rio de Janeiro: PUC-Rio, Departamento de Informática, 2026.
>
> v., 19 f: il. color. ; 30 cm
>
> Projeto Final de Graduação - Pontifícia Universidade Católica do Rio de Janeiro, Departamento de Informática.
>
> Inclui bibliografia
>
> 1. Geometric feature extraction. 2. High-frequency features. 3. Discrete Wavelet Transform. I. Ivson, Paulo. II. Pontifícia Universidade Católica do Rio de Janeiro. Departamento de Informática. III. Título.
>
> CDD: 004

---

## Acknowledgments

*(not yet written)*

---

## Abstract

Braga Nascif Filho, Marcello; Ivson, Paulo (Advisor). **Geometry-Aware High-Frequency Feature Extraction from Monocular Images**. Rio de Janeiro, 2026. 19p. Projeto Final de Graduação – Departamento de Informática, Pontifícia Universidade Católica do Rio de Janeiro.

*(abstract body not yet written)*

**Keywords:** Geometric feature extraction; High-frequency features; Discrete Wavelet Transform.

---

## Resumo

Braga Nascif Filho, Marcello; Ivson, Paulo (Orientador). **Geometry-Aware High-Frequency Feature Extraction from Monocular Images**. Rio de Janeiro, 2026. 19p. Projeto Final de Graduação – Departamento de Informática, Pontifícia Universidade Católica do Rio de Janeiro.

*(resumo body not yet written)*

**Palavras-chave:** Geometric feature extraction; High-frequency features; Discrete Wavelet Transform.

---

## Table of Contents

1. [Introduction](#1-introduction) — p. 10
2. [Literature Review](#2-literature-review) — p. 12
   1. [Implicit Neural Representations and Spectral Bias](#21-implicit-neural-representations-and-spectral-bias) — p. 12
   2. [Monocular Depth and Surface Normal Estimation](#22-monocular-depth-and-surface-normal-estimation) — p. 13
   3. [Frequency-Aware Geometric Feature Extraction](#23-frequency-aware-geometric-feature-extraction) — p. 13
3. [Proposed Approach](#3-proposed-approach) — p. 15
4. [Expected Benefits](#4-expected-benefits) — p. 16
5. [Bibliography](#bibliography) — p. 17

*List of Figures: (none)*
*List of Tables: (none)*

---

## 1. Introduction

Extracting fine geometric details from 2D images is a fundamental challenge in computer vision, with direct implications for tasks such as 3D reconstruction, surface inspection, and scene understanding. When observing a single image of a three-dimensional object, the signal reaching the camera conflates two fundamentally distinct sources of high-frequency variation: genuine geometric detail, such as surface curvature changes, fine ridges, creases, and depth discontinuities, and purely textural variation, such as printed patterns, fabric weaves, or painted grain that produce strong image gradients without any corresponding change in surface geometry. Disentangling these two sources from a 2D observation alone is the central problem this work addresses.

Existing approaches attack the problem from two largely independent directions. Frequency-domain methods, including wavelet-based encodings [1] and frequency-aware implicit neural representations [2, 3], are highly effective at localizing regions of high-frequency variation in an image. However, by construction they respond equally to geometric and textural variation: a flat shirt with a striped pattern and a genuinely corrugated surface may produce indistinguishable wavelet energy maps. Geometric estimation methods, on the other hand, in particular monocular depth [4, 5, 6] and surface normal estimators [7, 8], encode physically grounded information about how a surface is oriented and how far it lies from the camera. Their limitation is complementary: they tend to oversmooth fine geometric detail, especially in low-texture regions, and their predictions can be corrupted by strong appearance cues that are texturally rather than geometrically driven [9].

We propose to address this gap by combining frequency-domain analysis with monocular geometric estimation. Specifically, we extract high-frequency feature maps via discrete wavelet transform (DWT), obtaining per-pixel energy scores that identify regions of strong signal variation. We then combine these maps with surface normal and depth estimates to produce a geometrically grounded attention map, one that highlights regions where high-frequency image variation corresponds to genuine surface geometry changes, and suppresses regions where such variation is purely textural in origin. The key insight is that the intersection of high wavelet energy and significant normal variation is a strong indicator of real geometric detail, whereas high wavelet energy with a locally constant normal field is a reliable signature of textural variation.

---

## 2. Literature Review

The problem of extracting fine geometric details from images spans three bodies of literature: implicit neural representations and their treatment of spectral bias, monocular geometric estimation covering depth and surface normals, and methods that explicitly combine frequency decomposition with geometric reasoning.

### 2.1 Implicit Neural Representations and Spectral Bias

Coordinate-based neural networks have emerged as a powerful paradigm for continuous signal representation, mapping spatial coordinates directly to signal values. A fundamental obstacle in this setting is the spectral bias [10]: standard MLPs with ReLU activations converge preferentially to low-frequency components of the target function, rendering them incapable of faithfully representing fine geometric detail without explicit intervention.

The foundational responses to this limitation came from two complementary directions. Tancik et al. [11] showed that mapping input coordinates through random Fourier features transforms the network's Neural Tangent Kernel into a stationary kernel whose bandwidth is controllable, effectively lifting the spectral bias for a chosen frequency range. Sitzmann et al. [12] proposed SIREN, replacing ReLU with sinusoidal activations, noting that every derivative of a SIREN is itself a SIREN, a property that enables direct supervision via partial differential equations and makes the architecture well-suited for representing signed distance functions.

Subsequent work refined the frequency-locality trade-off. WIRE [2] adopts the complex Gabor wavelet as activation, situating it at the Heisenberg uncertainty bound and achieving superior robustness in inverse problems. FINER [13] introduces a variable-periodic activation whose local frequency scales with input magnitude, allowing explicit spectral tuning via bias initialization. Most recently, FLAIR [3] unifies frequency selectivity and spatial localization through a Band-Localized Activation (BLA) and a Wavelet-Energy-Guided Encoding (WEGE), which computes per-pixel DWT energy scores across subbands and concatenates them with spatial coordinates as explicit frequency priors. FLAIR is, to our knowledge, the first INR to use wavelet energy maps as a spatial conditioning signal, a design principle directly relevant to the approach proposed in this work.

### 2.2 Monocular Depth and Surface Normal Estimation

The estimation of depth and surface normals from a single image provides a direct route to geometric information without requiring multi-view setups. On the depth side, Ranftl et al. [4] established the MiDaS framework with a scale-and-shift-invariant loss that enables training across heterogeneous datasets. The DPT architecture [14] replaced CNN backbones with Vision Transformers, providing globally consistent feature aggregation. Depth Anything [5] scaled this paradigm to tens of millions of images via self-training, recovering fine structural details at inference speed. Marigold [6] reframes depth estimation as conditional diffusion generation, exploiting rich scene priors to avoid the mode-averaging artifacts of regression-based approaches.

Surface normal maps are more directly informative about fine geometric structure than depth maps, since they encode first-order surface orientation rather than a scalar distance that requires numerical differentiation to interpret geometrically. DSINE [7] is particularly relevant here: it injects two geometry-specific inductive biases, conditioning on per-pixel ray direction and modeling inter-pixel normal relationships as relative SO(3) rotations, effectively producing predictions that decouple textural gradients from genuine surface orientation changes. On the generative side, GeoWizard [8] jointly predicts depth and normals via fine-tuned Stable Diffusion; Lotus [15] compresses the diffusion pipeline to a single step with a detail-preserving token; StableNormal [16] combines a one-step coarse estimator with semantic-guided refinement to produce stable and sharp normals on difficult surfaces. Most recently, Vision-Banana [17] reformulates normal estimation as RGB image generation via instruction-tuned generative models, reporting competitive angular error across multiple benchmarks, though the work remains under review at the time of writing.

### 2.3 Frequency-Aware Geometric Feature Extraction

The combination of frequency analysis with geometric estimation has been explored in specific settings, though not yet as a general strategy for isolating geometric from textural high-frequency content. WaveletMonoDepth [1] predicts depth as a multi-scale wavelet decomposition, motivating the approach by the piecewise-flat structure of depth maps: coarse coefficients encode global layout while fine coefficients capture geometrically meaningful discontinuities. This is precisely the observation that motivates our work: wavelet energy in fine subbands does carry geometric signal, but only in regions where image gradients are driven by surface variation rather than by appearance.

The fine-grained geometry literature provides direct evidence for the necessity of frequency decomposition in this context. Huynh et al. [9] showed that recovering mesoscopic facial geometry requires separate sub-networks for mid- and high-frequency displacement, and that failing to decouple frequency bands leads to systematic loss of microgeometric detail drowned out by the dominant low-frequency signal. Finally, MonoSDF [18] demonstrated that monocular depth and normal pseudo-labels substantially improve implicit surface reconstruction in texturally ambiguous regions, establishing the utility of combining monocular geometric priors with learned representations, the precise combination this work seeks to operationalize at the level of feature extraction.

---

## 3. Proposed Approach

The central limitation of existing methods, as identified in the previous chapter, is that frequency-domain approaches respond equally to geometric and textural variation, while geometric estimators tend to oversmooth the fine detail that frequency decomposition is designed to capture. We propose to address this by combining both signals explicitly.

The proposed approach operates in three stages:

1. **Discrete Wavelet Transform (DWT).** A DWT is applied to the input image to produce a set of per-pixel energy scores across frequency subbands. These scores identify regions of strong high-frequency variation in the image, following the formulation used in FLAIR's WEGE module [3], where the residual between reconstructed high- and low-frequency subbands is used as a spatial attention prior.

2. **Monocular surface normal estimation.** A monocular surface normal estimator, such as DSINE [7], is applied to the same image to produce a per-pixel normal map. The normal map encodes the local orientation of the surface at each point and, unlike depth maps, does not require differentiation to reveal geometric variation, making it a direct and noise-robust proxy for surface structure.

3. **Fusion into a geometrically grounded attention map.** The wavelet energy map and the normal map are combined to produce a geometrically grounded attention map. The combination is guided by the following principle: a pixel with high wavelet energy and significant local normal variation is a strong candidate for genuine geometric detail. A pixel with high wavelet energy but a locally constant normal field is more likely to reflect textural rather than geometric variation. The fusion of these two signals allows the method to suppress texture-driven false positives that would contaminate a purely frequency-based feature map.

What is new in this approach relative to the existing literature is the explicit cross-modal grounding of frequency features in geometric priors. Prior work either extracts frequency features without geometric awareness, as in FLAIR [3], WIRE [2], and WaveletMonoDepth [1], or estimates geometry without explicit frequency decomposition, as in DSINE [7], Marigold [6], and GeoWizard [8]. The proposed combination has not, to our knowledge, been explored as a standalone feature extraction strategy aimed at isolating geometrically meaningful high-frequency structure from textural noise.

---

## 4. Expected Benefits

The proposed approach is expected to yield two primary benefits.

First, the resulting attention maps should be more faithful representations of true surface microgeometry than those produced by either frequency-only or geometry-only methods. By requiring that high-frequency image variation be corroborated by local normal variation before being flagged as geometric detail, the method discards appearance-driven false positives that plague purely frequency-based approaches [9]. Conversely, by using wavelet energy to identify candidate regions, the method avoids the oversmoothing tendency of geometric estimators, which often suppress fine detail in favor of globally consistent predictions [7, 6].

Second, the method is compositional by design. Because it relies on independently established components — DWT-based frequency analysis and state-of-the-art monocular geometric estimators — improvements in either component can be incorporated without architectural redesign. As monocular normal estimation continues to advance [17, 15, 16], the geometric grounding of the attention map will improve accordingly, without requiring changes to the frequency extraction pipeline.

More broadly, geometrically grounded high-frequency feature maps of the kind proposed here could serve as a useful intermediate representation for downstream tasks that depend on accurate surface microgeometry, including detail-preserving 3D reconstruction, surface defect detection, and material analysis from single images.

---

## Bibliography

[1] RAMAMONJISOA, M.; DU, Y.; LEPETIT, V. Single image depth prediction with wavelet decomposition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[2] SARAGADAM, V.; LEJEUNE, D.; TAN, J.; BALAKRISHNAN, G.; VEERARAGHAVAN, A.; BARANIUK, R. G. WIRE: Wavelet implicit neural representations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[3] KO, S.; YOON, S.; KYE, D.; MIN, K.; EOM, C.; OH, J. FLAIR: Frequency- and locality-aware implicit neural representations. arXiv preprint arXiv:2508.13544, 2025.

[4] RANFTL, R.; LASINGER, K.; HAFNER, D.; SCHINDLER, K.; KOLTUN, V. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.

[5] YANG, L.; KANG, B.; HUANG, Z.; XU, X.; FENG, J.; ZHAO, H. Depth anything: Unleashing the power of large-scale unlabeled data. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[6] KE, B.; OBUKHOV, A.; HUANG, S.; METZGER, N.; DAUDT, R. C.; SCHINDLER, K. Repurposing diffusion-based image generators for monocular depth estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[7] BAE, G.; DAVISON, A. J. Rethinking inductive biases for surface normal estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[8] FU, X.; YIN, W.; HU, M.; WANG, K.; MA, Y.; TAN, P.; SHEN, S.; LIN, D.; LONG, X. GeoWizard: Unleashing the diffusion priors for 3D geometry estimation from a single image. In: Proceedings of the European Conference on Computer Vision (ECCV), 2024.

[9] HUYNH, L.; CHEN, W.; SAITO, S.; XING, J.; NAGANO, K.; JONES, A.; DEBEVEC, P.; LI, H. Mesoscopic facial geometry inference using deep neural networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[10] RAHAMAN, N.; BARATIN, A.; ARPIT, D.; DRAXLER, F.; LIN, M.; HAMPRECHT, F.; BENGIO, Y.; COURVILLE, A. On the spectral bias of neural networks. In: Proceedings of the International Conference on Machine Learning (ICML), 2019.

[11] TANCIK, M.; SRINIVASAN, P.; MILDENHALL, B.; FRIDOVICH-KEIL, S.; RAGHAVAN, N.; SINGHAL, U.; RAMAMOORTHI, R.; BARRON, J.; NG, R. Fourier features let networks learn high frequency functions in low dimensional domains. In: Advances in Neural Information Processing Systems (NeurIPS), 2020.

[12] SITZMANN, V.; MARTEL, J.; BERGMAN, A.; LINDELL, D.; WETZSTEIN, G. Implicit neural representations with periodic activation functions. In: Advances in Neural Information Processing Systems (NeurIPS), 2020.

[13] LIU, Z.; ZHU, H.; ZHANG, Q.; FU, J.; DENG, W.; MA, Z.; GUO, Y.; CAO, X. FINER: Flexible spectral-bias tuning in implicit neural representation by variable-periodic activation functions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[14] RANFTL, R.; BOCHKOVSKIY, A.; KOLTUN, V. Vision transformers for dense prediction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[15] HE, J.; LI, H.; YIN, W.; LIANG, Y.; LI, L.; ZHOU, K.; LIU, H.; LIU, B.; CHEN, G. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. In: International Conference on Learning Representations (ICLR), 2025.

[16] YE, C.; NIE, L.; YI, X.; XU, Y.; CAO, Y.; WANG, J.; HUANG, G.; HAN, X. StableNormal: Reducing diffusion variance for stable and sharp normal. ACM Transactions on Graphics, 2024.

[17] GABEUR, V.; LONG, J.; PENG, S.; OTHERS. Image generators are generalist vision learners. arXiv preprint arXiv:2604.20329, 2026.

[18] YU, Z.; PENG, S.; NIEMEYER, M.; SATTLER, T.; GEIGER, A. MonoSDF: Exploring monocular geometric cues for neural implicit surface reconstruction. In: Advances in Neural Information Processing Systems (NeurIPS), 2022.
