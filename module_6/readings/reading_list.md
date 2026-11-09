# Module 6 — Reading List

---

## Before Lecture 29 (assigned in Lecture 28)

### [Primary] Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 9 (Convolutional Networks).

Sections 9.1–9.4. Free at deeplearningbook.org.

### [Primary] `module_6/readings/cnn_architecture_primer.md`

Quick reference for the conv/pool/receptive-field concepts.

### [Optional] LeCun, Bottou, Bengio & Haffner, *Gradient-based learning applied to document recognition*, Proc. IEEE 86(11), 1998. The LeNet paper.

### [Optional] He, Zhang, Ren & Sun, *Deep Residual Learning for Image Recognition*, CVPR 2016. ResNet.

### [Optional] From image recognition to field prediction (the story behind L28's slide 2)

One influential route in deep learning for images ran from **recognition** (an image in, one label out) to
**segmentation** and other dense prediction (a label map aligned with the input, one label per pixel). Field
prediction in mechanics borrowed that step: the geometry or material layout of a part or a microstructure goes in
as an image, with the material, loads and boundary conditions fixed or supplied as extra inputs, and the stress
field comes out as an image. Flattening such an image into a vector loses no data, but a dense layer does not
enforce neighborhoods or share weights across positions; convolutional networks build both in, which is why they
are the usual tool for these problems. Lab 6
(image → one number) follows the recognition branch; L30 and Lab 6's U-Net stretch follow the field branch. Read
any of these for a sense of the field; none is required.

- *Recognition:* Krizhevsky, Sutskever & Hinton, *ImageNet Classification with Deep Convolutional Neural
  Networks*, NeurIPS 2012 (AlexNet).
- *Dense prediction and image translation:* Long, Shelhamer & Darrell 2015 (FCN, segmentation) and Isola et al. 2017
  (pix2pix, general image-to-image translation with a conditional GAN), both under
  "Image-to-image for science" below; Ronneberger et al. 2015 (U-Net), assigned in L30.
- *Mechanics, an early surrogate:* Liang, Liu, Martin & Sun, *A deep learning approach to estimate stress
  distribution: a fast and accurate surrogate of finite-element analysis*, J. R. Soc. Interface 15(138), 20170844,
  2018. Aorta geometry in, wall stress distribution out.
- *Mechanics, stress as an image:* Nie, Jiang & Kara, *Stress field prediction in cantilevered structures using
  convolutional neural networks*, J. Comput. Inf. Sci. Eng. 20(1), 011002, 2020. Geometry, loads and boundary
  conditions as image channels in, the von Mises field out.
- *Mechanics, composites:* Yang, Yu & Buehler, *Deep learning model to predict complex stress and strain fields in
  hierarchical composites*, Science Advances 7(15), eabd7416, 2021. A U-Net generator trained as a GAN.
- *Mechanics, composites:* Bhaduri, Gupta & Graham-Brady, *Stress field prediction in fiber-reinforced composite
  materials using a deep learning approach*, Composites Part B 238, 109879, 2022. A U-Net on fiber composites.
- *Mechanics, heterogeneous materials:* Mianroodi, Siboni & Raabe 2021, listed below under Lab 6's stretch.

---

## Before Lecture 30 (assigned in Lecture 29)

### [Primary] Bessa, Bostanabad, Liu, Hu, Apley, Brinson, Chen & Liu, *A framework for data-driven analysis of materials under uncertainty: countering the curse of dimensionality*, CMAME 320, 2017.

The data-driven workflow Lab 6 follows: design of experiments, RVE simulations to build a database, then machine
learning on that database. It is not a CNN paper; read Sections 1–3 for the workflow and its cost argument.

### [Primary] `module_6/readings/unet_for_mechanics.md`

Companion reference for L30.

### [Optional] Yang et al., *Establishing structure-property localization linkages for elastic deformation of three-dimensional high contrast composites using deep learning approaches*, Acta Materialia 166, 2019. 3-D extension.

### [Optional] Cecen, Dai, Yabansu, Kalidindi & Song, *Material structure-property linkages using three-dimensional convolutional neural networks*, Acta Materialia 146, 2018.

---

## Before Lab 6 stretch (assigned in Lecture 30)

### [Primary] Ronneberger, Fischer & Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.

The U-Net paper.

### [Optional] Mianroodi, Siboni & Raabe, *Teaching solid mechanics to artificial intelligence — a fast solver for heterogeneous materials*, npj Computational Materials 7, 99, 2021.

A U-Net-style network that predicts stress fields in heterogeneous materials, elastic and elastoplastic, from
microstructure images: the full-field version of Lab 6's problem (the optional U-Net task).

### [Optional] Lu, Jin, Pang, Zhang & Karniadakis, *Learning nonlinear operators via DeepONet*, Nature Machine Intelligence 3, 2021.

Section 1 only — preview of M7.

---

## Reference

### Mechanics CNN papers (selected)

- Lefik et al. 2003 (mentioned in M4); NN constitutive precursors.
- Bessa 2017 (above); the data-driven RVE workflow (not a CNN).
- Yang, …, Agrawal, *Deep learning approaches for mining structure-property linkages in high contrast composites
  from simulation datasets*, Comput. Mater. Sci. 151, 278–287, 2018; a 3-D CNN that maps microstructure to
  effective stiffness.
- Mozaffar 2019 (mentioned in M4); RNN-vs-CNN for plasticity.
- Yang 2019 (above); 3-D.

### Modern CNN architectures (for context)

- Krizhevsky et al. 2012 — AlexNet.
- He et al. 2016 — ResNet.
- Huang et al. 2017 — DenseNet.
- Liu et al. 2022 — ConvNeXt.

### Image-to-image for science

- Long, Shelhamer & Darrell, *Fully Convolutional Networks for Semantic Segmentation*, CVPR 2015. A landmark fully convolutional segmentation architecture; precursor to U-Net.
- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017. The pix2pix paper.
