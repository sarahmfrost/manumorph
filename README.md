# Manumorph - A Tensorflow implementation of Deep Painterly Harmonization
Sarah Frost, David Abramov, and Manu Mathew Thomas 

[Link to our paper](https://github.com/sarahmfrost/manumorph/blob/master/manumorph.pdf)
      

# Overview of Project:
![ArchitectureOfSystem](https://github.com/sarahmfrost/manumorph/blob/master/figures/architecture.png)

Style transfer, the technique by which the style of one image is applied to the content of another, is one of the most popular and well-known uses of neural network algorithms. Deep Painterly Harmonization is an extension of style transfer, but includes a content object which is placed on the style image. The network then harmonizes the style and the content. We build on Deep Painterly Harmonization, originally implemented in Torch, and re-implement the paper in Tensorflow. We extend the uses of the algorithm to explore different categories of visual media modification. We discuss the ramifications of style harmonization and style transfer on societal concepts of art, and we compare the results of the Tensorflow and Torch algorithms. Finally, we propose a design for a web application that will allow casual creators to create new art using the algorithm, without a strong technical background. 

#### Table of Contents
* [Installation](#installation)
* [Running](#running)
* [Hyperparameters](#hyperparameter)
* [Results](#results)
* [Improvements](#improvements)
* [Credits](#credits)

## Installation

To run the project you will need:
 * python 3.5
 * tensorflow (pip install tensorflow-gpu)
 * scipy (pip install scipy)
 * numpy (pip install numpy)
 * cv2 (pip install opencv-python)
 
## Running
Once you have all the depenedencies ready, do the folowing:

In main.py, change content, style and mask to corresponding image paths

Run main.py -- python main.py

Pass 1 output will be saved as pass1_output.jpg and Pass2 output will be saved as pass2_output.jpg

## Hyperparameters
There are 3 main hyperparameters:

Content weight - controls the influence of content texture in the output

Style weight - controls how much style is applied on the content

Smoothness factor - controls smoothness across the pixels

## Results
![FourExamples](https://github.com/sarahmfrost/manumorph/blob/master/figures/4examples.png)

![Moon](https://github.com/sarahmfrost/manumorph/blob/master/figures/moon.png)

![SupperAndFrog](https://github.com/sarahmfrost/manumorph/blob/master/figures/supper%2Bfrog.png)

![sarah](https://github.com/sarahmfrost/manumorph/blob/master/figures/sarahpainting.png)


## Improvements

## Credits

A Neural Algorithm of Artistic Style (PDF in Github)
https://arxiv.org/abs/1508.06576 
https://github.com/jcjohnson/neural-style


Deep Painterly Harmonization (PDF in Github)
https://arxiv.org/abs/1804.03189
https://github.com/luanfujun/deep-painterly-harmonization

