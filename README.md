# Art Style Classification
Authors: Caroline Graebel, Kseniia Holovchenko, Naim Iskandar Zahari

## Acknowledgement

The WikiArt dataset can be used only for non-commercial research purposes. <br>
The images in the WikiArt dataset were obtained from WikiArt.org. The authors are neither responsible for the content nor the meaning of these images. <br>
By using the WikiArt dataset, you agree to obey the terms and conditions of WikiArt.org.

## Project Dataset

### WikiArt
WikiArt is a non-profit platform that showcases visual art pieces online from its users globally. According to WikiArt, the platform aims to provide the accessibility of the world’s art to anyone worldwide. The platform features over 250,000 artworks by 3,000 artists. The art pieces are works in museums, universities, and civic buildings in many countries. The platform intends to display the art history of all human civilizations from cave art to modern collections. Most of the art pieces on the platform are not available for public view. <br>
WikiArt maintains an organized database showcasing the artworks. 190 different art styles have been classified and subdivided into the flow of artworks in different time periods and regions. The contributors of the Wikiart platform are meticulous in defining the art styles, where the groupings depict meaningful connections between the artworks and its larger context. More information about the cultural and aesthetic criteria between Western arts and Easterns arts are outlined on the platform. The art styles available include Baroque and Realism in Western Renaissance Art, Impressionism and Expressionism in Modern Art and Minimalism in Contemporary Art, among the 190 art styles. <br>

### Dataset
The dataset used in this classification project are among the publicly available artworks on Kaggle from WikiArt for research purposes. The Kaggle dataset is a diverse collection of 80,000 art images from 1,119 artists in 27 styles. The 80,000 images were used to develop a deep learning conditional StyleGAN2 model for generating art in a research paper published in 2020. The deep learning model from the research includes a ResNet based encoder. <br>
<br>
Indeed, a wide selection of artworks is needed to successfully develop a classification model for this project. However, only a selection of art styles will be used in this classification project. The art styles chosen for this project includes: <br>
1. Baroque
2. Impressionism
3. Post Impressionism
4. Abstract Expressionism
5. Analytical Cubism
6. Cubism
7. Synthetic Cubism
8. Realism
9. New Realism
10. Contemporary Realism
11. Early Renaissance
12. Mannerism Late Renaissance
13. Northern Renaissance
14. High Renaissance
From the selection of art styles, 34,126 artworks are included in the dataset for this classification project.

## Implementation and Results

### VGG-16 (CG)

#### Introduction
VGG-16 is a convolutional neural network architecture that offers simplicity and depth. Developed by the Visual Geometry Group at the University of Oxford, it gained popularity after achieving remarkable performance in the 2014 ImageNet Large Scale Visual Recognition Challenge. Its design consistently uses 3x3 convolutional filters with a stride of 1, coupled with 2x2 max pooling layers with a stride of 2 throughout the entire network. This uniformity simplifies the architecture and allows for a deeper network, which contributes to VGG-16's ability to learn intricate patterns in images. The classification is done through linear transformations. A graphical representation of the architecture [6] can be seen in the image below. <br>
![alt text](http://url/to/img.png)


