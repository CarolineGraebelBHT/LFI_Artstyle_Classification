# Art Style Classification
Authors: Caroline Graebel, Kseniia Holovchenko, Naim Iskandar Zahari

## Acknowledgement

1. The WikiArt dataset can be used only for non-commercial research purposes.
2. The images in the WikiArt dataset were obtained from WikiArt.org. The authors are neither responsible for the content nor the meaning of these images.
3. By using the WikiArt dataset, you agree to obey the terms and conditions of WikiArt.org.

## Project Dataset

### WikiArt
WikiArt is a non-profit platform that showcases visual art pieces online from its users globally. According to WikiArt, the platform aims to provide the accessibility of the world’s art to anyone worldwide. The platform features over 250,000 artworks by 3,000 artists. The art pieces are works in museums, universities, and civic buildings in many countries. The platform intends to display the art history of all human civilizations from cave art to modern collections. Most of the art pieces on the platform are not available for public view. <br>
WikiArt maintains an organized database showcasing the artworks. 190 different art styles have been classified and subdivided into the flow of artworks in different time periods and regions. The contributors of the Wikiart platform are meticulous in defining the art styles, where the groupings depict meaningful connections between the artworks and its larger context. More information about the cultural and aesthetic criteria between Western arts and Eastern arts are outlined on the platform. The art styles available include Baroque and Realism in Western Renaissance Art, Impressionism and Expressionism in Modern Art and Minimalism in Contemporary Art, among the 190 art styles. <br>

### Dataset
The dataset used in this classification project are among the publicly available artworks on Kaggle from WikiArt for research purposes. The Kaggle dataset is a diverse collection of 80,000 art images from 1,119 artists in 27 styles. The 80,000 images were used to develop a deep learning conditional StyleGAN2 model for generating art in a research paper published in 2020. The deep learning model from the research includes a ResNet based encoder. <br>
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

## Implementation and Results

### ResNet

### GoogleNet

### VGG-16 (CG)

#### Introduction
VGG-16 is a convolutional neural network architecture that offers simplicity and depth. Developed by the Visual Geometry Group at the University of Oxford, it gained popularity after achieving remarkable performance in the 2014 ImageNet Large Scale Visual Recognition Challenge. Its design consistently uses 3x3 convolutional filters with a stride of 1, coupled with 2x2 max pooling layers with a stride of 2 throughout the entire network. This uniformity simplifies the architecture and allows for a deeper network, which contributes to VGG-16's ability to learn intricate patterns in images. The classification is done through linear transformations. A graphical representation of the architecture [6] can be seen in the image below. <br>

![VGG-16 model architecture.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/VGG-16%20architecture.png)

One of VGG-16's main strengths is its straightforward architecture, which makes it relatively easy to understand and implement. The consistent use of small convolutional filters enables the network to capture a lot of details in images, contributing to its generally strong performance in image classification and object recognition tasks. <br>
<br>
However, VGG-16 also has its weaknesses. Its depth and large number of parameters (approximately 138 million) can make it computationally expensive and memory-intensive, which is the biggest challenge for trying it for this art style classification project. This leads to long training times compared to other architectures and makes it challenging to deploy on devices with limited resources. Moreover, while its depth contributes to its representational power, it can also make the model prone to overfitting, especially when training on smaller datasets. <br>
For the model, not all of the 14 image folders mentioned at the beginning are used. VGG-16 has been trained on Realism, Impressionism, Baroque, Expressionism, Abstract Expressionism, Cubism and High Renaissance.

#### Preparing the data
Gathering the data from Kaggle, some of the file names were corrupted. There is an encoding issue that transforms letters with accents into unreadable characters. An alternative download from a google drive as suggested by the Discussion tab of the Kaggle page also contained corrupted filenames. However, these letters are in UTF-08 encoding, so it was possible to find out what these characters are meant to look like. These corrupted files are problematic because python can’t access them which in turn leads to image data not being loaded in.

##### Investigating corrupted data files (find_faulty_data_paths.py)
To understand how many paths are corrupted and what this corruption looks like, a script is used that tries to read in the image files and catches the paths if it fails. Looking at the results, some vocals that are common in French, German and Spanish were shown to be the issue, like ä, é, ü, ó etc.. In the patch_filename()-function contained in the script, there’s an atlas of how these signs translate, helped by an UTF-8 cheat sheet for most letters [4]. The faulty signs are transformed into only the base letter without accent to avoid further issues with data imports. This script also helps to verify that all file paths are in order for the model to run.

##### Fixing corrupted data files
When running a Python script to fix the corrupted filenames, it is generally not possible to access the corrupted files at all. Because of this, PowerShell commands have been used to rename the files in badges. A screenshot of the used commands is added below. It shows how the renaming got addressed.

![Screenshot containing the PowerShell commands used to fix the data paths.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/Screenshot%202025-02-05%20160615.png)

Through every art style folder, a batch of names that share one corrupted letter have been renamed. Afterwards, the find_faulty_data_paths.py script is used to verify the files have been successfully cleaned.
This is not the full shell history of the renaming, the shown lines should serve as an example of the approach. After this, all paths were accessible for loading into Python.

##### Getting paths and doing a train-test-split (dataloader.py)
To make the import easier and save space (as the whole dataset is bigger than 30GB), all art style folders that are not relevant to this project got deleted. Because of that, instead of selecting folders to be read, it can just read all files that are there (load_image_paths()). <br>
There’s also a function to do the train-test-split. For this, all paths get shuffled (with a seed) and the first 10000 images get selected. As introduced earlier, VGG-16 is really hardware-intensive and because I had to train the model on CPU. Pytorch has stopped supporting using AMD GPUs with ROCm for model training [5].

![Screenshot containing an oversight of available Pytorch Versions for Download, with ROCm not being available for Windows.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/Screenshot%202025-02-05%20160615.png)

A training on all 41k images on CPU would’ve taken 5 days, calculated from the benchmark of 200 badges / hour in CPU processing power. With 10000 images, it took ~1,5 days. I decided for a 70:30 split per default. The function then returns the finished lists with the paths to the training and testing data.

##### Loading and preparing images (prepare_image_data.py)
Art images come in varying formats and sizes. VGG-16 requires that all images are fed in the same aspect ratio and size. I decided to crop the largest possible square of the image and then resize it to 224x224 pixels, which is the standard size for the VGG-16 architecture. The perks of this are that the images don’t get “squished” into a square when only rescaling them with the cv2.resize()-function, so all objects on the image stay unchanged. The downside is that you lose some information as the picture gets cropped. Since models learn patterns, brushing techniques should be visually consistent over all images, so cropping was the tool of choice. This consistency might also apply to certain motifs, like faces in portraits for example. However, art is very diverse, and therefore this might not play a big role overall. The image arrays also get normalized to help the model converge faster.

##### Saving the label / art style of the image (get_label.py)
To train and test the model, labels are necessary. To save the label for the image, regular expression pattern was used to extract the art style from the file path that belongs to the image. An example for how the image paths would look like this ".././Data/Cubism\\andre-lhote_paysage-au-lambrequin-rose-1914.jpg". A regular expression pattern was created, so that the function always matches the string in between “Data/” and “\\”. The matching pattern looks like this: r"Data/(.*?)\\". <br>
This approach makes it possible to have different seeds for shuffling and differing amounts of data.

#### The Model (VGG16.py)
The VGG-16 model architecture gets captured in the class MyNeuralNetwork. As mentioned in the model introduction, VGG-16 mainly consists of Convolutional layers that use a 3x3-kernel and a padding of 1. ReLu is used as an activation function. At the end of each of the five stages, the results are pooled in the MaxPool2d()-function with a 2x2-kernel and a stride of 2. The classifier uses linear transformations to calculate the probabilities for each art style depending on the input image. <br>
The model parameters are:
* Batch size: 32
* Number of epochs: 20
* Learning rate: 0.001
* Momentum: 0.9
* Loss criterion: Cross Entropy Loss
* Optimizer: Stochastic Gradient Descent <br>

The training and test data are transformed into tensor datasets, with the labels containing the art style being encoded as numbers. For each epoch, the model is evaluated for accuracy and cross entropy loss. The model with the best test accuracy is saved. <br>
The model evaluations over the different epochs are plotted.

![Plot containing the accuracy evaluation for the model training.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/accuracy_VGG-16.png)

The resulting accuracy plot shows that the model succeeds at adapting to the training data but is very stagnant on the testing data, with the testing accuracy fluctuating between ~40% and ~50% accuracy.

#### Data Visualizations / Results

##### Saving predictions (get_predictions.py)
To save predictions instead of just performance, the training and testing data is forwarded to the final model in an adapted pipeline. The architecture is given and the model parameters are loaded in from the project directory. The training and testing data is again prepared like in the model training (VGG16.py). Instead of evaluating performance the predictions are simply saved into a list. For both testing and training data, the labels and the predictions are saved into data frames that can then be imported to csv-files for use in visualizations scripts.

##### Showing the distribution of art styles in the data (piechart.py)
For training and testing data, there is an investigation on how the art styles are distributed. For this, the labels for each dataset are loaded and plotted in a pie chart.

![Plot containing the distribution of art styles in the training data.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/training_pie.png)

![Plot containing the distribution of art styles in the testing data.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/testing_pie.png)

The distribution is very similar in both training and testing data. There are big differences in how many items of each art style are represented though. Images from the High Renaissance and Cubism are very underrepresented (~3.5% and ~5% respectively), while realism and impressionism dominate by making up more than 50% of the whole data.

##### Showing Accuracy by art style (visualize_predictions.py)
To better understand how well the model is able to predict art styles, differences in accuracy on the base of the extracted predictions and labels are plotted (get_predictions.py). For this, the true art style gets reconstructed from the encoded numerical values that the model requires. Furthermore, a new Boolean variable is created that captures whether the prediction of the model was correct by comparing the label to the prediction. After this, a simple bar plot gets created to visualize the accuracy per art style. As a reminder, when training the model, the model adapted well to the training data over the epochs (80% accuracy at the end), however its predictive power for the test data was quite stagnant (~40-50%).

![Plot containing the prediction accuracy by art style for the training set.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/Training_Pred_Acc.png)

![Plot containing the prediction accuracy by art style for the testing set.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/Testing_Pred_Acc.png)

The overall higher accuracy in the training data shows well. Realism and Impressionism gets correctly classified in more than 80% and 70% of cases respectively. The model also performs comparatively strongly for classifying Baroque, even if it isn’t as strongly represented in the data as Realism and Impressionism. For Expressionism, Abstract Expressionism, Cubism and High Renaissance, the performance is much worse. This makes sense, as modern art styles are more diverse in techniques and the High Renaissance and Cubism are very underrepresented in the dataset. The testing predictions mirror the performance of the training predictions in ranking of art styles. Realism, Impressionism and Baroque are still strongest but perform worse on the testing data which is explained by the model not knowing the images it is trying to predict. For Expressionism, Abstract Expressionism, Cubism and High Renaissance the model performs really badly on the testing data. These plots show that the model isn’t able to adapt to the more diverse art styles with the given training data. Both the diversity of modern art, the information loss through cropping and the underrepresentation issues in the dataset might be reasons why this is the case.

# Resources
1. https://www.wikiart.org/
2. https://www.kaggle.com/datasets/steubk/wikiart/data
3. https://archive.org/details/wikiart-stylegan2-conditional-model
4. https://bueltge.de/wp-content/download/wk/utf-8_kodierungen.pdf 
5. https://pytorch.org/get-started/locally/ 
6. https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918 













