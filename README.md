# Art Style Classification
Authors: Caroline Graebel, Kseniia Holovchenko, Naim Iskandar Zahari

## Acknowledgement

1. The WikiArt dataset can be used only for non-commercial research purposes.
2. The images in the WikiArt dataset were obtained from WikiArt.org. The authors are neither responsible for the content nor the meaning of these images.
3. By using the WikiArt dataset, you agree to obey the terms and conditions of WikiArt.org.

## Motivation
Combines art, technology, and machine learning to create a practical tool for automatic style classification.

## Goal
Develop a ML model to classify paintings into distinct art styles using visual features.

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

![Screenshot containing an oversight of available Pytorch Versions for Download, with ROCm not being available for Windows.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/Screenshot%202025-02-11%20124824.png)

A training on all 41k images on CPU would’ve taken 5 days, calculated from the benchmark of 200 badges / hour in CPU processing power. With 10000 images, it took ~1,5 days. I decided for a 70:30 split per default. The function then returns the finished lists with the paths to the training and testing data.

##### Loading and preparing images (prepare_image_data.py)
Art images come in varying formats and sizes. VGG-16 requires that all images are fed in the same aspect ratio and size. I decided to crop the largest possible square of the image and then resize it to 224x224 pixels, which is the standard size for the VGG-16 architecture. The perks of this are that the images don’t get “squished” into a square when only rescaling them with the cv2.resize()-function, so all objects on the image stay unchanged. The downside is that you lose some information as the picture gets cropped. Since models learn patterns, brushing techniques should be visually consistent over all images, so cropping was the tool of choice. This consistency might also apply to certain motifs, like faces in portraits for example. However, art is very diverse, and therefore this might not play a big role overall. The image arrays also get normalized to help the model converge faster.

##### Saving the label / art style of the image (get_label.py)
To train and test the model, labels are necessary. To save the label for the image, regular expression pattern was used to extract the art style from the file path that belongs to the image. An example for how the image paths would look like this ".././Data/Cubism\\andre-lhote_paysage-au-lambrequin-rose-1914.jpg". A regular expression pattern was created, so that the function always matches the string in between “Data/” and “\\”. The matching pattern looks like this: r"Data/(.*?)\\". <br>
This approach makes it possible to have different seeds for shuffling and differing amounts of data.

#### The Model (VGG16.py)
The VGG-16 model architecture gets captured in the class MyNeuralNetwork. As mentioned in the model introduction, VGG-16 mainly consists of Convolutional layers that use a 3x3-kernel and a padding of 1. ReLu is used as an activation function. At the end of each of the five stages, the results are pooled in the MaxPool2d()-function with a 2x2-kernel and a stride of 2. The classifier uses linear transformations to calculate the probabilities for each art style depending on the input image. <br>
The model parameters are:
* not pretrained
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

![Plot containing the prediction accuracy by art style for the training set.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/train%20prediction%20accuracy%20coloured.png)

![Plot containing the prediction accuracy by art style for the testing set.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/test%20prediction%20accuracy%20coloured.png)

The overall higher accuracy in the training data shows well. Realism and Impressionism gets correctly classified in more than 80% and 70% of cases respectively. The model also performs comparatively strongly for classifying Baroque, even if it isn’t as strongly represented in the data as Realism and Impressionism. For Expressionism, Abstract Expressionism, Cubism and High Renaissance, the performance is much worse. Modern art styles are more diverse in techniques and the High Renaissance and Cubism are very underrepresented in the dataset. The testing predictions mirror the performance of the training predictions in ranking of art styles. Realism, Impressionism and Baroque are still strongest but perform worse on the testing data which is explained by the model not knowing the images it is trying to predict. For Expressionism, Abstract Expressionism, Cubism and High Renaissance the model performs really badly on the testing data. These plots show that the model isn’t able to adapt to the more diverse art styles with the given training data. <br>
The accuracy ratings per class are somewhat similar to the distribution of the classes. It will be investigated whether this is due to class imbalance, too little data, or both.

##### Data Preparation for training a second model with balance in art styles (dataloader2.py)
To implement balancing, now paths are loaded for each art style folder respectively. For each art style, 1000 images are selected and split into training and testing data. Afterwards, these seven sub lists then get concatenated into two big training and testing data path lists and shuffled. In comparison to the first model, instead of 10k now there are only 7k images used. The model pipeline used for the data is the same as for the first model.

##### Looking at accuracy by art style again for the second model
The model pipeline hasn't been changed for training the second model. The model had worse accuracy, peaking at only 43%. The development of the training performance again is overall stagnant.

![Plot of accuracy over epochs for the second VGG-16 model with balanced classes.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/accuracy_VGG-16_balanced.png)

The change in accuracy for each class on the testing set is as follows:

![Plot containing the prediction accuracy by art style for the testing set with balanced data.](https://github.com/CarolineGraebelBHT/LFI_Artstyle_Classification/blob/main/Caro/Pictures/Testing_Prediction_balanced.png)

The balancing of the classes had a positive impact on the consistency of the model. Instead of only three classes reaching an accuracy around 60% or higher, now five classes are detected consistently well. However, the overall accuracy has decreased.

##### Discussion
The development of the model was strongly constrained by hardware limitations. The goal was to check out the most recent CNN methods to classify art. Since VGG-16 has so many parameters, a cluster or at least powerful Nvidia GPU is needed to make training times tolerable. This couldn't be provided and therefore the resulting model's capabilities is unsatisfying. The data used was too little and for the first model too unbalanced to achieve good prediction power. <br>
Even though the parameters of the model worked out on a test run with 1000 images, the problems of the model could also be linked to issues with the learning rate that weren't apparent in the hyper parameter testing. Furthermore, it could clearly be shown that the art classes should be balanced when training a model, as the model strongly fits itself to data that is comparatively overrepresented. A successful VGG-16 model needs to be trained with many thousands of images per art style to ensure that the art style is well represented and varied enough for a proper classification by the model. How many images exactly are a good benchmark couldn't be explored due to the hardware constraints. <br>
Art styles that have a very consistent style over different artists like Impressionism and Cubism were shown to be well captured by the model when the classes are balanced, unlike Expressionism and Realism where artists use varying techniques and styles to make their vision come true. This in turn negatively impacts accuracy, so that a better model might be created by just leaving out Realism and Expressionism. <br>
The images have been cropped to achieve a consistent image size for the model. It would be interesting to see if padding them to a square might be a better alternative, even though when making the image smaller, again details would be lost. The cropping was done under the assumption, that generally the most important image parts are placed in or near the center. However, this isn't always a valid assumption, especially for modern art styles.

### ResNet
#### Introduction
ResNet-50 is a deep convolutional neural network (CNN) that belongs to the ResNet (Residual Network) family, which was introduced in 2015 by Microsoft Research. The key innovation of ResNet is the use of residual learning through skip connections, which allows the network to train very deep models without suffering from the vanishing gradient problem. ResNet-50 consists of 50 layers and is composed of bottleneck blocks that help in efficient feature extraction and computational optimization. 
Among various versions of ResNet (ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152), ResNet-50 is widely used due to its balance between accuracy and computational efficiency. Visualization [7] of ResNet-50 architecture is provided in the image below.

![ResNet-50 model architecture](https://raw.githubusercontent.com/CarolineGraebelBHT/LFI_Artstyle_Classification/main/Kseniia/ResNet50_model_architecture.png)

ResNet-50 is a powerful deep learning model widely used in computer vision due to its ability to overcome the vanishing gradient problem with residual connections. These skip connections enable efficient training of deep networks without performance degradation. ResNet-50 achieves high accuracy on benchmark datasets and is commonly used for image classification, object detection, and segmentation, often serving as a backbone for transfer learning.

However, it comes with challenges. Its high computational cost requires powerful hardware for efficient training and inference. On smaller datasets, it may overfit without proper fine-tuning. Additionally, it has longer inference times compared to lightweight models like MobileNet, making it less suitable for real-time applications. Effective hyperparameter tuning is essential for optimal performance.
Despite these limitations, ResNet-50 remains a powerful, versatile model that serves as the foundation for many deep learning applications

#### Preparing the Data
The data preparation process  for ResNet-50 followed the same approach as described in the VGG-16 Preparing Data section. 

#### Fixing Corrupted Filenames (path_fix.py, img_path.py)
In addition to the approach used in the VGG-16 Preparing Data section, the script path_fix.py was created to automate the process of fixing corrupted filenames. This script addresses encoding issues that prevent Python from correctly loading some image files. It systematically scans all image filenames in the dataset, identifies corrupted characters, and renames the files to ensure they are accessible during training.

img_path.py file is designed to detect corrupt image files within a dataset by using OpenCV to check if each image is readable. It extracts file paths from the dataset, including handling ImageFolder and Subset datasets, and uses multithreading to speed up the validation process. The script helps ensure that all images can be correctly loaded before training, preventing errors during model execution.

#### Data Preprocessing and Splitting (dataload.py)
Before training the model, the dataset undergoes a series of preprocessing steps to ensure consistency and improve generalization. The dataset is loaded using torchvision.datasets.ImageFolder, which organizes images based on their directory structure. 
##### Data Splitting
To train and evaluate the model effectively, the dataset is divided into three subsets:
- **Training Set (70%):** Used to update the model weights.
- **Validation Set (15%):** Used to monitor model performance and tune hyperparameters.
- **Test Set (15%):** Used for final model evaluation.

**Total images:** 47186
**Train:** 33030, **Validation:** 7077, **Test:** 7079.

Data augmentation techniques such as random cropping, horizontal flipping, and rotation were used to improve generalization. The images were normalized to a mean of 0.5 and a standard deviation of 0.5. The dataset is split using random_split() from torch.utils.data, ensuring a balanced and unbiased distribution across classes. However, If certain art styles have significantly fewer images, they may be underrepresented (or missing) in validation or test sets.

For cases where rapid prototyping or debugging is needed, a Quick Test Mode is implemented:
- Only 3,000 images from the training set and 500 from the validation set are used.
- This reduces training time while allowing the model to be tested on a smaller subset.
##### Data Loading:
The processed data is wrapped into DataLoader objects, which handle efficient batch loading:
- train_loader: Loads shuffled training data for stochastic gradient descent (SGD).
- val_loader: Loads validation data without shuffling.
- test_loader: Loads test data for final evaluation.
These preprocessing and splitting techniques ensure an efficient training pipeline while maintaining data diversity and preventing overfitting.

#### System Setup
The ResNet-50 model was trained using PyTorch with CUDA 12.8, leveraging GPU acceleration to significantly speed up the training process. Training a deep learning model like ResNet-50 on a CPU would be extremely slow due to the high computational requirements, especially when working with large datasets like this (47186 images). By utilizing a GPU, matrix operations and tensor computations are parallelized, allowing for faster backpropagation and weight updates.

#### Model ResNet-50 (ResNet.py)
The ResNet-50 model was implemented from scratch using PyTorch. As was mentioned, the key innovation of ResNet is its residual connections, which help overcome the vanishing gradient problem by allowing gradients to flow through the network more efficiently. This enables training deeper networks without performance degradation. 
The architecture of ResNet-50 includes the following key components:
1. **Initial Convolutional Layer:**
   - A 7×7 convolution with 64 filters is applied to the input images.
   - A Batch Normalization (BN) layer stabilizes training.
   - A Max Pooling layer reduces the spatial dimensions before feeding into deeper layers.
2. **Residual Bottleneck Blocks:**
   - Unlike traditional convolutional layers, ResNet-50 uses Bottleneck Blocks to improve efficiency.
   - Each block consists of:
      - 1×1 Convolution (reduces dimensionality)
      - 3×3 Convolution (extracts features)
      - 1×1 Convolution (restores dimensionality)
   - Skip connections add the input (residual) directly to the output, helping to preserve important information and ensure smooth gradient flow.
3. **Layer Structure:**
   - The network consists of 4 main stages:
      - Layer 1: 3 Bottleneck Blocks
      - Layer 2: 4 Bottleneck Blocks
      - Layer 3: 6 Bottleneck Blocks
      - Layer 4: 3 Bottleneck Blocks
   - These layers gradually increase the number of channels, making feature extraction more powerful at deeper levels.
4. **Global Average Pooling & Fully Connected Layer:**
   - After the final convolutional layers, an Adaptive Average Pooling layer reduces the feature maps to a 1×1 spatial size.
   - A fully connected (FC) layer with 2048 input features maps the extracted features to the final number of classes (14 different art styles).

##### Training and Implementation Considerations
- Batch Normalization (BN) was used in every layer to help stabilize gradients and improve convergence.
- Skip connections (shortcuts) were implemented to improve training stability and reduce the risk of vanishing gradients.
- The final fully connected layer was adjusted to match the number of art styles (14 classes) in the dataset.

#### Hyperparameters:
- **Batch Size:** 64
- **Epochs:** 50
- **Learning Rate:** 0.001 (decayed by a factor of 0.5 every 10 epochs)
- **Optimizer:** Stochastic Gradient Descent (SGD) with Momentum = 0.9
- **Loss Function:** CrossEntropyLoss
- **Early Stopping:** Stop training if validation loss does not improve for 10 consecutive epochs.

The model is evaluated for accuracy and cross-entropy loss at each epoch. The version that achieves the highest validation accuracy is saved as the best model. The validation and final test accuracy, as well as the validation and final test loss, are visualized in the plots below.

![Validation and test accuracy](https://raw.githubusercontent.com/CarolineGraebelBHT/LFI_Artstyle_Classification/main/Kseniia/results/val_test_accuracy_plot.png) 

![Validation and test loss](https://raw.githubusercontent.com/CarolineGraebelBHT/LFI_Artstyle_Classification/main/Kseniia/results/val_test_loss_plot.png) 

#### Training and Evaluating the ResNet-50 Model (main.py)
The main.py script is the core of the project, responsible for training, validating, testing, and visualizing predictions using a ResNet-50 deep learning model for art style classification. The key steps in this script include loading the dataset, training the model, evaluating its performance, saving the best model, and visualizing predictions. 

A function predict_multiple_images selects 6 random test images, predicts their art style, and compares the predicted label to the real label. The predictions are saved as "results/test_predictions.png" and displayed. The visualization of six random test images, as implemented in main.py, is also available separately in the visualize_prediction.py file for independent evaluation and testing. 

This script automates the end-to-end deep learning pipeline, including data loading, model training, evaluation, and visualization. It ensures that the best-performing model is saved and used for further inference. By leveraging CUDA, the script allows efficient training, which would otherwise be computationally expensive on a CPU. The final trained model can be used to classify new paintings into their respective art styles.

#### Results
#### Training Performance:
- **Total Training Time:** 631 minutes (≈10.5 hours)
- **Best Model Epoch:** 31 (Validation Accuracy = 56.65%)
- **Final Test Accuracy:** 57.41%
- **Final Test Loss:** 1.2467

The ResNet-50 model was trained for 41 epochs before early stopping was triggered due to no significant improvement in validation loss. The best model was saved at epoch 31, achieving a validation accuracy of 56.65% and a validation loss of 1.2473. The final test accuracy reached 57.41%, demonstrating the model's ability to generalize to unseen data.

The training process lasted 631 minutes (10.5 hours), utilizing CUDA 12.8 for acceleration. The accuracy and loss trends across epochs indicate a steady improvement, with occasional fluctuations. The final model's predictions on test images show a mix of correct and incorrect classifications, highlighting strengths and areas for potential improvement.

The visualization of predictions is shown below.

![The visualization of predictions](https://raw.githubusercontent.com/CarolineGraebelBHT/LFI_Artstyle_Classification/main/Kseniia/results/random_predictions.png) 

##### Confusion Matrix Analysis (visual_confusion_matrix.py)
The confusion matrix  in visual_confusion_matrix.py visualizes the model’s classification performance, highlighting correct predictions (diagonal) and misclassifications (off-diagonal). Since class sizes vary, the matrix is normalized to show percentages per class for better interpretability. This helps identify frequently confused styles and potential improvements like data balancing or augmentation. The matrix is shown below.
![Confusion Matrix](https://raw.githubusercontent.com/CarolineGraebelBHT/LFI_Artstyle_Classification/main/Kseniia/results/confusion_matrix.png) 

##### Class-Wise Accuracy Analysis(visual_class_wise_accuracy.py)
The class-wise accuracy plot in visual_class_wise_accuracy.py provides insight into how well the model performs across different art styles. Some classes achieve higher accuracy, while others may suffer due to dataset imbalance or similarities between styles. This analysis helps identify areas for improvement, such as data balancing or additional fine-tuning. The bar plot is shown below.

![Confusion Matrix](https://raw.githubusercontent.com/CarolineGraebelBHT/LFI_Artstyle_Classification/main/Kseniia/results/class_wise_accuracy.png) 

While the results indicate the model's ability to classify art styles, further optimizations such as data augmentation, hyperparameter tuning, or using a larger dataset could enhance its performance. Additionally, instead of randomly splitting the dataset, organizing the data into train, validation, and test sets based on folders/classes could ensure a more balanced and structured split, potentially improving generalization. Moreover, addressing class imbalance by adding more images to underrepresented classes could help the model learn more evenly across all categories. 
Overall, the model provides a solid foundation for automated art style classification.


### GoogLeNet

GoogLeNet, introduced in the 2014 ILSVRC competition, is a deep convolutional neural network (CNN) known for its **Inception modules**, which improve computational efficiency by using multiple filter sizes (1x1, 3x3, and 5x5) in parallel. It also employs **1x1 convolutions** to reduce dimensionality before applying larger convolutions, significantly lowering computational cost. 
Beyond image classification, GoogLeNet has been adapted for various tasks, including object detection (e.g., Faster R-CNN with Inception), medical imaging analysis, and scene recognition. 
Using a **pretrained model** when training a classification model with the GoogLeNet architecture provides several benefits:
1. Faster Training & Convergence
- A pretrained GoogLeNet model has already been trained on a large dataset (e.g., ImageNet with millions of images), meaning it has learned general visual features such as edges, textures, and object shapes.
- Start with a model that already knows useful representations, requiring fewer epochs to converge.
2. Better Performance with Limited Data
- A pretrained GoogLeNet model transfers learned knowledge, making it more robust when training on a small dataset.
3. Feature Extraction vs. Fine-Tuning
- Feature Extraction: Freeze most layers and only train the final classification layer. This is useful if your dataset is small and similar to ImageNet categories.
- Fine-Tuning: Unfreeze some deeper layers and train them on the dataset. This adapts the model to new features, which is useful if the dataset is very different from ImageNet.
4. Computational Efficiency
- Training a deep model like GoogLeNet from scratch requires significant computational power. Using a pretrained model saves time and reduces the need for high-end GPUs.

**When To Use a Pretrained GoogLeNet Model?**
- Dataset is **small** or lacks diversity.
- To achieve **high accuracy quickly**.
- **Save compute resources** and avoid long training times.
- Task is similar to ImageNet (e.g., classifying common objects).

#### Implementing the Model

**Preparing the Data (model_googlenet.ipynb)**
The dataset for classification would be taken from the listed art styles mentioned above. The folder names will be the 14 classes for GoogLeNet to train the classification model.

The dimension of the images vary a lot and a standard image size is needed to effectively train the classification model. Initializing a transformation to resize the images to 128x128 using torchvision.transforms() from the torchvision module.

![Transforming the images](https://raw.githubusercontent.com/CarolineGraebelBHT/LFI_Artstyle_Classification/main/Naim/googlenet_transformimages.png)

The transforms.Compose() function chains multiple image transformations in a pipeline. The mean and std parameter in transforms.Normalize() normalizes the pixel value and are applied to each RGB channel. This transformation method helps the model to train faster and to generalize better around zero mean. 

Next, the dataset is split randomly into a training set and validation set for the model training and model evaluation. A 70:30 split has been configured for the training set and validation set respectively. 

**Loading GoogLeNet**
The GoogLeNet model is installed with the torchvision package and then loaded from the torchvision.models module. An advantage included in the GoogLeNet model is the parameter to train a classification model with the pretrained model with ImageNet. Training the model with the pretrained model with ImageNet provides several advantage, namely:
- Better model performance with limited dataset
- Efficient computation as using a pretrained model saves time and reduces CPU usage

**Training the Model**

Implementing the GoogLeNet architecture with the dataset is a straightforward process. The parameters for the model are similar to conventional Convolutional Neural Network architectures, with the exception of the option to use the pretrained model for training. In particular, the parameters include **learning rate**, and **epochs**. 

The parameters for the model training are as follows:
1. Using pretrained model: ✅
2. Epochs: 10
3. Learning rate: 0.001
4. Batch size: 64
5. Loss function: Cross Entropy Loss
6. Optimizer: Adam

**Model Performance**

The model performance in the dataset is satisfactory. The model accuracy and loss progressively improve over epochs. The accuracy of the model in the training set is at **90.6%** from the last epoch and the accuracy in the validation set averages at **67.1%** on the fourth epoch.

![Transforming the images](https://raw.githubusercontent.com/CarolineGraebelBHT/LFI_Artstyle_Classification/main/Naim/googlenet_modelaccuracy.png)

![Transforming the images](https://raw.githubusercontent.com/CarolineGraebelBHT/LFI_Artstyle_Classification/main/Naim/googlenet_modelclassaccuracy.png)

# Resources
1. https://www.wikiart.org/
2. https://www.kaggle.com/datasets/steubk/wikiart/data
3. https://archive.org/details/wikiart-stylegan2-conditional-model
4. https://bueltge.de/wp-content/download/wk/utf-8_kodierungen.pdf 
5. https://pytorch.org/get-started/locally/ 
6. https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918
7. https://www.geeksforgeeks.org/understanding-googlenet-model-cnn-architecture/













