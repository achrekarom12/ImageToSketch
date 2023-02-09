# Image to sketch using Autoencoders

Autoencoders are a type of neural network architecture used for unsupervised learning. The main purpose of autoencoders is to learn a compact representation (or encoding) of input data, typically for dimensionality reduction or feature extraction.

An autoencoder consists of two parts: an encoder, which maps the input data to a lower-dimensional representation, and a decoder, which maps the lower-dimensional representation back to an approximation of the original input. The goal of training an autoencoder is to minimize the reconstruction error between the original input and its reconstructed output.

During training, the encoder and decoder learn to form a bottleneck that captures the most important information in the input data, while ignoring the noise or redundant information. The bottleneck representation can then be used as input to other machine learning models, or as a preprocessing step for further analysis.

There are various types of autoencoders, including vanilla autoencoders, convolutional autoencoders, recurrent autoencoders, denoising autoencoders, and variational autoencoders, to name a few. Each type of autoencoder is designed to handle specific types of input data and address certain tasks.

<img width="639" alt="image" src="https://user-images.githubusercontent.com/88442486/217734551-af7aaf03-3859-47a9-ba2c-e1184d05ec31.png">

Dataset used: https://www.kaggle.com/datasets/arbazkhan971/cuhk-face-sketch-database-cufs <br>
Reference: https://www.kaggle.com/code/theblackmamba31/photo-to-sketch-using-autoencoder
