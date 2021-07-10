# LayerVision

### Update: The packages used as part of this project may not be up to date. 

LayerVision is a type of Machine Learning application that I have been working on to allow data scientists, engineers, and researchers to deploy and keep track of their deep learning models with more ease. In its current state, it is only a proof-of-concept for an application of such sort. 

This version of the application is built on a Flask backend and a HTML & CSS frontend. It will incorporate some elements of what is shown in the previous versions of the application. 

## Features

With this application, you are able to:

- Give hyperparameters to model. 
- Edit model hyperparameters.
- Delete redundant models.
- View loss and accuracy of the model with graphs, including:
  * Training Loss
  * Training Accuracy
  * Testing Loss
  * Testing Accuracy
- Feed the set hyperparameters of the model to the python file with ease.

## Installation

To install, you can clone the repository in your specified file on the command prompt:

```bash
git clone https://github.com/axie123/LayerVision.git
```
or download the zip file and unpack it in the folder.

## Usage/Running 

To run the program, you need to go into the location of the repo in you computer either on your Linux or Windows Bash/Shell. Once there, type

```bash
python app.py
```
to run. You can also use Anaconda Prompt. After that, go to http://http://127.0.0.1:5000/.

### Prerequisites

You need the following libraries installed to use this application:

* Flask
* SQLAlchemy (Flask)
* PyTorch

You can just use the requirements file to download the necessary dependencies.
