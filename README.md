# NetView
NetView is a type of Machine Learning UI that I have been working on to allow beginning data scientists, engineers, and researchers to deploy and analyze deep learning models with more ease.

I have three versions of the interface: v1.0, v1.5, and v2.0. v1.5 is an improved version of v1.0 with a few more functions. v2.0 hasn't changed in functionality, but a UI was attempted to integrate everything. This was done via PyQt5. However, we only got the program to train on the neural network. This is because the button calls can only run NoneType functions, which means we can't get any return values. Also, we can only run one command since that other functions can't be run after the first one since PyQt5 executes directly from the app. 
Hopefully we can get more practice or a better UI library in the future.
