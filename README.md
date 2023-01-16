# -DeepStack-object-decection

This repository provides a pre-trained object detection model that can be used to detect  common objects in dark/night images and videos.
The model, called DeepStack, was trained on the ExDark dataset and is capable of detecting objects such as bicycles, boats, bottles, buses, chairs, cars, cats, cups, dogs, motorbikes, people, and tables.
![image](https://user-images.githubusercontent.com/107410165/212606805-e4d788e5-24d7-41df-aced-7e14b7cb3d05.png)


To use the model, you will first need to install DeepStack AI Server, download the trained custom model from this GitHub release,
and run the server with the custom model. Once the server is running, you can send a POST request to the API endpoint with an image to detect objects.
A sample Python code and image are provided in the repository for reference.

Additionally, this repository also provides instructions on how to discover more custom models and how to train your own model.
To discover more custom models, you can visit the Custom Models sample page on DeepStack's documentation. 
To train your own model, you will need to collect and annotate images of the objects you wish to detect and then train the model using the instructions provided.
