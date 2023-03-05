# Mutant Mafia - Real Face detection and gender classification

## Demo website built on replit
This has the documentation for out api and has some frontend for a sample run of our project.
Can be found here : https://cocfrontend.fuushyn.repl.co/

## API hosted at :
Link : https://134f-2409-4040-e84-511f-5060-1259-5544-32a9.in.ngrok.io/get_url

We used flask-ngrok for hosting our API publically.

## Code for API 
`app.py` has the code for the POST api, defined at /get_url, it also uses the models for predicting whether image is cartoon or real, and then correspondingly classifies the gender

## Training the model :
We trained the model using binary classification for real and cartoon images using cross entropy loss, on top of a pre-trained model, essentially finetuning it. The datasets used for cartoon and real faces are : IIIT CFW (Cartoon faces in the wild : https://cvit.iiit.ac.in/research/projects/cvit-projects/cartoonfaces#:~:text=The%20IIIT%2DCFW%20is%20database,images%20of%20100%20public%20figures) and LFW (Labelled faces in the wild : http://vis-www.cs.umass.edu/lfw/). This model is saved in `model_2.pt`
The code for training is in `real_cartoon_training.ipynb`

## Model used for gender prediction:
We used the `deepface` library which is a wrapper around existing state-of-the-art methods for face detection and gender classification. The analyze function defined in this library directly extracts the human faces and predicts their corresponding genders


