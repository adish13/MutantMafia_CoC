from flask import Flask, jsonify, request, render_template
import urllib.request
from PIL import Image
from deepface import DeepFace
from flask_cors import CORS
import cv2
import wget
import torch
from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# import torchvision
from PIL import Image
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
CORS(app)
run_with_ngrok(app)

final_out = {}

@app.route('/get_url', methods=['POST'])
def get_url():
    # global results, model
    global final_out
    data = request.get_json()
    url = data["image_url"]
    print(url)    
    urllib.request.urlretrieve(url, "image.jpg")
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
    img = Image.open('image.jpg').convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    model = torch.load('model_2.pt')
    with torch.no_grad():
        output = model(img)
    probs = torch.softmax(output, dim=1)
    if(probs[0][1].item() < 0.5):
        final_out = {
            "prob_real": probs[0][1].item(),
            "prob_cartoon": probs[0][0].item(),
            "type": "cartoon_face"
        }
        return jsonify(final_out)
    else:
        out = []
        try:
            results = DeepFace.analyze("image.jpg",actions=["gender"])
            for result in results:
                temp = result['gender']
                gender = result['dominant_gender']
                temp['gender'] = gender
                out.append(temp)
            final_out = {
                "prob_real": probs[0][1].item(),
                "prob_cartoon": probs[0][0].item(),
                "type": "Human_face",
                "gender_probs": out
            }
            return jsonify(final_out)
        except:
            final_out = {
                "prob_real": probs[0][1].item(),
                "prob_cartoon": probs[0][0].item(),
                "type": "No_faces"
            }  
            return jsonify(final_out)

@app.route('/results')
def show_results():
    return jsonify(final_out)

if __name__ == '__main__':
   app.run()

