import requests
import json
import urllib.request
import numpy as np
from PIL import Image

subscription_key = open('subscription_key').read()
assert subscription_key

face_api_url = 'https://brazilsouth.api.cognitive.microsoft.com/face/v1.0/detect'

image_url = 'https://s3.portalt5.com.br/imagens/familia-adams.jpg?mtime=20171224175642'
image_extension = image_url.split('.')[-1].split('?')[0]

headers = { 'Ocp-Apim-Subscription-Key': subscription_key }
    
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
}

response = requests.post(face_api_url, params=params, headers=headers, json={"url": image_url})
faces = response.json()
print(faces)
image_path = 'xxx.' + image_extension
urllib.request.urlretrieve(image_url, image_path)

im = Image.open(image_path)
mat = np.array(im)

#   descricao           rgb                 traducao        cor
colors = {
    'anger':     [ 231, 76,  60  ],         # raiva     |   vermelho
    'contempt':  [ 211, 84,  0   ],         # desprezo  |   laranja escuro
    'disgust':   [ 155, 89,  182 ],         # aversão   |   roxo
    'fear':      [ 0,   0,   0   ],         # medo      |   preto
    'happiness': [ 46,  204, 113 ],         # alegria   |   verde
    'neutral':   [ 255, 255, 255 ],         # neutro    |   branco
    'sadness':   [ 128, 128, 128 ],         # triste    |   cinza
    'surprise':  [ 241, 196, 15  ],         # surpreso  |   amarelo
}

def getEmotion(emotions):
    values = list( emotions.values() )
    index = np.argmax( np.array(values) )
    return list(emotions.keys())[index]

def getEmotionColor(emotion):
    return np.array( colors[emotion], dtype=np.uint8 )

for face in faces:
    rect = face['faceRectangle']
    col1, col2 = rect['left'], rect['left'] + rect['width']
    row1, row2 = rect['top'], rect['top'] + rect['height']    
    emotions = face['faceAttributes']['emotion']
    emotion = getEmotion(emotions)
    color = getEmotionColor(emotion)
    mat[row1:row2, col1:col2] = color

Image.fromarray(mat).show()