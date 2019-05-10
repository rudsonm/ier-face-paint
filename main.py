import io
import requests
import json
import urllib.request
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

subscription_key = open('subscription_key').read()
assert subscription_key

face_api_url = 'https://brazilsouth.api.cognitive.microsoft.com/face/v1.0/detect'

# para fazer upload de imagem rápido: https://imgbb.com/

image_url = 'http://4.bp.blogspot.com/--u7Fho2C3K8/UhIKepmpzqI/AAAAAAAAC7A/lgirTyldrHc/s1600/flo.jpg'

image_path = image_url.split('/')[-1].split('?')[0]
urllib.request.urlretrieve(image_url, image_path)

im = Image.open(image_path)
mat = np.array(im)

headers = { 'Ocp-Apim-Subscription-Key': subscription_key }
    
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
}

response = requests.post(face_api_url, params=params, headers=headers, json={"url": image_url})
faces = response.json()
print('faces:', faces)

#   descricao           rgb              traducao     cor
colors = {
    'anger':     [ 213, 0, 0 ],         # raiva    | vermelho
    'contempt':  [ 255, 109, 0 ],       # desprezo | laranja escuro
    'disgust':   [ 98, 0, 234 ],        # nojo     | roxo
    'fear':      [ 0, 77, 64 ],         # medo     | preto
    'happiness': [ 255, 234, 0 ],       # alegria  | verde
    'neutral':   [ 255, 255, 255 ],     # neutro   | branco
    'sadness':   [ 41, 98, 255 ],       # triste   | cinza
    'surprise':  [ 0, 229, 255  ],      # surpreso | amarelo
}

def getEmotion(emotions):
    values = list( emotions.values() )
    index = np.argmax( np.array(values) )
    return list(emotions.keys())[index]

def getEmotionColor(emotion):
    return np.array( colors[emotion], dtype=np.uint8 )

grayIntensity = np.array([ .21, .72, .07 ])

def euclidianDistance(p1, p2):
    return math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

for face in faces:
    rect = face['faceRectangle']
    col1, col2 = rect['left'], rect['left'] + rect['width']
    row1, row2 = rect['top'], rect['top'] + rect['height']    
    emotions = face['faceAttributes']['emotion']
    emotion = getEmotion(emotions)
    color = getEmotionColor(emotion)
    # increased faces square
    row1 -= 10
    row2 += 10
    col1 -= 10
    col2 += 10
    rect = mat[row1:row2, col1:col2]
    center = ( (col2-col1)/2, (row2-row1)/2 )
    maxDistance = min(
        euclidianDistance( center, (col2-col1, center[1]) ),
        euclidianDistance( center, (center[0], row2-row1) ),
    )
    for i, j in np.ndindex(rect.shape[:-1]):
        pix = rect[i][j]
        gray = (pix * grayIntensity).sum() / 255.
        colored = np.array(color * gray, dtype=np.uint8)
        distance = euclidianDistance( center, (j,i) )
        prcDist = min(1, max(0, distance / maxDistance - 0.2))
        finalColor = pix * prcDist + colored * (1-prcDist)
        rect[i][j] = np.array(finalColor, dtype=np.uint8)
    # square on faces
    # b = border = 2
    # mat[row1:row1+b, col1:col2  ] = color
    # mat[row2:row2+b, col1:col2  ] = color
    # mat[row1:row2,   col1:col1+b] = color
    # mat[row1:row2,   col2:col2+b] = color

Image.fromarray(mat).save('emotionfull.png')
plt.imshow(mat)
plt.show()
