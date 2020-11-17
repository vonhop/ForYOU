from django.shortcuts import render, get_object_or_404
# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect
from nlp.MixPoet.codes.generator import Generator

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# 视图
'''
在 Django 中，网页和其他内容都是从视图派生而来。
每一个视图表现为一个 Python 函数（或者说方法，如果是在基于类的视图里的话）。
Django 将会根据用户请求的 URL 来选择使用哪个视图
（更准确的说，是根据 URL 中域名之后的部分）。
'''
def index(request):
    return render(request, 'nlp/index.html')

import base64
import numpy as np
from django.views.decorators.csrf import csrf_exempt  # 跨站点验证
from django.http import JsonResponse   # json字符串返回

import io
from nlp.objectDetection.models import *
from nlp.objectDetection.utils.datasets import *
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

model_def = "./nlp/objectDetection/config/yolov3.cfg"
weights_path = "./nlp/objectDetection/weights/yolov3.weights"
class_path = "./nlp/objectDetection/data/coco_chinese.names"
conf_thres = 0.8
nms_thres = 0.4
img_size = 416
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_class_ck(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r",encoding="utf-8")
    names = fp.read().split("\n")[:-1]
    return names

# Set up model
model_image = Darknet(model_def, img_size=img_size).to(device)
if weights_path.endswith(".weights"):
    # Load darknet weights
    model_image.load_darknet_weights(weights_path)
else:
    # Load checkpoint weights
    model_image.load_state_dict(torch.load(weights_path))

# 图像检测模型
model_image.eval()  # Set in evaluation mode
# 诗生成模型
generator = Generator()

classes = load_class_ck(class_path)  # Extracts class labels from file


@csrf_exempt  #用于规避跨站点请求攻击
def image_poem(request):
    default = {"safely executed": False}  # 初始未执行
    #print("image_poem")
    keyword = None

    if request.method == "POST":
        #print(request.POST['poem_length'])
        #print(request.POST['key_word'])
        if request.FILES.get('image') is not None:
            data_temp = request.FILES["image"].read()
            image = np.asarray(bytearray(data_temp), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            retval, buffer_img = cv2.imencode('.jpg', image)  # 在内存中编码为jpg格式
            img64 = base64.b64encode(buffer_img)  # base64编码转换用于网络传输

            img_b64decode = base64.b64decode(img64)  # base64解码
            image_ck = io.BytesIO(img_b64decode)
            input_img = Image.open(image_ck)
            #input_img.show()
            input_imgs = transforms.ToTensor()(input_img)
            # Pad to square resolution
            input_imgs, _ = pad_to_square(input_imgs, 0)
            # Resize
            input_imgs = resize(input_imgs, img_size)
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))
            input_imgs = input_imgs[np.newaxis, :]
            keywordList = []
            # Get detections
            with torch.no_grad():
                detections = model_image(input_imgs)
                detections = non_max_suppression(detections, conf_thres, nms_thres)

            image_tensor = detections[0]
            if image_tensor is not None:
                for feature_tensor in image_tensor:
                    keyword = classes[int(feature_tensor[6])]
                    keywordList.append(keyword)

            if len(keywordList) == 0:
                keywordList.append("云")
                # print('图像检测失败')
            keyword = keywordList[0]

            img64 = str(img64, encoding='utf-8')  # bytes转换为str类型
            default["img64"] = img64  # json封装
        elif request.POST['key_word'] is not None:
            # print(request.POST['key_word'])
            keyword = request.POST['key_word']
        else:
            return HttpResponse("no files for upload!")
    if keyword is not None:
        # print(keyword)
        lines, info = generator.generate_one(keyword, int(request.POST['poem_length']),
                                             int(request.POST['living_experience']), int(request.POST['historical_background']),
                                             20, 1, False)
        poem=''
        for i in range(len(lines)-1):
            poem += lines[i] + '\n'
        poem += lines[len(lines)-1]
        default["poem"] = poem
        print(poem)
    else:
        default["poem"] = "error!"
    return JsonResponse(default)
