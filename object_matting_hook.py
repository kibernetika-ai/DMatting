import io
import logging

import numpy as np
from PIL import Image
from PIL import ImageFilter
from ml_serving.utils import helpers
LOG = logging.getLogger(__name__)
import math


def init_hook(**params):
    LOG.info('Loaded.')

interploations = {
    'NEAREST':Image.NEAREST,
    'BICUBIC':Image.BICUBIC,
    'BILINEAR':Image.BILINEAR,
}
obj_classes = {
    'Person':1
}

def preprocess(inputs, ctx):
    image = inputs.get('inputs')
    if image is None:
        raise RuntimeError('Missing "inputs" key in inputs. Provide an image in "inputs" key')

    ctx.original_image = Image.open(io.BytesIO(image[0]))
    ctx.original_image = ctx.original_image.convert('RGB')
    ratio = 1.0
    w = float(ctx.original_image.size[0])
    h = float(ctx.original_image.size[1])
    if w>h:
        if w>1024:
            ratio = w/1024.0
    else:
        if h>1024:
            ratio = h/1024.0

    if ratio>1:
        ctx.image = ctx.original_image.resize((w/ratio,h/ratio))
    else:
        ctx.image = ctx.original_image
    ctx.np_image = np.array(ctx.image)
    ctx.area_threshold = int(inputs.get('area_threshold', 0))
    ctx.max_objects = int(inputs.get('max_objects', 100))
    ctx.pixel_threshold = float(inputs.get('pixel_threshold', 0.5))
    ctx.object_classes = [obj_classes.get(inputs.get('object_class', ['Person'])[0].decode("utf-8"),1)]
    ctx.effect = inputs.get('effect', ['Remove background'])[0].decode("utf-8")#Remove background,Mask,Blur
    ctx.blur_radius = int(inputs.get('blur_radius', 2))
    ctx.interpolation = interploations[inputs.get('interpolation', ['BILINEAR'])[0].decode("utf-8")]#NEAREST,BICUBIC,BILINEAR
    ctx.matting = inputs.get('matting', ['DEFAULT'])[0].decode("utf-8")#DEFAULT,KNN,NONE
    return {'inputs': [ctx.np_image]}


def kibernetika_matte(img, trimap,pixel_threshold,out_put):
    outputs = helpers.predict_grpc({'image': np.expand_dims(img,0),'mask': np.expand_dims(trimap,0),'in_type':np.array([1],dtype=np.int32),'out_type':np.array([out_put],dtype=np.int32),'pixel_threshold':np.array([int(pixel_threshold*255)],dtype=np.int32)},'deepmatting-0-0-1:9000')
    return outputs['image']


def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return np.stack([gray,gray,gray],axis=2)

def postprocess(outputs, ctx):
    num_detection = int(outputs['num_detections'][0])

    def return_original():
        image_bytes = io.BytesIO()
        ctx.original_image.save(image_bytes, format='PNG')
        outputs['output'] = image_bytes.getvalue()
        return outputs

    if num_detection < 1:
        return return_original()

    width = ctx.np_image.shape[1]
    height = ctx.np_image.shape[0]
    image_area = width * height
    detection_boxes = outputs["detection_boxes"][0][:num_detection]
    detection_boxes = detection_boxes * [height, width, height, width]
    detection_boxes = detection_boxes.astype(np.int32)
    detection_classes = outputs["detection_classes"][0][:num_detection]
    detection_masks = outputs["detection_masks"][0][:num_detection]

    masks = []
    for i in range(num_detection):
        if int(detection_classes[i]) not in ctx.object_classes:
            continue
        mask_image = Image.fromarray(detection_masks[i])
        box = detection_boxes[i]
        mask_image = mask_image.resize((box[3] - box[1], box[2] - box[0]), ctx.interpolation)
        left = max(0,box[1]-50)
        right = min(ctx.np_image.shape[1],box[3]+50)
        upper = max(0,box[0]-50)
        lower = min(ctx.np_image.shape[0],box[2]+50)
        box_mask = np.array(mask_image)
        box_mask = np.pad(box_mask, ((box[0]-upper, lower-box[2]), (box[1]-left, right - box[3])), 'constant')
        area = int(np.sum(np.greater_equal(box_mask, ctx.pixel_threshold).astype(np.int32)))
        if area * 100 / image_area < ctx.area_threshold:
            continue
        masks.append((area, box_mask,[upper,left,lower,right]))

    if len(masks) < 1:
        return return_original()
    masks = sorted(masks, key=lambda row: -row[0])
    total_mask = np.zeros((height, width), np.float32)
    for i in range(min(len(masks), ctx.max_objects)):
        pre_mask = masks[i][1]
        box = masks[i][2]
        left = max(0,box[1])
        right = min(ctx.np_image.shape[1],box[3])
        upper = max(0,box[0])
        lower = min(ctx.np_image.shape[0],box[2])
        if ctx.matting == 'DEFAULT':
            pre_mask[np.less(pre_mask, ctx.pixel_threshold)] = 0
        elif ctx.matting == 'Kibernetika':
            pre_mask = kibernetika_matte(ctx.np_image[upper:lower,left:right,:],np.uint8(pre_mask*255),ctx.pixel_threshold,0)
        elif ctx.matting == 'Testing':
            pre_mask = kibernetika_matte(ctx.np_image[upper:lower,left:right,:],np.uint8(pre_mask*255),ctx.pixel_threshold,1)
            outputs['output'] = pre_mask
            return outputs
        box_mask = np.pad(pre_mask, ((upper, height - lower), (left, width - right)), 'constant')
        total_mask = np.maximum(total_mask,box_mask)
    if total_mask.shape[0] != ctx.original_image.size[1] or total_mask.shape[1] != ctx.original_image.size[0]:
        total_mask = Image.fromarray(np.uint8(total_mask*255))
        total_mask = total_mask.resize((ctx.original_image.size[0],ctx.original_image.size[1]))
    if ctx.effect == 'Remove background':
        original_np_image = np.array(ctx.original_image)
        image = original_np_image.astype(np.float32)
        #image = np.expand_dims(total_mask,2)*image
        total_mask = np.uint8(total_mask*255)
        image = np.dstack((image, total_mask))
        image = Image.fromarray(np.uint8(image))
    elif ctx.effect == "Mask":
        total_mask = total_mask*255
        image = Image.fromarray(np.uint8(total_mask))
    else:
        original_np_image = np.array(ctx.original_image)
        image = original_np_image.astype(np.float32)
        mask = np.expand_dims(total_mask,2)
        foreground = mask*image
        radius = min(max(ctx.blur_radius,2),10)
        if ctx.effect == 'Grey':
            background = rgb2gray(ctx.np_image)
        else:
            background = ctx.image.filter(ImageFilter.GaussianBlur(radius=radius))
        background = (1.0-mask)*np.array(background,dtype=np.float32)
        image = foreground+background
        image = Image.fromarray(np.uint8(image))

    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    outputs['output'] = image_bytes.getvalue()
    return outputs
