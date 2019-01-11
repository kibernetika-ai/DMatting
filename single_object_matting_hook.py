import io
import logging

import numpy as np
from PIL import Image
from PIL import ImageFilter
from scipy import ndimage
import cv2 as cv

LOG = logging.getLogger(__name__)


def init_hook(**params):
    LOG.info('Loaded.')


interploations = {
    'NEAREST': Image.NEAREST,
    'BICUBIC': Image.BICUBIC,
    'BILINEAR': Image.BILINEAR,
}
obj_classes = {
    'Person': 1
}


def preprocess_objects(inputs, ctx):
    image = inputs.get('inputs')
    if image is None:
        raise RuntimeError('Missing "inputs" key in inputs. Provide an image in "inputs" key')

    ctx.original_image = Image.open(io.BytesIO(image[0]))
    ctx.original_image = ctx.original_image.convert('RGB')
    ratio = 1.0
    w = float(ctx.original_image.size[0])
    h = float(ctx.original_image.size[1])
    if w > h:
        if w > 1024:
            ratio = w / 1024.0
    else:
        if h > 1024:
            ratio = h / 1024.0

    if ratio > 1:
        image = ctx.original_image.resize((int(w / ratio), int(h / ratio)))
    else:
        image = ctx.original_image
    if not bool(inputs.get('return_origin_size', False)):
        ctx.original_image = image
    ctx.process_np_image = np.array(image)
    ctx.original_np_image = np.array(ctx.original_image)
    ctx.area_threshold = int(inputs.get('area_threshold', 0))
    ctx.max_objects = int(inputs.get('max_objects', 100))
    ctx.pixel_threshold = int(float(inputs.get('pixel_threshold', 0.5)) * 255)
    ctx.object_classes = [obj_classes.get(inputs.get('object_class', ['Person'])[0].decode("utf-8"), 1)]
    ctx.effect = inputs.get('effect', ['Remove background'])[0].decode("utf-8")  # Remove background,Mask,Blur
    ctx.blur_radius = int(inputs.get('blur_radius', 2))
    ctx.interpolation = interploations[
        inputs.get('interpolation', ['BILINEAR'])[0].decode("utf-8")]  # NEAREST,BICUBIC,BILINEAR
    ctx.matting = inputs.get('matting', ['DEFAULT'])[0].decode("utf-8")  # DEFAULT,KNN,NONE
    return {'inputs': [ctx.process_np_image]}


def postprocess_objects(outputs, ctx):
    def return_original():
        return {}

    num_detection = int(outputs['num_detections'][0])
    if num_detection < 1:
        return return_original()
    process_width = ctx.process_np_image.shape[1]
    process_height = ctx.process_np_image.shape[0]

    image_area = process_width * process_height
    detection_boxes = outputs["detection_boxes"][0][:num_detection]
    detection_boxes = detection_boxes * [process_height, process_width, process_height, process_width]
    detection_boxes = detection_boxes.astype(np.int32)
    detection_classes = outputs["detection_classes"][0][:num_detection]
    detection_masks = outputs["detection_masks"][0][:num_detection]

    masks = []
    for i in range(num_detection):
        if int(detection_classes[i]) not in ctx.object_classes:
            continue
        box = detection_boxes[i]
        mask_image = cv.resize(detection_masks[i], (box[3] - box[1], box[2] - box[0]), interpolation=cv.INTER_LINEAR)
        left = max(0, box[1] - 50)
        right = min(process_width, box[3] + 50)
        upper = max(0, box[0] - 50)
        lower = min(process_height, box[2] + 50)
        box_mask = np.pad(mask_image, ((box[0] - upper, lower - box[2]), (box[1] - left, right - box[3])), 'constant')
        area = int(np.sum(np.greater_equal(box_mask, ctx.pixel_threshold).astype(np.int32)))
        if area * 100 / image_area < ctx.area_threshold:
            continue
        masks.append((area, box_mask, [upper, left, lower, right]))

    if len(masks) < 1:
        return return_original()
    masks = sorted(masks, key=lambda row: -row[0])
    total_mask = np.zeros((process_height, process_width), np.float32)
    min_left = process_width
    min_upper = process_height
    max_right = 0
    max_lower = 0
    for i in range(min(len(masks), ctx.max_objects)):
        pre_mask = masks[i][1]
        box = masks[i][2]
        left = max(0, box[1])
        right = min(process_width, box[3])
        upper = max(0, box[0])
        lower = min(process_height, box[2])
        box_mask = np.pad(pre_mask, ((upper, process_height - lower), (left, process_width - right)), 'constant')
        total_mask = np.maximum(total_mask, box_mask)
        if left < min_left:
            min_left = left
        if right > max_right:
            max_right = right
        if upper < min_upper:
            min_upper = upper
        if lower > max_lower:
            max_lower = lower
    return {'mask': np.uint8(total_mask[min_upper:max_lower, min_left:max_right] * 255),
            'box': (min_upper, min_left, max_lower, max_right)}


def preprocess_mask(inputs, ctx):
    if len(inputs) < 1:
        return {'ml-serving-ignore': True}
    mask = inputs['mask']
    box = inputs['box']
    ctx.mask_box = box
    if len(mask.shape) > 2:
        logging.warning('Mask shape is {}'.format(mask.shape))
        mask = mask[:, :, 0]
    if ctx.matting == 'DEFAULT':
        mask[np.less(mask, ctx.pixel_threshold)] = 0
        return {'ml-serving-ignore': True,'mask':mask}
    image = cv.resize(ctx.process_np_image[box[0]:box[2], box[1]:box[3], :], (320, 320), interpolation=cv.INTER_LINEAR)
    mask = cv.resize(mask, (320, 320), interpolation=cv.INTER_LINEAR)
    mask[np.less_equal(mask, ctx.pixel_threshold)] = 0
    mask[np.greater(mask, ctx.pixel_threshold)] = 255
    input_trimap = generate_trimap(mask)
    input_trimap = np.expand_dims(input_trimap.astype(np.float32), 2)
    image = image.astype(np.float32)
    input_image = image - g_mean
    return {'input': [input_image], 'trimap': [input_trimap]}


def postprocess_mask(outputs, ctx):
    if len(outputs) < 1:
        image_bytes = io.BytesIO()
        ctx.original_image.save(image_bytes, format='PNG')
        return {'output': image_bytes.getvalue()}
    mask = outputs.get('mask',None)
    if mask is None:
        mask = outputs['output'][0] * 255
        mask = np.reshape(mask, (320, 320))
        mask = np.clip(mask, 0, 255)
        mask = mask.astype(np.uint8)
        mask = cv.resize(mask, (ctx.mask_box[3] - ctx.mask_box[1], ctx.mask_box[2] - ctx.mask_box[0]),
                     interpolation=cv.INTER_LINEAR)

    mask = mask.astype(np.float32) / 255
    process_width = ctx.process_np_image.shape[1]
    process_height = ctx.process_np_image.shape[0]
    mask = np.pad(mask,
                  ((ctx.mask_box[0], process_height - ctx.mask_box[2]), (ctx.mask_box[1], process_width - ctx.mask_box[3])),
                  'constant')
    if mask.shape != ctx.original_np_image.shape:
        mask = cv.resize(mask, (ctx.original_np_image.shape[1], ctx.original_np_image.shape[0]),
                         interpolation=cv.INTER_LINEAR)
    if ctx.effect == 'Remove background':
        image = ctx.original_np_image.astype(np.float32)
        mask = np.uint8(mask * 255)
        image = np.dstack((image, mask))
        image = Image.fromarray(np.uint8(image))
    elif ctx.effect == "Mask":
        mask = mask * 255
        image = Image.fromarray(np.uint8(mask))
    else:
        image = ctx.original_np_image.astype(np.float32)
        mask = np.expand_dims(mask, 2)
        foreground = mask * image
        radius = min(max(ctx.blur_radius, 2), 10)
        if ctx.effect == 'Grey':
            background = rgb2gray(ctx.original_np_image)
        else:
            background = ctx.original_image.filter(ImageFilter.GaussianBlur(radius=radius))
        background = (1.0 - mask) * np.array(background, dtype=np.float32)
        image = foreground + background
        image = Image.fromarray(np.uint8(image))

    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    return {'output': image_bytes.getvalue()}


def rgb2gray(rgb):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return np.stack([gray, gray, gray], axis=2)


g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])
unknown_code = 128


def generate_trimap(alpha):
    trimap = np.copy(alpha)
    k_size = 20
    trimap[np.where((ndimage.grey_dilation(alpha[:, :], size=(k_size, k_size)) - ndimage.grey_erosion(alpha[:, :],
                                                                                                      size=(k_size,
                                                                                                            k_size))) != 0)] = unknown_code
    return trimap

preprocess = [preprocess_objects, preprocess_mask]
postprocess = [postprocess_objects, postprocess_mask]