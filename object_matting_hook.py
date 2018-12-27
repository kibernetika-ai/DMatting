import io
import logging

import numpy as np
from PIL import Image
from PIL import ImageFilter
import sklearn.neighbors
import scipy.sparse
from mlboardclient.api import client
from ml_serving.utils import helpers
import  os
LOG = logging.getLogger(__name__)


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

    image = Image.open(io.BytesIO(image[0]))
    image = image.convert('RGB')
    np_image = np.array(image)
    inputs['inputs'] = None
    logging.info('Inputs: {}'.format(inputs))
    ctx.image = image
    ctx.np_image = np_image
    ctx.area_threshold = int(inputs.get('area_threshold', 0))
    ctx.max_objects = int(inputs.get('max_objects', 100))
    ctx.pixel_threshold = float(inputs.get('pixel_threshold', 0.5))
    ctx.object_classes = [obj_classes.get(inputs.get('object_class', ['Person'])[0].decode("utf-8"),1)]
    ctx.effect = inputs.get('effect', ['Remove background'])[0].decode("utf-8")#Remove background,Mask,Blur
    ctx.blur_radius = int(inputs.get('blur_radius', 2))
    ctx.interpolation = interploations[inputs.get('interpolation', ['BILINEAR'])[0].decode("utf-8")]#NEAREST,BICUBIC,BILINEAR
    ctx.matting = inputs.get('matting', ['DEFAULT'])[0].decode("utf-8")#DEFAULT,KNN,NONE
    return {'inputs': [np_image]}

def kibernetika_matte(img, trimap):
    outputs = helpers.predict_grpc({'image': np.expand_dims(img,0),'mask': np.expand_dims(trimap,0),'in_type':np.array([1],dtype=np.int32)},'deepmatting-0-0-1:9000')
    return outputs['image']

def knn_matte(img, trimap, mylambda=100):
    [m, n, c] = img.shape
    img, trimap = img/255.0, trimap/255.0
    trimap = np.expand_dims(trimap,2)
    trimap = np.concatenate((trimap,trimap,trimap),axis=2)
    foreground = (trimap > 0.99).astype(int)
    background = (trimap < 0.01).astype(int)
    all_constraints = foreground + background
    a, b = np.unravel_index(np.arange(m*n), (m, n))
    feature_vec = np.append(np.transpose(img.reshape(m*n,c)), [ a, b]/np.sqrt(m*m + n*n), axis=0).T
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=10, n_jobs=4).fit(feature_vec)
    knns = nbrs.kneighbors(feature_vec)[1]

    row_inds = np.repeat(np.arange(m*n), 10)
    col_inds = knns.reshape(m*n*10)
    vals = 1 - np.linalg.norm(feature_vec[row_inds] - feature_vec[col_inds], axis=1)/(c+2)
    A = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)),shape=(m*n, m*n))

    D_script = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    L = D_script-A
    D = scipy.sparse.diags(np.ravel(all_constraints[:,:, 0]))
    v = np.ravel(foreground[:,:,0])
    c = 2*mylambda*np.transpose(v)
    H = 2*(L + mylambda*D)
    try:
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(H, c), 0), 1).reshape(m, n)
    except Warning:
        x = scipy.sparse.linalg.lsqr(H, c)
        alpha = np.minimum(np.maximum(x[0], 0), 1).reshape(m, n)
    return alpha

def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return np.stack([gray,gray,gray],axis=2)

def postprocess(outputs, ctx):
    num_detection = int(outputs['num_detections'][0])

    def return_original():
        image_bytes = io.BytesIO()
        ctx.image.save(image_bytes, format='PNG')
        outputs['output'] = image_bytes.getvalue()
        return outputs

    if num_detection < 1:
        return return_original()

    width = ctx.image.size[0]
    height = ctx.image.size[1]
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
        box_mask = np.array(mask_image)
        box_mask = np.pad(box_mask, ((box[0], height - box[2]), (box[1], width - box[3])), 'constant')
        area = int(np.sum(np.greater_equal(box_mask, ctx.pixel_threshold).astype(np.int32)))
        if area * 100 / image_area < ctx.area_threshold:
            continue
        masks.append((area, box_mask))

    if len(masks) < 1:
        return return_original()
    masks = sorted(masks, key=lambda row: -row[0])
    total_mask = np.zeros((height, width), np.float32)
    for i in range(min(len(masks), ctx.max_objects)):
        total_mask = np.maximum(total_mask,masks[i][1])
    if ctx.matting == 'KNN':
        total_mask[np.less(total_mask, ctx.pixel_threshold)]=0
        total_mask = knn_matte(ctx.np_image,total_mask*255)
    elif ctx.matting == 'DEFAULT':
        total_mask[np.less(total_mask, ctx.pixel_threshold)]=0
    elif ctx.matting == 'Kibernetika':
        total_mask = kibernetika_matte(ctx.np_image,np.uint8(total_mask*255))
        total_mask = total_mask/255

    if ctx.effect == 'Remove background':
        image = ctx.np_image.astype(np.float32)
        image = np.expand_dims(total_mask,2)*image
        total_mask = np.uint8(total_mask*255)
        image = np.dstack((image, total_mask))
        image = Image.fromarray(np.uint8(image))
    elif ctx.effect == "Mask":
        total_mask = total_mask*255
        image = Image.fromarray(np.uint8(total_mask))
    else:
        image = ctx.np_image.astype(np.float32)
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
