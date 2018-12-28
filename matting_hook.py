import io
import logging

import numpy as np
from PIL import Image
from scipy import ndimage

LOG = logging.getLogger(__name__)


def init_hook(**params):
    LOG.info('Loaded.')

interploations = {
    0:Image.BILINEAR,
    1:Image.BILINEAR,
    2:Image.BILINEAR,
    3:Image.BILINEAR,
}

g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])
unknown_code = 128

def generate_trimap(alpha):
    trimap = np.copy(alpha)
    k_size = 10
    trimap[np.where((ndimage.grey_dilation(alpha[:, :], size=(k_size, k_size)) - ndimage.grey_erosion(alpha[:, :],
                                                                                                      size=(k_size,
                                                                                                            k_size))) != 0)] = unknown_code
    return trimap

def preprocess(inputs, ctx):
    in_type = 'image'
    if inputs.get('in_type',None) is not None:
        in_type = 'np'
    ctx.in_type = in_type
    if in_type == 'np':
        image = inputs['image'][0]
        image = Image.fromarray(image)
        mask = inputs['mask'][0]
        mask = Image.fromarray(mask)
    else:
        image = inputs.get('image')
        if image is None:
            raise RuntimeError('Missing "image" key in inputs. Provide an image in "image" key')
        mask = inputs.get('mask')
        if mask is None:
            raise RuntimeError('Missing "mask" key in inputs. Provide an mask in "mask" key')
        image = Image.open(io.BytesIO(image[0]))
        image = image.convert('RGB')
        mask = Image.open(io.BytesIO(mask[0]))

    ctx.interpolation = interploations[int(inputs.get('interpolation', 0))]
    ctx.image = image
    image = image.resize((320,320),ctx.interpolation)
    ctx.resized_image = image
    mask = mask.resize((320,320),ctx.interpolation)
    np_mask = np.array(mask)
    if len(np_mask.shape)>2:
        logging.warning('Mask shape is {}'.format(np_mask.shape))
        np_mask = np_mask[:,:,0]
    #np_mask[np.less(np_mask,128)]=0
    np_mask[np.logical_and(np_mask>0, np_mask<230)]=128
    np_mask[np.greater(np_mask,250)]=255
    #np_mask[np.less(np_mask,255)]=0
    ctx.np_mask = np_mask
    #input_trimap = generate_trimap(np_mask)
    input_trimap = np_mask
    input_trimap = np.expand_dims(input_trimap.astype(np.float32),2)
    ctx.input_trimap = input_trimap
    image = np.array(image).astype(np.float32)
    input_image = image-g_mean
    return {'input': [input_image],'trimap':[input_trimap]}


def postprocess_base(outputs, ctx):
    mask = outputs['output'][0]
    mask = np.reshape(mask,(320,320,1))
    np_mask = np.expand_dims(ctx.np_mask,2).astype(np.float32)
    masks = np.concatenate((np_mask,ctx.input_trimap,mask*255),axis=1)
    masks = np.concatenate((masks,masks,masks),axis=2).astype(np.uint8)
    image = (mask*(np.array(ctx.resized_image,dtype=np.float32))).astype(np.uint8)
    result = np.concatenate((masks,image),axis=1)
    image_bytes = io.BytesIO()
    result = Image.fromarray(result)
    result.save(result, format='PNG')
    outputs['image'] = image_bytes.getvalue()
    return outputs

def postprocess(outputs, ctx):
    mask = outputs['output'][0]*255
    mask = np.reshape(mask,(320,320))
    mask = np.clip(mask,0,255)
    mask_image = Image.fromarray(mask.astype(np.uint8))
    mask_image = mask_image.resize((ctx.image.size[0],ctx.image.size[1]),ctx.interpolation)
    mask_image = np.array(mask_image).astype(np.float32)/255
    if ctx.in_type=='np':
        logging.info("Return {}".format(mask_image.shape))
        outputs['image'] = mask_image
        return outputs
    mask_image = np.expand_dims(mask_image,2)
    image = np.array(ctx.image).astype(np.float32)
    result = (mask_image*image).astype(np.uint8)
    image_bytes = io.BytesIO()
    Image.fromarray(result).save(image_bytes, format='PNG')
    outputs['image'] = image_bytes.getvalue()
    return outputs