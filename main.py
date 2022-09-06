import numpy as np
import skimage as sk
import scipy
import skimage.io as skio
import skimage.transform
from IPython.display import Image
import PIL
import glob
import argparse


import os

path = './out_path'
isExist = os.path.exists(path)
if not isExist:
    # Create a new directory because it does not exist 
    os.makedirs(path)
    print("The new directory is created!")


parser = argparse.ArgumentParser(description='Align images')
parser.add_argument('--diff', type=int, default=15, help='difference between images')
parser.add_argument('--levels', type=int, default=None, help='levels of pyramid')
parser.add_argument('--auto', action='store_true', default=False, help='auto levels and diff')
parser.add_argument('--contrast', action='store_true', default=False, help='auto contrast')
parser.add_argument('--crop', action='store_true', default=False, help='crop images')
parser.add_argument('--fname', type=str, default='church.tif', help='file name')
parser.add_argument('--align_metric', type=str, default='ssd', help='align metric')
parser.add_argument('--align_func', type=str, default='align', help='align function')

parser.add_argument('--all', action='store_true', default=False, help='all images')
args = parser.parse_args()


def ncc(im1, im2):

    norm_im1 = im1 / np.sqrt(np.sum(im1**2))
    norm_im2 = im2 / np.sqrt(np.sum(im2**2))
    return -np.sum(norm_im1 * norm_im2) / (im1.shape[0] * im1.shape[1])

def ssd(im1, im2):
    return np.sum((im1 - im2)**2) / (im1.shape[0] * im1.shape[1])

def align_unrolled(im1, im2, diff):
    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)
    # diff = min(im1.shape[0]//2, im1.shape[1]//2, 15) # diff
    diff = min(im1.shape[0]//2, im1.shape[1]//2, diff)
    best_score = float('inf')
    aligned_image = im1
    best_indices = 0, 0
    for i in range(-diff, diff + 1):
        for j in range(-diff, diff + 1):
            img1_rolled = np.roll(im1, (i, j), axis=(0, 1))
            img1_eval, img2_eval = img1_rolled.copy(), im2.copy()
            if i >= 0:
                img1_eval[:i, :] = 0
                img2_eval[:i, :] = 0
            else:
                img1_eval[i:, :] = 0
                img2_eval[i:, :] = 0
            if j >= 0:
                img1_eval[:, :j] = 0
                img2_eval[:, :j] = 0
            else:
                img1_eval[:, j:] = 0
                img2_eval[:, j:] = 0

            if args.align_metric == 'ssd':
                score = ssd(img1_eval, img2_eval)
            elif args.align_metric == 'ncc':
                score = ncc(img1_eval, img2_eval)
            # score = ncc(img1_rolled, im2)
            if score < best_score:
                best_score = score
                best_indices = (i, j)
                aligned_image = translate_img(im1, best_indices)
                
    return aligned_image, best_indices

def align(im1, im2, diff):
    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)
    # diff = min(im1.shape[0]//2, im1.shape[1]//2, 15) # diff
    hbound = min(im1.shape[0]//4,  diff)
    wbound = min(im1.shape[1]//4,  diff)
    best_score = float('inf')
    aligned_image = im1
    best_indices = 0, 0
    for i in range(-hbound, hbound + 1):
        for j in range(-wbound, wbound + 1):
            img1_rolled = np.roll(im1, (i, j), axis=(0, 1))
            if args.align_metric == 'ssd':
                score = ssd(img1_rolled, im2)
            elif args.align_metric == 'ncc':
                score = ncc(img1_rolled, im2)
            if score < best_score:
                best_score = score
                aligned_image = img1_rolled
                best_indices = (i, j)
    return aligned_image, best_indices


def align_raw(im1, im2, diff=15):
    best_indices = 0, 0
    return im1, best_indices

def align_img_pyramid(im1, im2, diff, levels, align_func=align):
    shrink = 0.75
    if levels == 0:
        
        return align_func(im1, im2, diff=diff)
    else:
        im1_rescaled = sk.transform.rescale(im1, shrink, anti_aliasing=True)
        im2_rescaled = sk.transform.rescale(im2, shrink, anti_aliasing=True)
        _, found_displacement =  align_img_pyramid(im1_rescaled, im2_rescaled, align_func=align_func, levels=levels - 1, diff=diff)
        print("Rescaled shape:", im1_rescaled.shape, "Estimated shift: {}".format(found_displacement))
        found_displacement = int(found_displacement[0] * 1/shrink), int(found_displacement[1] * 1/shrink)
        img1_rolled = translate_img(im1, found_displacement)
        aligned_img, new_displacement = align_func(img1_rolled, im2, diff=diff)
        return aligned_img, (found_displacement[0] + new_displacement[0], found_displacement[1] + new_displacement[1])

def crop_img(im, percent=20): # the percentage of image to be cropped
    h, w = im.shape[0], im.shape[1]
    half_percentage = percent/4
    h_crop = int(h * half_percentage/100)
    w_crop = int(w * half_percentage/100)
    return im[h_crop:-max(h_crop, 1), w_crop:-max(w_crop, 1)]

def translate_img(img, offsets):
    return np.roll(img, offsets, axis=(0, 1))


def align_img(imname, align_func, diff=15, levels=None, auto=False):
    
    # read in the image
    fname = imname
    im = skio.imread(fname)
    imname = imname.split('\\')[-1]
    imname = imname.split('/')[-1]
    print("Aligning {}...".format(imname))
    
    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    r_eval, g_eval, b_eval = r.copy(), g.copy(), b.copy()
    
    if r.shape[0] > 1000:
        r_eval, g_eval, b_eval = crop_img(r_eval), crop_img(g_eval), crop_img(b_eval)
    # skio.imshow(r_eval)
    # skio.show()
    # skio.imshow(g_eval)
    # skio.show()
    # skio.imshow(b_eval)
    # skio.show()
    if auto:
        h, w = r_eval.shape
        diff=4
        shrink=0.75
        levels = min(int(np.log(h)/np.log(1/shrink)), int(np.log(w)/np.log(1/shrink)))
        print('Auto levels:', levels)
        print('Auto diff:', diff)
    # if levels:
    #     ag, dis1 = align_img_pyramid(g_eval, b_eval, diff=diff, levels=levels, align_func=align_func)
    #     print("Green shift: {}".format(dis1))
    #     ar, dis2 = align_img_pyramid(r_eval, b_eval, diff=diff, levels=levels, align_func=align_func)
    #     print("Red shift: {}".format(dis2))
    # else:
    #     ag, dis1 = align_func(g_eval, b_eval, diff=diff)
    #     print("Green shift: {}".format(dis1))
    #     ar, dis2 = align_func(r_eval, b_eval, diff=diff)
    #     print("Red shift: {}".format(dis2))
    # # create a color image
    
    # ag = translate_img(g, dis1)
    # ar = translate_img(r, dis2)
    # im_out = (np.dstack([ar, ag, b]) * 255).astype(np.uint8)

    if levels:
        ab, dis1 = align_img_pyramid(b_eval, g_eval, diff=diff, levels=levels, align_func=align_func)
        
        ar, dis2 = align_img_pyramid(r_eval, g_eval, diff=diff, levels=levels, align_func=align_func)
        
    else:
        ab, dis1 = align_func(b_eval, g_eval, diff=diff)
        ar, dis2 = align_func(r_eval, g_eval, diff=diff)

    # create a color image
    print("blue shift: {}".format(dis1))
    print("Red shift: {}".format(dis2))
    ab = translate_img(b, dis1)
    ar = translate_img(r, dis2)
    im_out = (np.dstack([ar, g, ab]) * 255).astype(np.uint8)

    # save the image
    fname = 'out_path/aligned_{}.jpg'.format(imname[:-4])
    skio.imsave(fname, im_out)

    # display the image
    # skio.imshow(im_out)
    # skio.show()
    
    return im_out, (dis1, dis2)


    
def auto_contrast(im):
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    
    for i, co in enumerate([r.flatten(), g.flatten(), b.flatten()]):
        histogram = np.zeros(256)
        for pixel in co:
            histogram[pixel] += 1
        cs = np.cumsum(histogram)
        nj = (cs - cs.min()) * 255
        N = cs.max() - cs.min()
        cs = nj / N
        cs = cs.astype('uint8')
        im[:, :, i] = cs[co].reshape(im.shape[0], im.shape[1])

    
    return im

tifs = glob.glob('data/*.tif')
jpgs = glob.glob('data/*.jpg')

if args.align_func == 'align':
    align_func = align
elif args.align_func == 'align_unrolled':
    align_func = align_unrolled

if args.all:
    offset_dict = dict()
    for imname in tifs:
        im, (dis1, dis2) = align_img(imname, align_func=align_func, auto=True)
        offset_dict[imname] = (dis1, dis2)
    for imname in jpgs:
        im, (dis1, dis2) = align_img(imname, align_func=align_func, auto=True)
        offset_dict[imname] = (dis1, dis2)
    print(offset_dict)
else:
    img, dis = align_img(args.fname, align_func=align_func, diff=args.diff, levels=args.levels, auto=args.auto)
    skio.imshow(img)
    skio.show()
    imname = args.fname.split('\\')[-1]
    imname = imname.split('/')[-1][:-4]
    if args.crop:
        img = crop_img(img, percent=20)
        skio.imsave('out_path/cropped_{}.jpg'.format(imname), img)
        skio.imshow(img)
        skio.show()
    if args.contrast:
        img = auto_contrast(img)
        skio.imsave('out_path/contrasted_{}.jpg'.format(imname), img)
        skio.imshow(img)
        skio.show()
    # img, dis1, dis2 = align_img('melons.tif', align_img_pyramid,auto=True)

    # img = auto_contrast(img)
    # skio.imshow(img)
    # skio.show()