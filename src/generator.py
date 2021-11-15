import os
import math
import random
import numpy as np
import pandas as pd

# numpy random
from numpy.random import uniform
from numpy.random import random_integers

# sci-image  
from skimage.io import imread
from skimage.util import random_noise
from sklearn.utils import shuffle as pd_shuffle

# sci-image transform
from skimage.transform import AffineTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp


def ImageMaskDataGenerator( obj,
                            path,
                            batch_size,
                            image_stats=None,
                            shuffle=True,
                            horizontal_flip=False,
                            vertical_flip=False,
                            rotation_range=0,
                            shear_range=0,
                            scale_range=1,
                            transform_range=0,
                            filling_mode='edge',
                            speckle=None,
                            crop=False,
                            crop_size=None ):

    """
    generate batches of augmented multichannel imagery catalogued in data frame list
    """

    # load frames with examples of each group 
    groups = obj.getLabelGroups( os.path.join( path, 'label_stats.csv' ) )

    # shuffle data frames
    if shuffle:
        for group in groups:
            group = pd_shuffle(group)
    else:
        # verify single dataframe without shuffle
        assert ( len( groups ) == 1 )

    # create index
    current_idx = 0
    while True:
        
        # initialise batch lists
        batch_X = []
        batch_Y = []
        
        # create empty sampling dataframe
        sample = pd.DataFrame( columns=groups[ 0 ].columns )
        if shuffle:

            # construct random stratified sample from input dataframes
            for idx in range( batch_size ):
            
                # randomly select grouping and random pick single record
                fid = random.randrange( 0, len( groups ) )
                sample = sample.append ( groups[ fid ].sample(), ignore_index=True )

        else:

            # get sample by slicing dataframe at incremental index 
            sample = groups[ 0 ][ current_idx : current_idx + batch_size ]
            current_idx += batch_size

            # reset at end
            if current_idx >= len ( groups[ 0 ] ):
                current_idx = 0

            # check sample size equals batch size
            if batch_size > len ( sample ):
                sample = sample.append ( groups [ 0 ] [ 0 : ( batch_size - len ( sample ) ) ] )
                current_idx = ( batch_size - len ( sample ) )

    
        # iterate through sample rows
        #counts = np.zeros( 5 )
        for idx, row in sample.iterrows():

            # read image and mask
            images = obj.loadImages( path, row[ 'prefix'] )
            labels = obj.loadLabels( path, row[ 'prefix'] )
                                        
            # apply normalisation / standardisation
            if image_stats is not None:
                images = standardiseImage( images, image_stats )

            # optionally apply random flip
            if horizontal_flip or vertical_flip:
                images, labels = applyFlip(     images, 
                                                labels,
                                                horizontal_flip=horizontal_flip,
                                                vertical_flip=vertical_flip )
        
            # optionally apply random affine transformation
            images, labels = applyAffineTransform(      images, 
                                                        labels,
                                                        rotation_range=rotation_range, 
                                                        shear_range=shear_range,
                                                        scale_range=scale_range,
                                                        transform_range=transform_range, 
                                                        warp_mode=filling_mode )

            # optionally apply random speckle noise
            if speckle is not None:
                images = applySpeckleNoise( images, speckle )

            # optionally apply random crop
            if crop:
                if crop_size is None:
                    crop_size = images.shape[0:2]
                images = applyCentreCrop( images, labels, crop_size )

            # add image and mask
            batch_X += [ images ]
            batch_Y += [ labels ]

        # convert lists to np.array
        X = np.array(batch_X)
        Y = np.array(batch_Y)

        """
        total = np.sum( counts )
        print ( counts / total )
        """

        # pump out batch
        yield ( X, Y )

    
    return None


def standardiseImage(   images, 
                        stats ):

    """
    compute z-score image 
    """

    # compute z score across each channel
    for idx, row in stats.iterrows():
        images[..., idx] -= row[ 'mean' ]

        if 'stdev' in stats.columns:
            images[..., idx] /= row[ 'stdev' ]

    return images


def applyFlip(  images, 
                labels,
                horizontal_flip=False, 
                vertical_flip=False ):

    """
    apply random flip to N-channel image
    """

    # randomly flip image up/down
    if horizontal_flip:
        if random.choice([True, False]):
            images = np.flipud(images)
            labels = np.flipud(labels)
    
    # randomly flip image left/right
    if vertical_flip:
        if random.choice([True, False]):
            images = np.fliplr(images)
            labels = np.fliplr(labels)

    return images, labels


def applyAffineTransform(   images, 
                            labels,
                            rotation_range=0, 
                            shear_range=0, 
                            scale_range=1, 
                            transform_range=0,
                            warp_mode='edge' ):

    """
    apply optional random affine transformation to N-channel image
    """

    # generate transformation parameters
    image_shape = images.shape

    rotation_angle = uniform(low=-abs(rotation_range), high=abs(rotation_range) )
    shear_angle = uniform(low=-abs(shear_range), high=abs(shear_range))
    scale_value = uniform(low=abs(1 / scale_range), high=abs(scale_range))
    translation_values = (random_integers(-abs(transform_range), abs(transform_range)), random_integers(-abs(transform_range), abs(transform_range)))

    # initialise transformations
    transform_toorigin = SimilarityTransform(   scale=(1, 1), 
                                                rotation=0, 
                                                translation=(-image_shape[0], -image_shape[1]))
    
    transform_revert = SimilarityTransform( scale=(1, 1), 
                                            rotation=0, 
                                            translation=(image_shape[0], image_shape[1]))

    # generate affine transformation
    transform = AffineTransform(    scale=(scale_value, scale_value), 
                                    rotation=np.deg2rad(rotation_angle),
                                    shear=np.deg2rad(shear_angle), 
                                    translation=translation_values)

    # apply affine transform
    images = warp(      images, 
                        ((transform_toorigin) + transform) + transform_revert, 
                        mode=warp_mode, 
                        preserve_range=True )

    labels = warp(      labels, 
                        ((transform_toorigin) + transform) + transform_revert, 
                        mode=warp_mode, 
                        preserve_range=True )

    return images, labels


def applySpeckleNoise(  images, 
                        variance ):

    """
    add optional random speckle noise
    """

    # normalise image
    image_max = np.max(np.abs(images), axis=(0, 1))
    images /= image_max

    # add speckle noise and rescale
    images = random_noise(images, mode='speckle', var=variance)
    images *= image_max

    return images


def applyCentreCrop(    images, 
                        labels,
                        target_size ):

    """
    apply predefined centre crop to image
    """

    # check bounds
    x_crop = min(images.shape[0], target_size[0])
    y_crop = min(images.shape[1], target_size[1])
    midpoint = [math.ceil(images.shape[0] / 2), math.ceil(images.shape[1] / 2)]

    # apply crop
    crop_images = images[int(midpoint[0] - math.ceil(x_crop / 2)):int(midpoint[0] + math.floor(x_crop / 2)),
                    int(midpoint[1] - math.ceil(y_crop / 2)):int(midpoint[1] + math.floor(y_crop / 2)),
                    :]

    # apply crop
    crop_labels = labels[int(midpoint[0] - math.ceil(x_crop / 2)):int(midpoint[0] + math.floor(x_crop / 2)),
                    int(midpoint[1] - math.ceil(y_crop / 2)):int(midpoint[1] + math.floor(y_crop / 2)),
                    :]
    
    assert crop_images.shape[0:2] == target_size and crop_labels.shape[0:2] == target_size
    return crop_images, crop_labels
