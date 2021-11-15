import os
import time
import argparse
import pandas as pd

# keras optimisers
from tensorflow.keras.optimizers import Adam

# callbacks
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras_unet_collection import models

# local imports
from loss import *
from kiln import Kiln
from generator import ImageMaskDataGenerator

class Train:

    def __init__( self, args ):

        """
        constructor
        """

        # create dataset specific object
        self._channels = 3
        self._obj = Kiln( args.rows, args.cols, self._channels )

        # create xnet model from scratch
        self._model = models.unet_2d((None, None, self._channels), [64, 128, 256, 512], n_labels=2,
                                        stack_num_down=2, stack_num_up=2,
                                        output_activation='Softmax', 
                                        batch_norm=True, 
                                        pool='max', 
                                        unpool='bilinear', 
                                        name='unet')

        # optionally load weights
        if args.load_weights is not None:
            self._model.load_weights( args.load_weights )

        return


    def process( self, args ):

        """
        main path of execution
        """

        # convert list to dataframe and save to csv
        stats = pd.read_csv( os.path.join( args.data_path, 'image_stats.csv' ) )

        # get train generator
        train_generator = ImageMaskDataGenerator(   self._obj,
                                                    os.path.join( args.data_path, 'train' ),
                                                    args.batch_size,
                                                    image_stats=stats,
                                                    horizontal_flip=args.horizontal_flip,
                                                    vertical_flip=args.vertical_flip,
                                                    rotation_range=args.rotation,
                                                    shear_range=args.shear,
                                                    scale_range=args.scale,
                                                    transform_range=args.transform,
                                                    speckle=args.speckle )

        # get test generator
        test_generator = ImageMaskDataGenerator(    self._obj,
                                                    os.path.join( args.data_path, 'val' ),
                                                    args.batch_size,
                                                    image_stats=stats )

        # compile model
        opt = Adam( lr=1e-6 )
        #opt = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

        self._model.compile(    optimizer=opt, 
                                loss=dice_coef,
                                metrics=["binary_crossentropy" ] )

        # setup callbacks
        callbacks = [ CSVLogger( 'log.csv', append=True ) ]
        if args.checkpoint_path is not None:

            # create sub-directory if required
            if not os.path.exists ( args.checkpoint_path ):
                os.makedirs( args.checkpoint_path )

            # setup checkpointing callback
            pathname = os.path.join( args.checkpoint_path, "weights-{epoch:02d}-{val_loss:.2f}.h5" )
            checkpoint = ModelCheckpoint(   pathname, 
                                            monitor='val_loss', 
                                            verbose=1, 
                                            save_best_only=True, 
                                            mode='min' )
            callbacks.append( checkpoint )


        # initiate training
        self._model.fit_generator(  train_generator,
                                    steps_per_epoch=args.train_steps,
                                    epochs=args.epochs,
                                    callbacks=callbacks,
                                    validation_data=test_generator,
                                    validation_steps=args.validation_steps )

        # save final set of weights to file
        if args.save_weights is not None:

            if not os.path.exists( os.path.dirname( args.save_weights ) ):
                os.makedirs( os.path.dirname( args.save_weights ) )

            self._model.save_weights( args.save_weights )

        return


def parseArguments(args=None):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('data_path', action='store')

    # epochs
    parser.add_argument(    '--epochs', 
                            type=int,
                            action='store',
                            default=100 )

    # epochs
    parser.add_argument(    '--rows', 
                            type=int,
                            action='store',
                            default=128 )

    # epochs
    parser.add_argument(    '--cols', 
                            type=int,
                            action='store',
                            default=128 )

    # steps per epoch
    parser.add_argument(    '--train_steps', 
                            type=int,
                            action='store',
                            default=100 )

    parser.add_argument(    '--validation_steps', 
                            type=int,
                            action='store',
                            default=20 )

    # batch size
    parser.add_argument(    '--batch_size', 
                            type=int,
                            action='store',
                            default=8 )

    # -------------- augmentation parameters -----------------------

    # affine transform
    parser.add_argument(    '--rotation', 
                            type=int,
                            action='store',
                            default=0 )

    parser.add_argument(    '--shear', 
                            type=int,
                            action='store',
                            default=0 )

    parser.add_argument(    '--scale', 
                            type=int,
                            action='store',
                            default=1 )

    parser.add_argument(    '--transform', 
                            type=int,
                            action='store',
                            default=0 )

    # flip parameters
    parser.add_argument(    '--horizontal_flip', 
                            type=bool,
                            action='store',
                            default=False )

    parser.add_argument(    '--vertical_flip', 
                            type=bool,
                            action='store',
                            default=False )

    # centre crop parameters
    parser.add_argument(    '--crop', 
                            type=bool,
                            action='store',
                            default=False )

    parser.add_argument(    '--crop_size', 
                            type=int,
                            action='store',
                            default=None )

    # speckle noise
    parser.add_argument(    '--speckle', 
                            type=float,
                            action='store',
                            default=None )

    # warp fill mode
    parser.add_argument(    '--filling_mode', 
                            action='store',
                            default='edge' )

    # checkpoint path
    parser.add_argument(    '--checkpoint_path', 
                            action='store',
                            default=None )

    # load path
    parser.add_argument(    '--load_weights', 
                            action='store',
                            default=None )

    # save path
    parser.add_argument(    '--save_weights', 
                            action='store',
                            default=None )


    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()
    
    # create and execute training instance
    obj = Train( args )
    obj.process( args )

    return


# execute main
if __name__ == '__main__':
    main()
