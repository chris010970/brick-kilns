import os
import time
import argparse
import numpy as np
import pandas as pd
import generator 
import matplotlib.pyplot as plt

# unet models + local imports
from keras_unet_collection import models
from kiln import Kiln


class Predict:

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

        # load weights
        self._model.load_weights( args.weights )
        return


    def process( self, args ):

        """
        main path of execution
        """

        # convert list to dataframe and save to csv
        image_stats = pd.read_csv( os.path.join( args.data_path, 'image_stats.csv' ) )

        # load frames with examples of each group 
        path = os.path.join( args.data_path, 'test' )
        groups = self._obj.getLabelGroups( os.path.join( path, 'label_stats.csv' ) )

        # for each group
        batch_size=32
        for group in groups:

            df = group.sample(frac=1).reset_index(drop=True)
            df = df[ : batch_size ] 

            batch = []
            for idx, row in df.iterrows():

                # read image and mask
                image = self._obj.loadImages( path, row[ 'prefix' ] )

                # apply normalisation / standardisation
                if image_stats is not None:
                    batch.append( generator.standardiseImage( image, image_stats ) )

            # compute prediction
            results = self._model.predict( np.asarray( batch ), batch_size=1 )

            # load mask
            for idx, row in df.iterrows():

                image = self._obj.loadImages( path, row[ 'prefix' ] )
                label = self._obj.loadLabels( path, row[ 'prefix' ] )

                data = image[ ::, ::, 0:3 ]

                # get stats
                mean = np.mean( data )
                std = np.std( data ) 

                # clip to 95%
                _min = mean-2*std; _max = mean+2*std
                clip_data = np.clip( data, mean-2*std, mean+2*std )

                # compute normalised data
                norm_data = ( clip_data - _min ) / ( _max - _min )
                
                # plot label mask vs predicted mask
                _, ax = plt.subplots(nrows=1, ncols=3 )
                ax[ 0 ].imshow( norm_data )                
                ax[ 1 ].imshow( label[ :, :, 0 ] )
                ax[ 2 ].imshow( results[ idx, :, :, 0 ] )

                # show images
                plt.tight_layout()
                plt.show()
                

        return


def parseArguments(args=None):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='eurosat train')
    parser.add_argument('data_path', action='store')
    parser.add_argument('weights', action='store')

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
                            default=6 )

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()
    
    # create and execute training instance
    obj = Predict( args )
    obj.process( args )

    return


# execute main
if __name__ == '__main__':
    main()
