import os
import math
import glob
import random
import shutil
import numpy as np
import pandas as pd

# sci-image  
from skimage.io import imread
from skimage import img_as_bool
from skimage.transform import resize

class Kiln:

    def __init__( self, rows, cols, channels ):

        """
        constructor
        """

        # zip class names into dict
        labels = [  "kiln" ]

        self._classes = dict(zip( labels, range ( len( labels ) ) ) )
        self._channels = channels 

        self._rows = rows
        self._cols = cols

        # all data valid - legacy code
        self._nodata = np.ones( ( self._rows, self._cols ) )
        return


    def getPrefixes( self, path ):

        """
        get file prefixes
        """

        # get basename prefixes
        basenames = [ os.path.basename( f ) for f in glob.glob( os.path.join( path, 'images/*.tif' ) ) ]
        return [ os.path.splitext( f )[ 0 ] for f in basenames ]


    def getNoData( self, path, prefix ):

        """
        load image masked for nodata and farm boundaries
        """

        # read images
        #mask = np.array( imread( os.path.join( path, 'masks/{p}.png'.format( p=prefix ) ) ), dtype=np.uint8 ).clip( max=1 )
        #boundary = np.array( imread( os.path.join( path, 'boundaries/{p}.png'.format( p=prefix ) ) ), dtype=np.uint8 ).clip ( max=1 )

        #return mask * boundary
        return self._nodata


    def loadImages( self, path, prefix ):

        """
        load image masked for nodata and farm boundaries
        """

        # read and stack images
        rgb = np.array( imread( os.path.join( path, 'images/{p}.tif'.format( p=prefix ) ) ), dtype=np.uint8 )
        rgb = resize( rgb, ( self._rows, self._cols ), anti_aliasing=True, preserve_range=True )

        #nir = np.array( imread( os.path.join( path, 'images/nir/{p}.jpg'.format( p=prefix ) ) ), dtype=np.uint8 )
        #nir = resize( nir, ( self._rows, self._cols ), anti_aliasing=True, preserve_range=True )

        #images = np.dstack( ( rgb, nir ) )
        images = rgb

        # apply no data
        nodata = self.getNoData( path, prefix )
        nodata = nodata[ :, :, np.newaxis ]

        nodata = resize( nodata.astype( np.float64 ), ( self._rows, self._cols ), preserve_range=True )
        return images * img_as_bool( nodata )


    def loadLabels( self, path, prefix ):

        """
        load binary label masks
        """

        # load label masks
        labels = []; keys = list( self._classes.keys() )
        if len( keys ) == 1:

            data = np.array( imread( os.path.join( path, 'labels/{name}/{p}-mask.tif'.format( name=keys[ 0 ], p=prefix ) ) ), dtype=np.uint8 ).clip( max=1 )
            data = img_as_bool( resize( data.astype( np.float64 ), ( self._rows, self._cols ) ) )

            labels.append ( data.astype( np.float64 ) )
            labels.append ( 1.0 - data.astype( np.float64 ) )

        else:

            for k in keys:
                
                data = np.array( imread( os.path.join( path, 'labels/{name}/{p}-mask.tif'.format( name=k, p=prefix ) ) ), dtype=np.uint8 ).clip( max=1 )
                data = img_as_bool( resize( data.astype( np.float64 ), ( self._rows, self._cols ) ) )

                labels.append ( data.astype( np.float64 ) )


        return np.dstack( labels )


    def getLabelStatistics( self, path, prefix ):
        
        """
        get statistics
        """

        # load label masks
        labels = self.loadLabels( path, prefix )
        nodata = self.getNoData( path, prefix )

        # compute coverage for each label class
        stats = { 'prefix' : prefix }
        for k, v in self._classes.items():
            stats[ k ] = ( np.sum( labels[ :, :, v ] ) / np.sum( nodata ) ) * 100.0

        return stats


    def getLabelGroups( self, pathname ):

        """
        get class-specific data frames 
        """

        label_dfs = []

        # read stats csv 
        df = pd.read_csv( pathname )
        for k in self._classes.keys():
            label_dfs.append( df[ df[ k ] > 0 ] )
        
        return label_dfs


    def getImageStatistics( self, root ):

        """
        get normalisation stats of multispectral imagery
        """

        # initialise stats
        sum_x = np.zeros( self._channels ); sum_x2 = np.zeros( self._channels )
        count = np.zeros( self._channels )

        for subset in [ 'train', 'val', 'test' ]:

            path = os.path.join( root, subset )
            prefixes = self.getPrefixes( path )

            for prefix in prefixes:

                # read and stack images
                rgb = np.array( imread( os.path.join( path, 'images/{p}.tif'.format( p=prefix ) ) ), dtype=np.float64 )
                #nir = np.array( imread( os.path.join( path, 'images/nir/{p}.jpg'.format( p=prefix ) ) ), dtype=np.float64 )
                #images = np.dstack( ( rgb, nir ) )
                images = rgb

                # apply no data
                nodata = np.reshape( self.getNoData( path, prefix ), -1 )
                for channel in range( self._channels ):

                    # flatten channel data
                    data = np.reshape(images[:,:,channel], -1)
                    count[ channel ] += np.sum( nodata )

                    # update sum and sum of squares 
                    sum_x[ channel ] += np.sum( data[ nodata > 0 ] )
                    sum_x2[ channel ] += np.sum( data[ nodata > 0 ] **2 )


        # for each channel
        stats = []
        for channel in range( self._channels ):

            # compute mean and stdev from summations
            mean = sum_x[ channel ] / count [ channel ]
            stdev = math.sqrt ( sum_x2[ channel ] / count [ channel ] - mean**2 )
                
            # append channel stats to list
            stats.append ( [ channel, mean, stdev ] )

        # convert list to dataframe and save to csv
        df = pd.DataFrame( stats, columns =['channel', 'mean', 'stdev'], dtype=float ) 
        df.to_csv( os.path.join( root, 'image_stats.csv' ), index=False )
            
        return


    def splitDataset( self, params, train_pc=0.6, val_pc=0.2, test_pc=0.2 ):
    
        """
        split images / labels into randomly selected subsets
        """

        def copySubset( images, subset ):

            """
            copy files to subset directory
            """

            def copyFile( pathname, dst_path ):

                """
                copy file to new directory
                """

                if not os.path.exists( dst_path ):
                    os.makedirs( dst_path )

                shutil.copy( pathname, dst_path )
                return

            # get subset path
            out_path = os.path.join( params[ 'out_path' ], subset )
            for image in images:

                # locate mask file
                mask = os.path.join( params[ 'mask_path' ], os.path.basename( image ).replace( '.tif', '-mask.tif' ) )
         
                #pdb.set_trace()
                if os.path.exists ( mask ):

                    # copy image + mask to new location 
                    copyFile( image, os.path.join( out_path, 'images/' ) )
                    copyFile( mask, os.path.join( out_path, 'labels/kiln/' ) )

            return

        # get image pathnames
        images = glob.glob( os.path.join( params[ 'image_path' ], '*.tif' ) )

        # shuffle prefixes        
        random.shuffle( images )

        # get training data
        train_idx = int( train_pc * len( images ) )
        copySubset( images[ : train_idx ], 'train'  )

        # get validation and test
        val_idx = int( val_pc * len( images ) )
        copySubset( images[ train_idx : train_idx + val_idx ], 'val' )
        copySubset( images[ train_idx + val_idx : ], 'test' )

        return

"""
# create test object
obj = Kiln( 128, 128, 3 )
params = {  'image_path' : 'C:\\Users\\crwil\Documents\\chips-kilns\\images',
            'mask_path' : 'C:\\Users\\crwil\Documents\\chips-kilns\\labels\\kiln',
            'out_path' : 'C:\\Users\\crwil\\Documents\\chips\\data\\' 
}

obj.splitDataset( params, train_pc=1.0, val_pc=0.0, test_pc=0.0 )

# get image stats
root = 'C:\\Users\\crwil\\Documents\\chips\\data' 
obj.getImageStatistics( root )

# test prefixes
path = os.path.join( root, 'train' )
prefixes = obj.getPrefixes( path )

# example loads
images = obj.loadImages( path, prefixes[ 0 ] )
labels = obj.loadLabels( path, prefixes[ 0 ] )

root = params[ 'out_path' ]
for subset in [ 'train', 'val', 'test' ]:

    path = os.path.join( root, subset )
    prefixes = obj.getPrefixes( path )

    # compute class coverage stats
    stats = []
    for p in prefixes:
        stats.append( obj.getLabelStatistics( path, p  ) ) 

    # convert to dataframe and save
    df = pd.DataFrame( stats )
    df.to_csv( os.path.join( path, 'label_stats.csv' ), index=False )
"""

