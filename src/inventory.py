import geopandas as gpd

from osgeo import osr
from osgeo import gdal
from shapely.geometry import box


class Inventory():

    # construct geographic srs
    _geo_srs = osr.SpatialReference()
    _geo_srs.ImportFromEPSG( 4326 )

    @staticmethod
    def get( pathnames, maxRecords=0 ):

        """
        get spatially indexed image records
        """

        # for each pathname
        records = []
        for pathname in pathnames:

            # open image
            ds = gdal.Open( pathname )
            if ds is not None:

                # check 24bit or above
                if ds.RasterCount >= 3:

                    # append record with pathname and bbox geometry
                    records.append( {   'cols' : ds.RasterXSize,
                                        'rows' : ds.RasterYSize,
                                        'pathname' : pathname,
                                        'transform' : ds.GetGeoTransform(),
                                        'projection' : ds.GetProjection(),
                                        'geometry' : Inventory.getBoundingBox( ds ) } )

                    
                    if maxRecords > 0 and len( records ) > maxRecords:
                        break

        return gpd.GeoDataFrame( records, crs='EPSG:4326', geometry='geometry' ) if len( records ) > 0 else None


    @staticmethod
    def getBoundingBox( ds ):

        """ 
        convert geographic extent into shapely polygon
        """

        # get extent
        extent = Inventory.getExtent ( ds )

        # get spatial reference
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt( ds.GetProjection() )

        # get geographic extent - convert to shapely geometry
        coords = Inventory.reprojectCoordinates( extent, src_srs, Inventory._geo_srs )
        return box ( coords[ 0 ][  1 ], coords[ 0 ][ 0 ], coords[ 1 ][ 1 ], coords[ 1 ][ 0 ] )


    @staticmethod
    def getExtent( ds ):

        """ 
        get corner coordinates
        """
        
        # get extent
        xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
        width, height = ds.RasterXSize, ds.RasterYSize
        xmax = xmin + width * xpixel
        ymin = ymax + height * ypixel

        return (xmin, ymin), (xmax, ymax)


    @staticmethod
    def reprojectCoordinates( coords, src_srs, tgt_srs ):

        """ 
        reproject list of x,y coordinates. 
        """

        # transfrom coordinates
        tx_coords=[]
        transform = osr.CoordinateTransformation( src_srs, tgt_srs )
        for x,y in coords:
            x,y,z = transform.TransformPoint(x,y)
            tx_coords.append([x,y])

        return tx_coords
