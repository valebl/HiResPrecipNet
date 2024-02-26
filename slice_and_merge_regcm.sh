#!/bin/bash

LON_MIN=$1
LON_MAX=$2
LAT_MIN=$3
LAT_MAX=$4
INTERVAL=$5
INPUT_PATH_PHASE_1A=$6
OUTPUT_PATH=$7
PREFIX=$8

lon_min_=$(echo $LON_MIN-3\*$INTERVAL | bc)
lon_max_=$(echo $LON_MAX+3\*$INTERVAL | bc)
lat_min_=$(echo $LAT_MIN-3\*$INTERVAL | bc)
lat_max_=$(echo $LAT_MAX+3\*$INTERVAL | bc)

echo $lon_min_
echo $lon_max_
echo $lat_min_
echo $lat_max_

#for v in 'hus' 'ta' 'ua' 'va' 'zg' ; do
#	for year in '2000' '2001' '2002' '2003' '2004' '2005' '2006' '2007' '2008' '2009'; do
#	cdo -O -f nc4 -z zip -L -b F32 merge "${v}200_${year}.nc" "${v}500_${year}.nc" "${v}700_${year}.nc" "${v}850_${year}.nc" "${v}1000_${year}.nc" "${OUTPUT_PATH}${v}_${year}.nc"
#	done
#done

cd ${OUTPUT_PATH}

## slice each file to the desired lon and lat window
for v in 'hus' 'ta' 'ua' 'va' 'zg' ; do
	files=$(ls ${v}_*.nc)
	for file in $files ; do
		cdo sellonlatbox,$lon_min_,$lon_max_,$lat_min_,$lat_max_ "${file}" "${PREFIX}${file}"
	done
done

## merge all the sliced files into a single file for all the time span considered
for v in 'hus' 'ta' 'ua' 'va' 'zg' ; do
	cdo -O -f nc4 -z zip -L -b F32 mergetime "${PREFIX}${v}_2000.nc" "${PREFIX}${v}_2001.nc" "${PREFIX}${v}_2002.nc" "${PREFIX}${v}_2003.nc" "${PREFIX}${v}_2004.nc" "${PREFIX}${v}_2005.nc" "${PREFIX}${v}_2006.nc" "${PREFIX}${v}_2007.nc" "${PREFIX}${v}_2008.nc" "${PREFIX}${v}_2009.nc" "${PREFIX}${v}.nc"
	## remove temporary files no longer usefule after merging
	#files=$(ls ${PREFIX}${v}_*.nc)
	#for file in $files ; do
#		rm ${file}
#	done
done

