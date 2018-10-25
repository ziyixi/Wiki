# Develop a parallel signal processing tool for seismic data analysis
### Ziyi Xi

## Brief Introduction

Seismic signals are usually transient waveforms radiated from a localized natural or manmade
seismic source. They can be used to locate the source, to analyze source processes, and
to study the structure of the medium of propagation. 

So seismic signal processing is very important for further geophysical research. This processing procedure includes the
following steps:

1. **data storage**: Usually we have data that could attain the TB scale, so storage it in HDF5 format could be very efficient.

2. **data resampling**: Data in different stations may have different sampling rate, and in our research, we may need some specific
sampling rate.

3. **data rotation**: The seismic data contains three components(directions) to descripe the movement of the groud. But in our research
we need data in some specific directions, which need rotate the recorded data.

4. **data filtering**: The seismic data may contain lots of high frequency noise, and there are also some very low frequency wave
such as tidal wave combined in. Filtering the data could let us focus on where we are interested in.

5. **others**: We may also need some other processing procedures such as remove the station response(deconvolution), waveform stacking, etc.

## The connection between this project and our class

My aim is to implement some parallel signal processing algorithms to above steps. There exists a python package called ASDF, which 
has implemented the HDF5 to storage seismic data, but it lacks interface to other commonly used seismic file format. So I could firstly
provide some APIs for it, which should use the knowledge of parallel IO.

And then, for signal processing algorithms, I could directly use the way of SPMD to implement them. Also I could parallel
the algorithm itself. Thus MPi, CUDA are some good choices. Especially I want to test CUDA for this kind of problems.

## The meaningful part

1. Now in my research I'm facing the problem of analysing large amount of seismic data, so this project would be meaningful for my future research.

2. I may write a parallel signal processing package based on it.
