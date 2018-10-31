# Develop a parallel signal processing tool for seismic data analysis
### Ziyi Xi

## Brief Introduction

Seismic signals are usually transient waveforms radiated from a localized natural or manmade
seismic source. They can be used to locate the source, to analyze source processes, and
to study the structure of the medium of propagation. But now almost all the processing tools are not paralleled, which may reduce the effiency to analysis this kind of digital signal, so I'm planning to develop a parallel processing tool to handle this problem.

## The connection between this project and our class

Here I'd like to discuss each part of widely used seismic signal processing, and how I could connect each part with the parallel computing.

1. **data storage**: 
>After we have collected data from some data center, we should store them in some spectfic data format. Generally we store them in [sac](https://ds.iris.edu/files/sac-manual/manual/file_format.html) format, which has some attributes containing information like sampling rate, seismic event time, location, source mechanism and etc.. And it also has a time series that represents the waveform. People have also developed a python package named [pyasdf](https://github.com/SeismicData/pyasdf) that uses h5py to store the data in hdf5 format. As the data we download from the data center is usually in sac format, however, the conversion from sac to asdf format are not paralleled. 

>So I am thinking using MPI IO to read data files in sac format and convert them to hdf5 format, then we have the data in hdf5 format to analysis. That's the first thing I'm planning to do. Basically First I am planning to refer to the existing serial code to convert different file formats and adapt it to the parallel MPI form.

> #### Time: From now to 11/04.

2. **Remove instrumental response**: 
>Generally the seismic stations store data in electric signal, and we have to convert them to the real seismic signal. That could be represented by a mathematical problem of deconvolution. Since deconvolution could be calculated firstly padding and then do convolution, this problem transfers to convolution problem.

>According to the julia package [DSP.jl](https://github.com/JuliaDSP/DSP.jl), they have the Algorithm implementation of convolution and deconvolution.

>As their code shows here:

>@[DSP.jl](https://github.com/JuliaDSP/DSP.jl/blob/f3c9382ba995d6d5d2b06e517c77c9d5b99e0c82/src/dspbase.jl)
```julia
"""
    deconv(b,a) -> c
Construct vector `c` such that `b = conv(a,c) + r`.
Equivalent to polynomial division.
"""
function deconv(b::StridedVector{T}, a::StridedVector{T}) where T
    lb = size(b,1)
    la = size(a,1)
    if lb < la
        return [zero(T)]
    end
    lx = lb-la+1
    x = zeros(T, lx)
    x[1] = 1
    filt(b, a, x)
end
"""
    conv(u,v)
Convolution of two vectors. Uses FFT algorithm.
"""
function conv(u::StridedVector{T}, v::StridedVector{T}) where T<:BLAS.BlasFloat
    nu = length(u)
    nv = length(v)
    n = nu + nv - 1
    np2 = n > 1024 ? nextprod([2,3,5], n) : nextpow(2, n)
    upad = [u; zeros(T, np2 - nu)]
    vpad = [v; zeros(T, np2 - nv)]
    if T <: Real
        p = plan_rfft(upad)
        y = irfft((p*upad).*(p*vpad), np2)
    else
        p = plan_fft!(upad)
        y = ifft!((p*upad).*(p*vpad))
    end
    return y[1:n]
end
```
> We could see the algorithm could be seprated into non-fft part and fft part. And I may use MPi and CUDA to implement the algorithm seprately in two parts. 
> #### time:from 11/04 to 11/07.

3. **Filter**: 
>Since seismic signal may contain some noise, usually we filter them before further analysis. There are lots of filtering algorithm and design, and I want to design the most widely used ones. Here I could also refer to the [DSP.jl](https://github.com/JuliaDSP/DSP.jl/tree/master/src/Filters), and make its algorithm parallel。

>Since I haven't seen how they implement the filtering algorithm, I couldn't give much more information about what exactly my code would look like. But in generous, there should be lots of numerical calculation, and we could always parallel them.
> #### time: 

> #### time:from 11/07 to 11/14. 
4. **resample**: 
>For our research, it's meaningful to resample the time series data using methods like interpolation. As for the interpolation algorithm, usually it's a linear algorithm problem, including solving some matrix functions which has very big dimension since our time series usually have a great amout of data points.

>So I'm planning to try some built in packages like BLAS or cuBLAS to see how to implement these algorithm. Or just wrap existing packages using MPI or CUDA to parallel them since usually if such the packages haven't linked with MKL, they couldn't be paralleled. I suppose I could have a try at this point.

> #### time: from 11/14 to 11/17

5. **Calculating SNR**: 
>The signal to noise ratio is useful for selecting good data that don't have too much noise, and in geophysics we usually first the data in time domain to frequency domain, and then select the base frequency of data. By calculating to energy of signal and noise in frequency domain, we are able to get the SNR.

>So that is just a problem of FFT and some other mathematical calculating. It should be easy to implement in MPI or CUDA.

> #### time: from 11/17 to 11/20

6. **Others**: 
>There are some other processing procedures existing, like the rotation of different components of seismic data. (Usually we have three components data to represent the movement for a seismic siganl receiver.), stacking different seismic waveform to reduce the noise, do correlation for different waveforms and etc.. They need just a little calculation and could easily be paralleled.

> #### time: from 11/20 to 11/23

7. **Order them one by one, wrap to packages, including unit test and document**: 
>If we order the processing procedures metioned above into streams like cuda stream, we may improve the efficency. And finally wrap them such like a python package may help us easilly to use them.

> #### time: from 11/23 to 11/26

## Summary

As for the consideration mentioned in the class webpage, I'd like to share my thinking about them:
1. >Combine two different parallel programming models: distributed memory (i.e., MPI), shared memory (i.e., OpenMP), GPUs (i.e., CUDA or OpenACC).

   >I am planning to use MPI (use mpi4py to wrap cython code) and CUDA (pycuda is a good choice and I have used it before).

2. >Explore different parallelization strategies (i.e., domain decomposition, task-based, etc.).

   >I suppose I could test both of them. The algorithms may be able to do domain decomposition. And since each seismic          signal is independent for most algorithm, I may also be able to do some task-based work.
   
3. >Develop a verification test to ensure the correctness of your solution. Ensure that the solution does not change with      the number of parallel tasks.

   >That's what my unit tests would like to do.
   
4. >Address load balancing and strategies for maintaining balance as tasks are increased.

   >That may be a problem I have to consider, maybe I could maintain a queue to do that.
   
5. >Address memory usage and how it scales with tasks for your problem.

   >My aim is to reduce the memory usage if possible, this should be a matter with the algorithm itself, and I could do some    tests to address that.
   
6. >Perform extensive scaling studies (i.e., weak, strong, thread-to-thread speedup). Your scaling studies should extend to    as large a number of tasks as you are able to with your problem.

   >I'll test it.
   
7. >All I/O should be handled with HDF5.

   >I think it's meaningful not only for the reason of Parallelization, but also HDF5 could represent data in the disk as if    they are in memory, which may be useful for coding.
   
   
## Expected Result

I'm planning to develop a Python package which provides both GPU and CPU parallelization which could be used for seismic signal processing. Also I'm planning to seprate the processing parts with the interface for seismic data reading, for further extending to a generic parallel digital signal processing package.
