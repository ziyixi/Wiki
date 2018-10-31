# Develop a parallel signal processing tool for seismic data analysis
### Ziyi Xi

## Brief Introduction

Seismic signals are usually transient waveforms radiated from a localized natural or manmade
seismic source. They can be used to locate the source, to analyze source processes, and
to study the structure of the medium of propagation. But now almost all the processing tools are not paralleled, which may reduce the effiency to analysis this kind of digital signal, so I'm planning to develop a parallel processing tool to handle this problem.

## The connection between this project and our class

Here I'd like to discuss each part of seismic signal processing, and how I could connect each part with the parallel computing.

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

3. **Filter**: Since seismic signal may contain some noise, usually we filter them before further analysis. There are lots of filtering algorithm and design, and I just to design the most widely used ones. Here I could also refer to the [DSP.jl](https://github.com/JuliaDSP/DSP.jl/tree/master/src/Filters), and make its algorithm parallel。

For the algorithm itself, I could also refer to [FIR](https://en.wikipedia.org/wiki/Finite_impulse_response) in wikipedia. 
Anyway, there are lots of existing reference and I could refer them to write the code.

> #### time:from 11/07 to 11/14. 
4. **resample**: data from different stations 
