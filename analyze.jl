using HDF5

function read_electrodes()
    data_path = "./2014_03_20_0001.h5"
    df = h5open(data_path,"r")
    df["Data/Recording_0/AnalogStream/Stream_0/ChannelData"]
end

function detect_spikes(ch; prec = 2)
    mean_p = mean(ch)
    noise = std(ch) * prec

    spikes = (Int32,Int32)[]

    for i=1:length(ch)
        v = ch[i]
        if (abs(v - mean_p) > noise)
            push!(spikes, (i, v))
        end
    end

    spikes
end

function analize()
    info("Reading electrodes")
    channels = read_electrodes()

    for i=1:size(channels,2)
        ch = channels[:,i] :: Array{Int32,2}
        ch = reshape(ch, length(ch))
        spikes = detect_spikes(ch)
        @printf("channel %u; spikes = %u", i, length(spikes))
    end
end

function plot_stats(xs; slice=20000)
    slices = reshape(xs, div(length(xs),slice), slice)
    s_mean = mean(slices,2)
    s_std = std(slices,2)
    plot(layer(x=1:length(s_mean), y = s_mean, Geom.line),
         layer(x=1:length(s_std), y = s_std, Geom.line))
end

function anomaly_filter(xs, x_mean, x_deviation)
    sieved = (Int64,Int32)[]
    for x in xs
        p,v = x
        if (abs(v - x_mean) > x_deviation)
            push!(sieved, x)
        end
    end
    sieved
end

function anomaly_hist(ch)
    mean_p = mean(ch)
    noise = std(ch)
    bins = Int64[]

    n=0
    pairs = map(ch) do x
        global n
        n += 1
        (n, x)
    end

    k=1
    while length(pairs) > 0
        push!(bins,length(pairs))
        @printf("len_pairs %u\n", length(pairs))
        pairs = anomaly_filter(pairs, mean_p, noise*k)
        k += 1
    end

    plot(x=1:k-1, y=bins)
    bins
end

function wave_export()
    channels = read_electrodes()
    n_channels = length(channels[1,:])
    for c in (1:n_channels)
        ch = channels[:, c]
        ch = reshape(ch, length(ch))
        file = "ch_$c.wav"
        print(file)
        wavwrite(ch, file,Fs=22050,nbits=16)
    end
end

function get_snippet(t0, len)
    channels = read_electrodes()
    n_channels = size(channels,2)
    snippet = Array(Int32, len, n_channels)
    for c in (1:n_channels)
        snippet[:,c] = channels[t0:t0+len-1,c]
        @printf("got channel %u\n", c)
    end
    snippet
end

using PyPlot

function draw_snippets(snippets; num_cols=3)
    for i=1:size(snippets,2)
        subplot(div(size(snippets,2), num_cols), num_cols, i)
        title("ch$i")
        plot(snippets[:,i])
    end
end

#
# DSP
#

function lowpass_kernel(freq; size = 1000, sample_rate=25000)
    t = [-size / 2 : size / 2] / sample_rate
    sinc_samples = sinc(2*freq*t)
    sinc_samples
end

function hamming_window(;size=1000)
    i = [0:size]
    hamming_w = 0.54 - 0.46 * cos(2 * pi * i / size)
end

function delta(;size=1000)
    delta_w = zeros(Float32, size + 1)
    delta_w[size/2 + 1] = 1.0
    delta_w
end

function highpass_kernel(freq;  size = 1000, sample_rate=25000)
    delta(size=size) - lowpass_kernel(freq, size=size, sample_rate=sample_rate)
end
