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
        
    
