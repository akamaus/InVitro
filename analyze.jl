using HDF5

function read_electrodes()
    data_path = "./2014_03_20_0001.h5"
    df = h5open(data_path,"r")
    df["Data/Recording_0/AnalogStream/Stream_0/ChannelData"]
end

function detect_spikes1(ch; prec = 2)
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

using DSP

function band_filter(low, high; fs = 25000)
    responsetype = Bandpass(low, high; fs=fs)
    prototype = Butterworth(8)
    digitalfilter(responsetype, prototype)
end

function arr_disp(xs)
    vcat(xs[2:end],xs[end])
end

typealias SigVal Int32
typealias Time Int32

function find_minima(xs :: Vector{SigVal}; bound = typemax(SigVal))
    diffs = arr_disp(xs) - xs
    signs = arr_disp(diffs) .* diffs

    mins = (Time,SigVal)[]

    for i=2:length(signs)-1
        if signs[i] < 0
            if diffs[i] < 0 && xs[i+1] < bound
                push!(mins,(i,xs[i+1]))
            end
        end
    end
    return mins
end

function detect_spikes2(ch; prec = 2)
    mean_p = mean(ch)
    noise = std(ch) * prec

    find_minima(ch; bound = mean_p - noise)
end

function slice(xs, i; width = 125)
    xs[ i - width : i + width]
end

function draw_neighs(ch, points; width = 125)
    xs = if typeof(points) == Vector{Time}
        points
    else
        [p[1] for p in points]
    end

    n = length(xs)
    rows = int(ceil(sqrt(n)))
    cols = int(ceil(n / rows))

    for i = 1:n
        subplot(rows,cols, i)
        title("Spike-$(xs[i])")
        plot(slice(ch, xs[i], width = width))
    end
    ;
end

function similarity_matrix(ch, points; F=square_dist)
    xs = if typeof(points) == Vector{Time}
        points
    else
        [p[1] for p in points]
    end

    snippets = [ slice(ch, x) for x in xs]
    matrix = corr_matrix(snippets,F=F)
    imshow(matrix)
    matrix
end

function corr_matrix(mat,F=cor)
    n = size(mat,2)
    corrs = zeros(Float32, n, n)

    for i=1:n
        for j=1:n
            corrs[i,j] = F(mat[:,i],mat[:,j])
        end
    end
    imshow(corrs)
    corrs
end

function square_dist(sig1, sig2)
    sqrt(sum((sig1 .- sig2) .^ 2))
end

function detect_dead_electrodes(snippet)
    corrs = corr_matrix(snippet)
    sum_corrs = sum(corrs,1)
    dead = find(x->abs(x - mean(sum_corrs)) > std(sum_corrs), sum_corrs)
end
    
