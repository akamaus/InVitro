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

# Возвращает матрицу - срез сигнала по всем электродам во времени [t0..t0+len]
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

function find_maxima{T <: Real}(xs :: Vector{T}; bound = typemin(SigVal))
    diffs = arr_disp(xs) - xs
    signs = arr_disp(diffs) .* diffs

    maxs = (Time,T)[]

    for i=2:length(signs)-1
        if signs[i] < 0
            if diffs[i] > 0 && xs[i+1] > bound
                push!(maxs,(i,xs[i+1]))
            end
        end
    end
    return maxs
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
        if isa(ch, Array{Int32,2})
            for ci in 1:size(ch,2)
                plot(slice(ch[:,ci], xs[i], width = width))
            end
        else
            plot(slice(ch, xs[i], width = width))
        end
    end
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

function energy(sig)
    sum(sig .^2)
end

function downsample(signal, block; F=mean)
    d_size = div(length(signal), block)
    low_res = Array(Float64, d_size)
    for i=1:d_size
        low_res[i] = F(signal[1 + block*(i-1) : block*i])
    end
    low_res
end

function enlarge_while(xs, i :: Time, P)
    a = i :: Time
    b = int32(i+1) :: Time

    while a > 1 && P(xs[a])
        a -= 1
    end

    while b < length(xs) && P(xs[b])
        b += 1
    end
    (int32(a),int32(b))
end

function grow_intervals{T}(small_ints :: Vector{(T, T)})
    big_ints = (T, T)[]
    i=1
    local a,b
    while i <= length(small_ints)
        a,b = small_ints[i]
        i += 1
        while i <= length(small_ints)
            x1,x2 = small_ints[i]
            grew = false
            if b+1 >= x1
                a = min(a,x1)
                b = max(b,x2)
                grew = true
            end
            if !grew
                push!(big_ints,(a,b))
                break
            end
            i += 1
        end
    end
    push!(big_ints,(a,b))

    big_ints
end

# Ищет точки с максимальной энергией на загрублённом сигнале, возвращает их окрестности
function find_energy_maxima(ch; block=100, low_peak = 0.01, draw = false)
    ch_energy = downsample(ch, block, F=energy)
    if draw
        plot(ch_energy)
        yield
    end
    low_peak_bound = - select( - ch_energy, int(low_peak * length(ch_energy)))
    println("low_peak $low_peak_bound")
    peaks = find_maxima(ch_energy, bound = low_peak_bound) :: Vector{(Time, Float64)}
    if draw
        for p in peaks
            PyPlot.axvspan(p[1]-1, p[1]+1,0, 0.05, color="green")
        end
    end

    hot_elementary_zones = [ enlarge_while(ch_energy, x, v-> v > low_peak_bound) for (x,p) in peaks] :: Vector{(Time,Time)}
    hot_zones = grow_intervals(hot_elementary_zones)
#    return hot_zones

    for (a,b) in hot_zones
        PyPlot.axvspan(a,b, 0, 0.05, color="red")
    end

    #hot_zones
    map(r->scale_range(r,block), hot_zones)
end

function signal_to_noise_ratio(ch; block = 100, low_peak = 0.01, draw = false)
    ch_energy = downsample(ch, block, F=energy)
    if draw
        plot(ch_energy)
        yield
    end
    low_peak_bound = - select( - ch_energy, int(low_peak * length(ch_energy)))
    if draw
        axhline(y = low_peak_bound, color = "red")
        yield
    end
    mean_p = - select( - ch_energy, int(length(ch_energy) / 2))
    if draw
        axhline(y = mean_p, color = "blue")
        yield
    end
    low_peak_bound / mean_p
end

function signal_to_noise_ratio_rude(ch; block = 100, low_peak = 0.01, draw = false)
    ch_energy = downsample(ch, block, F=energy)
    if draw
        plot(ch_energy)
        yield
    end
    low_peak_bound = maximum(ch_energy)
    if draw
        axhline(y = low_peak_bound, color = "red")
        yield
    end
    mean_p = minimum(ch_energy)
    if draw
        axhline(y = mean_p, color = "blue")
        yield
    end
    low_peak_bound / mean_p
end


function for_each_channel(F, snippet)
    res = Any[]
    ch_range = 1:size(snippet,2)
    for ci in ch_range
        t1 = time()
        ch = snippet[:, ci]
        push!(res, F(ch))
        t2 = time()
        if t2 - t1 > 1
            @printf("got res[%u] = %s\n", ci, sprint(print, res[ci]))
        end
    end
    res
end

scale_range{T<:Time,K<:Real}(r::(T,T), k::K) = (convert(T,r[1]*k), convert(T,r[2]*k))

# Рисует избранные куски сигнала на разных графиках
function draw_snippets{T}(ch, ranges :: Vector{(T,T)}; num_cols=3, gap = 100)
    n = length(ranges)
    for i=1:n
        subplot(ceil(n/num_cols), num_cols, i)
        r = ranges[i]
        l_bound = max(r[1] - gap,1)
        r_bound = min(r[2] + gap,length(ch))
        plot(ch[l_bound:r_bound])
        PyPlot.axvspan(r[1] - l_bound, r_bound-l_bound - min(length(ch) - r[2],gap) , 0, 0.02, color="red")
        yield
    end
end

# рисует график, помечает зоны на оси абсцисс
function draw_ranges(ch, ranges; color="red", height=0.05)
    plot(ch)
    for r in ranges
        PyPlot.axvspan(r[1], r[2],0, height, color=color)
    end
end

# поиск спайков с учетом динамически вычисляемого порога. Он определяется путём выделения заданной доли самых амплитудных точек
function detect_spikes3(ch; ratio=0.01, prec = 3, draw = false)
    bound = select(ch, int(ratio * length(ch)))
    if draw
        plot(ch)
        yield
    end
    mins = find_minima(ch; bound = bound * prec)
    if draw
        for m in mins
            PyPlot.axvspan(m[1]-1, m[1]+1,0, 0.05, color="red")
        end
    end
    mins
end

# подсчет межспайковых интервалов
function calc_isi(spikes :: Vector{Time}; draw = false)
    intervals = spikes[2:end] .- spikes[1:end-1]
    if draw
        plot(sort(intervals))
    end
    intervals
end

# поиск групп событий
function detect_bursts(spikes :: Vector{Time}, max_isi)
    local i=1, j;
    bursts = (Time,Time)[]
    L = length(spikes)
    while i < L
        j = i+1
        while j <= L && (spikes[j] - spikes[i]) / (j - i + 1) < max_isi
            j += 1
        end
        if j > i+1
            push!(bursts, (spikes[i],spikes[j-1]))
        end
        i = j
    end
    bursts
end

#ints = calc_isi(spikes)
    
