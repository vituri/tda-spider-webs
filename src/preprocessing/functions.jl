
using Images, ImageFiltering, ImageTransformations
using Random
using MetricSpaces
using ..TDAweb

function crop_image(img; threshold = 0.8)
    ids = findall_ids(<(threshold), img)
    x1, x2 = extrema(first.(ids))
    y1, y2 = extrema(last.(ids))
    img[x1:x2, y1:y2]
end

function load_web(path::AbstractString; blur = 1)
    img = imfilter(load(path), Kernel.gaussian(blur)) .|> Gray
    resize_image(img, pixels = 500)
end

function resize_image(img; pixels = 150)
    ratio = pixels / size(img)[1]
    imresize(img, ratio = ratio)
end

function image_to_array(img::Matrix)
    A = convert(Array{Float32}, img)
    A .|> (x -> 1-x)        
end

function image_to_r2(img::Matrix; threshold = 0.5)
    A = image_to_array(img)
    findall_ids(>(threshold), A) |> EuclideanSpace
end
