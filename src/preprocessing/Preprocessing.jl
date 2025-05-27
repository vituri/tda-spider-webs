module Preprocessing

using ..TDAweb

include("functions.jl");
export load_web, 
    image_to_r2, 
    image_to_array, 
    resize_image;    
end