using Random
using PyCall
np = pyimport("numpy")

p = 5 # x,y,z

gaussian(x) = sqrt(3/π)*exp(-3*(x - 2)^2)

function f14(x::Float64)
    f1 = sin(x)
    f2 = (x+23)/(27)
    f3 = cos(2x)
    f4 = x^2/9
    return cat(f1,f2,f3,f4,dims=1)
end


function f15(x::Float64)
    f1 = sin(x)
    f2 = (x+23)/(27)
    f3 = cos(2x)
    f4 = x^2/9
    f5 = -log(1/(1+x))
    return cat(f1,f2,f3,f4,f5,dims=1)
end

function f16(x::Float64)
    f1 = sin(x)
    f2 = (x+23)/(27)
    f3 = cos(2x)
    f4 = x^2/9
    f5 = -log(1/(1+x))
    f6 = gaussian(x)
    return cat(f1,f2,f3,f4,f5,f6,dims=1)
end


function f35(x::Float64,y::Float64,z::Float64)
    f1 = sin(x+y)
    f2 = x+y/(z+23)
    f3 = cos(2x)sin(5z)
    f4 = 2x + 1/(z+1) + 5y
    f5 = log(1/(y+4))
    return cat(f1,f2,f3,f4,f5,dims=1)
end

function f36(x::Float64,y::Float64,z::Float64)
    f1 = sin(x+y)
    f2 = x+y/(z+23)
    f3 = cos(2x)sin(5z)
    f4 = 2x + 1/(z+1) + 5y
    f5 = log(1/(y+4))
    f6 = 7gaussian(x) + 2gaussian(y)
    return cat(f1,f2,f3,f4,f5,f6,dims=1)
end

function f39(x::Float64,y::Float64,z::Float64)
    f1 = sin(x+y)
    f2 = 3exp(2z)+y
    f3 = cos(2x)sin(5z)
    f4 = 2x + 1/(z+1) + 5y
    f5 = log(1/(y+4))
    f6 = 7gaussian(x) + 2gaussian(y)
    f7 = 14tanh((1+x)/(z+12))
    f8 = x+y/(z+23)
    f9 = 5x - sin(z+5y)
    return cat(f1,f2,f3,f4,f5,f6,f7,f8,f9,dims=1)
end

function f310(x::Float64,y::Float64,z::Float64)
    f1 = sin(x+y)
    f2 = 3exp(2z)+y
    f3 = cos(2x)sin(5z)
    f4 = 2x + 1/(z+1) + 5y
    f5 = log(1/(y+4))
    f6 = 7gaussian(x) + 2gaussian(y)
    f7 = 14tanh((1+x)/(z+12))
    f8 = x+y/(z+23)
    f9 = 5x - sin(z+5y)
    f10 = cosh(x-z) + 56 
    return cat(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,dims=1)
end

function f55(x::Float64,y::Float64,z::Float64,t::Float64,s::Float64)
    f1 = sin(x+y) + 7gaussian(t) 
    f2 = 3exp(2z)+y + 5t - sin(s+5x)
    f3 = cos(2x)sin(5z) + log(1/(t+4))
    f4 = 2x + 1/(z+1) + 5y +  cosh(t)sinh(s)
    f5 = 14tanh((1+x)/(z+12)) + t+z/(s+23)
    return cat(f1,f2,f3,f4,f5,dims=1)
end

xarr = 1:0.000001:3
yarr = 1:0.000001:3
zarr = 1:0.000001:3
# tarr = 0:0.05:10000000
# sarr = 0:0.05:10000000

index = Vector(1:length(xarr))
# shuffle!(MersenneTwister(1234), index)

# test_perc = 0.2
# train_perc = 1-0.2
# val_perc = 0.2*train_perc
# train_perc -= val_perc

# train_size = Int64(floor(train_perc*length(xarr)))
# index_train = index[1:train_size]
# val_size = Int64(floor(val_perc*length(xarr)))
# index_val = index[train_size+1:train_size+val_size]
# test_size = Int64(floor(test_perc*length(xarr)))
# index_test = index[train_size+val_size+1:train_size+val_size+test_size]

# println("train_size: ", length(index_train))
# println("val_size: ", length(index_val))
# println("test_size: ", length(index_test))
# println("TOTAL: ", length(index_train)+length(index_val)+length(index_test))

# # # 6 - 4
n = 6

# # train
# x_train = Array{Float64,2}(undef, length(index_train), n)
# y_train = Array{Float64,2}(undef, length(index_train), n-2)

# for (j,i) in enumerate(index_train)
#     x_train[j,:] = f16(xarr[i])
#     y_train[j,:] = f14(xarr[i])
# end
# data = np.asarray(x_train)
# np.save("./data/data_64/x_train.npy",data)
# data = np.asarray(y_train)
# np.save("./data/data_64/y_train.npy",data)

# # valid
# x_val = Array{Float64,2}(undef, length(index_val), n)
# y_val = Array{Float64,2}(undef, length(index_val), n-2)

# for (j,i) in enumerate(index_val)
#     x_val[j,:] = f16(xarr[i])
#     y_val[j,:] = f14(xarr[i])
# end
# data = np.asarray(x_val)
# np.save("./data/data_64/x_val.npy",data)
# data = np.asarray(y_val)
# np.save("./data/data_64/y_val.npy",data)

# test
x_test = Array{Float64,2}(undef, length(index), n)
y_test = Array{Float64,2}(undef, length(index), n-2)
for (j,i) in enumerate(index)
    x_test[j,:] = f16(xarr[i])
    y_test[j,:] = f14(xarr[i])
end
data = np.asarray(x_test)
np.save("./data/data_64/x_test.npy",data)
data = np.asarray(y_test)
np.save("./data/data_64/y_test.npy",data)
