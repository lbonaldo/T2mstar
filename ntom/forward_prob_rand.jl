using Random
using PyCall
np = pyimport("numpy")

p = 5 # x,y,z
n = 10

gaussian(x) = sqrt(3/π)*exp(-3*(x - 2)^2)

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
    return cat(f1,f2,f3,f4,f5,f6,f7,f8,f9, dims=1)
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
    return cat(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10, dims=1)
end

function f55(x::Float64,y::Float64,z::Float64,t::Float64,s::Float64)
    f1 = sin(x+y) + 7gaussian(t) 
    f2 = 3exp(2z)+y + 5t - sin(s+5x)
    f3 = cos(2x)sin(5z) + log(1/(t+4))
    f4 = 2x + 1/(z+1) + 5y +  cosh(t)sinh(s)
    f5 = 14tanh((1+x)/(z+12)) + t+z/(s+23)
    return cat(f1,f2,f3,f4,f5, dims=1)
end

xarr = 0:0.05:1000000
yarr = 0:0.05:1000000
zarr = 0:0.05:1000000
# tarr = 0:0.05:10000000
# sarr = 0:0.05:10000000

index = Vector(1:length(xarr))
shuffle!(MersenneTwister(1234), index)

test_perc = 0.2
train_perc = 1-0.2
val_perc = 0.2*train_perc
train_perc -= val_perc

train_size = Int64(floor(train_perc*length(xarr)))
index_train = index[1:train_size]
val_size = Int64(floor(val_perc*length(xarr)))
index_val = index[train_size+1:train_size+val_size]
test_size = Int64(floor(test_perc*length(xarr)))
index_test = index[train_size+val_size+1:train_size+val_size+test_size]

println("train_size: ", length(index_train))
println("val_size: ", length(index_val))
println("test_size: ", length(index_test))
println("TOTAL: ", length(index_train)+length(index_val)+length(index_test))

# 10 - 9
# train
x_train = Array{Float64,2}(undef, length(index_train), n)
y_train = Array{Float64,2}(undef, length(index_train), n-1)

for (j,i) in enumerate(index_train)
    x_train[j,:] = f310(xarr[i],yarr[i],zarr[i])
    y_train[j,:] = f39(xarr[i],yarr[i],zarr[i])
end
data = np.asarray(x_train)
np.save("ntom\\data_1\\x_train.npy",data)
data = np.asarray(y_train)
np.save("ntom\\data_1\\y_train.npy",data)

# valid
x_val = Array{Float64,2}(undef, length(index_val), n)
y_val = Array{Float64,2}(undef, length(index_val), n-1)

for (j,i) in enumerate(index_val)
    x_val[j,:] = f310(xarr[i],yarr[i],zarr[i])
    y_val[j,:] = f39(xarr[i],yarr[i],zarr[i])
end
data = np.asarray(x_val)
np.save("ntom\\data_1\\x_val.npy",data)
data = np.asarray(y_val)
np.save("ntom\\data_1\\y_val.npy",data)

# test
x_test = Array{Float64,2}(undef, length(index_test), n)
y_test = Array{Float64,2}(undef, length(index_test), n-1)
for (j,i) in enumerate(index_test)
    x_test[j,:] = f310(xarr[i],yarr[i],zarr[i])
    y_test[j,:] = f39(xarr[i],yarr[i],zarr[i])
end
enddata = np.asarray(x_test)
np.save("ntom\\data_1\\x_test.npy",data)
data = np.asarray(y_test)
np.save("ntom\\data_1\\y_test.npy",data)