module Utils


using Dates
using Random

export  checkerange!, create_runfolder
        
# if range is not a range (e.g. single value) convert it to a 1-el range
function checkerange!(erange)
    if (typeof(erange) <: AbstractArray)                # if it's a vector (e.g. more than one band)
        for i in eachindex(erange)
            if !(typeof(erange[i]) <: AbstractRange)    # but it's element are not all ranges
                erange[i] = erange[i]:erange[i]         # convert them to a 1-el range
            end
        end
    else                                        # if it's not a vector (e.g. single band)
        if !(typeof(erange) <: AbstractRange)   # but it's element are not all ranges
            erange = erange:erange              # convert them to a 1-el range
        end
    end
    return erange
end


# function that creates folders for each test perfomed by RMC
# tests are organized as follows:
# results
#   └─── date_of_the_test
#           └─── time_of_the_test
function create_runfolder(script_path::String=".")
    run_path = joinpath(script_path,"results")
    if !isdir(run_path) # if there is not a results folder -> create it
        mkdir(run_path)
    end
    date = Dates.format(Dates.today(),"mm-dd-YYYY")
    run_path = joinpath(run_path,date)
    if !isdir(run_path) # if there is not a today folder -> create it
        mkdir(run_path)
    end
    test_name = Dates.format(now(), "HH-MM-SS")
    run_path = joinpath(run_path,test_name*"-"*randstring(4))
    mkdir(run_path)
    return run_path
end

end


