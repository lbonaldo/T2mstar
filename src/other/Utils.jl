module Utils


export  checkerange!
        
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


end


