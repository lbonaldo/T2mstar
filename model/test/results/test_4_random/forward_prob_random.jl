using Plots
using QuadGK
using Roots
using LaTeXStrings
using CSV
using DataFrames
#pyplot()

function f_export(idx::Int64, test_name)
    data = rand(2,6)

    a_pred,c_pred,a_true,c_true = rand(Float64, 4)
    b_pred,d_pred,b_true,d_true = rand(Float64, 4) .* sign.(randn(Float64, 4))
    betas = collect(0.05:0.01:5.0)
    β_pred = betas[rand(1:length(betas))]
    β_true = betas[rand(1:length(betas))]

    if abs(a_true) < abs(b_true)
        bs = sign(b_true)
        temp = a_true
        a_true = abs(b_true)
        b_true = bs*temp
    end
    
    if abs(c_true) < abs(d_true)
        cs = sign(d_true)
        temp = c_true
        c_true = abs(d_true)
        d_true = cs*temp
    end
    
    # no band crossing for this test
    if (d_true/c_true) < (-b_true/a_true)
        tmp, d_true = d_true, -b_true
        b_true = -tmp
        a_true, c_true = c_true, a_true
    end

    if abs(a_pred) < abs(b_pred)
        bs = sign(b_pred)
        temp = a_pred
        a_pred = abs(b_pred)
        b_pred = bs*temp
    end
    
    if abs(c_pred) < abs(d_pred)
        cs = sign(d_pred)
        temp = c_pred
        c_pred = abs(d_pred)
        d_pred = cs*temp
    end
    
    # no band crossing for this test
    if (d_pred/c_pred) < (-b_pred/a_pred)
        tmp, d_pred = d_pred, -b_pred
        b_pred = -tmp
        a_pred, c_pred = c_pred, a_pred
    end
    
    μ_t = rand(Float64)
    μ_true = (1-μ_t)*(-b_true/a_true) + μ_t*(d_true/c_true)
    μ_pred = (1-μ_t)*(-b_pred/a_pred) + μ_t*(d_pred/c_pred)

    function l(x::Float64, a::Float64, b::Float64)
        if -a*x - b > 0
            return sqrt(-a*x - b)
        else
            return 0
        end
    end

    function r(x::Float64, c::Float64, d::Float64)
        if c*x - d > 0
            return sqrt(c*x - d)
        else
            return 0
        end
    end

    bandcross = false

    gaussian_true(x::Float64) = sqrt(β_true/π)*exp(-β_true*(x - μ_true)^2)
    gaussian_pred(x::Float64) = sqrt(β_pred/π)*exp(-β_pred*(x - μ_pred)^2)

    xarr = -5:0.05:5
    fig = plot(xarr, r.(xarr, c_true, d_true), label=L"R\_true", c="blue", lw=2, xlabel="t", ylabel="L, R, G")
    plot!(fig, xarr, l.(xarr, a_true, b_true), label=L"L\_true", c="blue", lw=2)
    plot!(fig, xarr, r.(xarr, c_pred, d_pred), label=L"R\_pred", c="orange", lw=2)
    plot!(fig, xarr, l.(xarr, a_pred, b_pred), label=L"L\_pred", c="orange", lw=2)

    l_min_true(x) = l(x, a_true, b_true) - gaussian_true(x)
    r_min_true(x) = r(x, c_true, d_true) - gaussian_true(x)
    l_min_pred(x) = l(x, a_pred, b_pred) - gaussian_pred(x)
    r_min_pred(x) = r(x, c_pred, d_pred) - gaussian_pred(x)
    #@show a_true,c_true,b_true,d_true,μ_true,β_true
    #@show a_pred,c_pred,b_pred,d_pred,μ_pred,β_pred

    x0_true = find_zero(l_min_true, (-(β_true + pi*b_true)/(pi*a_true), -b_true/a_true), Bisection())
    x3_true = find_zero(r_min_true, (d_true/c_true, (β_true + pi*d_true)/(pi*c_true)), Bisection())
    x0_pred = find_zero(l_min_pred, (-(β_pred + pi*b_pred)/(pi*a_pred), -b_pred/a_pred), Bisection())
    x3_pred = find_zero(r_min_pred, (d_pred/c_pred, (β_pred + pi*d_pred)/(pi*c_pred)), Bisection())

    # Perform the gaussian integrals. These are independent of whether the bands cross.
    A_true = quadgk(gaussian_true, -Inf, x0_true)[1]
    B_true = quadgk(gaussian_true, x3_true, Inf)[1]
    A_pred = quadgk(gaussian_pred, -Inf, x0_pred)[1]
    B_pred = quadgk(gaussian_pred, x3_pred, Inf)[1]

    # Now for the other integrals, which do depend on whether the bands cross.
    # I solved these integrals analytically, so only the result is here.
    if !bandcross
        A_true += (2/(3a_true)) * (-a_true*x0_true - b_true)^1.5
        B_true += (2/(3c_true)) * (c_true*x3_true - d_true)^1.5
    else # I apoligize in advance, this algebra is ugly
        A_true += (2/(3a_true)) * ((-a_true*x0_true - b_true)^1.5 - (-(a_true*d_true + b_true*c_true)/(a_true+c_true))^1.5)
        B_true += (2/(3c_true)) * ((-(a_true*d_true + b_true*c_true)/(a_true+c_true))^1.5 - (c_true*x3_true-d_true)^1.5)
    end
    if !bandcross
        A_pred += (2/(3a_pred)) * (-a_pred*x0_pred - b_pred)^1.5
        B_pred += (2/(3c_pred)) * (c_pred*x3_pred - d_pred)^1.5
    else # I apoligize in advance, this algebra is ugly
        A_pred += (2/(3a_pred)) * ((-a_pred*x0_pred - b_pred)^1.5 - (-(a_pred*d_pred + b_pred*c_pred)/(a_pred+c_pred))^1.5)
        B_pred += (2/(3c_pred)) * ((-(a_pred*d_pred + b_pred*c_pred)/(a_pred+c_pred))^1.5 - (c_pred*x3_pred-d_pred)^1.5)
    end


    #Adds the gaussian and intersection points to the plot
    plot!(fig, xarr, gaussian_true.(xarr), label=L"g\_true", c="blue", lw=2)
    plot!(fig, xarr, gaussian_pred.(xarr), label=L"g\_pred", c="orange", lw=2)
    if !bandcross
        plot!(fig, [x0_true, -b_true/a_true, d_true/c_true, x3_true], [l(x0_true,a_true,b_true), l(-b_true/a_true,a_true,b_true), r(d_true/c_true, c_true, d_true), r(x3_true, c_true, d_true)], seriestype=:scatter, label="Intersections", lw=3)
    else
        plot!(fig, [x0_true, (d_true-b_true)/(a_true+c_true), x3_true], [l(x0_true,a_true,b_true), sqrt(-(a_true*d_true + b_true*c_true)/(a_true + c_true)), r(x3_true, c_true, d_true)], seriestype=:scatter, label="Intersections", lw=3)
    end
    if !bandcross
        plot!(fig, [x0_pred, -b_pred/a_pred, d_pred/c_pred, x3_pred], [l(x0_pred,a_pred,b_pred), l(-b_pred/a_pred,a_pred,b_pred), r(d_pred/c_pred,c_pred,d_pred), r(x3_pred,c_pred,d_pred)], seriestype=:scatter, label="Intersections", lw=3)
    else
        plot!(fig, [x0_pred, (d_pred-b_pred)/(a_pred+c_pred), x3_pred], [l(x0_pred,a_pred,b_pred), sqrt(-(a_pred*d_pred + b_pred*c_pred)/(a_pred + c_pred)), r(x3_pred,c_pred,d_pred)], seriestype=:scatter, label="Intersections", lw=3)
    end
    xpos = -1.8
    ypos = 1.7
    annotate!(xpos, ypos, text(L"I\ true = "*string(round(B_true-A_true,digits=5)), :black, :left, 8))
    annotate!(xpos, ypos+0.12, text(L"I\ pred = "*string(round(B_pred-A_pred,digits=5)), :black, :left, 8))
    #display(fig)
    plots_path = "./model/test/results/"*test_name*"/plots"
    if !isdir(plots_path)
        mkdir(plots_path)
    end
    savefig(plots_path*"/jplot_$idx.png")
end

test_name = "test_4_random"
idxs = 0:1:20
for i in idxs
    f_export(i, test_name) 
end