using LsqFit
@. model(x, p) = p[1]*exp(-x*p[2])
xdata = range(0, stop=10, length=20)
ydata = model(xdata, [1.0 , 2.0]) + 0.01*randn(length(xdata))
p0 = [0.5, 0.5];

fit = curve_fit(model, xdata, ydata, p0)
lb = [1.1, -0.5]
ub = [1.9, Inf]
p0_bounds = [1.2, 1.2] # we have to start inside the bounds
# Optional upper and/or lower bounds on the free parameters can be passed as an argument.
# Bounded and unbouded variables can be mixed by setting `-Inf` if no lower bounds
# is to be enforced for that variable and similarly for `+Inf`
fit_bounds = curve_fit(model, xdata, ydata, p0_bounds, lower=lb, upper=ub)

# We can estimate errors on the fit parameters,
# to get standard error of each parameter:
sigma = stderror(fit)
# to get margin of error and confidence interval of each parameter at 5% significance level:
margin_of_error = margin_error(fit, 0.05)
confidence_inter = confidence_interval(fit, 0.05)

# The finite difference method is used above to approximate the Jacobian.
# Alternatively, a function which calculates it exactly can be supplied instead.
function jacobian_model(x,p)
    J = Array{Float64}(undef, length(x), length(p))
    @. J[:,1] = exp(-x*p[2])     #dmodel/dp[1]
    @. @views J[:,2] = -x*p[1]*J[:,1] #dmodel/dp[2], thanks to @views we don't allocate memory for the J[:,1] slice
    J
end

fit = curve_fit(model, jacobian_model, xdata, ydata, p0)
