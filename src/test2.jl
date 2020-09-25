using LsqFit

@. model(x, p) = p[1]*exp(-x*p[2])

xdata = range(0, stop=10, length=20)
ydata = model(xdata, [1.0 , 2.0]) + 0.01*randn(length(xdata))
p0 = [0.5, 0.5];

fit_auto = curve_fit(model, xdata, ydata, p0; autodiff=:forwarddiff)
println(fit_auto)
