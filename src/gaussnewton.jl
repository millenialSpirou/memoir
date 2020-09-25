using ForwardDiff
using LinearAlgebra

struct algorithmic_parameters
    max_iter::Int
    tolerance::Float64
end


function fit(f, initguess, x, alg_param::algorithmic_parameters)

    if initguess == nothing:
        error("No initial guess was provided")
    end
   
    coefficients = initguess
        for k in 1:alg_param.max_iter:
            jacobian = x -> ForwardDiff.jacobian(f,x)
            residual = f(x, coefficients) -. y
            coefficients = coefficients  .- pinv(jacobian) * residual #pinv is pseudoinverse
            rmse = sqrt(sum(residual.^2))

            if rmse < alg_param.tolerance:
                return coefficients
            end
           
        end
end


#tests
function f(x, coefficients)
    a = coefficients
    return a[1]*x.^3 + a[2]*x.^2 + a[3]*x + a[4] + a[5]*sin(x)
end


function main()
    x = collect(range(0,stop=100,length=100))
    coefficients = [-0.001, 0.1, 0.1, 2, 15]

    y = f(x, coefficients)

    algorithmic_param = algorithmic_parameters(100,10^(-6))
    init_guess = 1000000 * rand(Float64, len(coefficients))
end
