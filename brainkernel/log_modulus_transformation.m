function x = log_modulus_transformation(x)

signf = sign(x);
x = signf.*log(abs(x)+1);