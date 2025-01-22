from sympy import simplify, symbols

s, b, h, h_ffn, L, pa, k, hc, v, t, d, c, p = symbols("s b h h_ffn L pa k hc v t d c p")

qkvo = 2*h**2 + 2*h**2*(k/hc)
norm = 2*s ## negligible
ffnn = 2*h*h_ffn
num_parameters_att = qkvo + norm + ffnn
patchify = pa**3 * h ## TODO: Change pa to cubed? 
final_norm = 2*s
num_parameters_vit = (num_parameters_att * L/p + patchify + final_norm) / t

att = (1 + 4/t) * b*s*h   # x, qkvo (activation of x is not parallelized)
norm = 2*b*s*h # need two because standardization, element-wise linear but easy to recompute? Am I using RMSNorm or LayerNorm?
ffnn = b*s*h + b*s*h_ffn/t ## only the intermediate input is paralleized by TP
num_act_elements = att + ffnn + norm ## 
num_bytes = 2
activation_transformer = num_bytes*num_act_elements #+ dropout
num_patchify = b*s*pa**2
## you are missing the last activation layer? No, negiligible only one head...
activation_vit_specific = num_bytes*num_patchify
activation_vit = (activation_transformer * L + activation_vit_specific) / (d * c)

## NOTE: Assuming ZeRO3
term1 = 18 / (d*c) * num_parameters_vit 
term2 = activation_vit
result = term1 + term2


simplified_term1 = simplify(term1)
simplified_term2 = simplify(term2)

print(f"simplified_term1: {simplified_term1}")
print(f"simplified_term2: {simplified_term2}")