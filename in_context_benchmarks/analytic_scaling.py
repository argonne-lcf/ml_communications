# iPort torch
import argparse

## TODO: Implement combination of different parallelisms
## TODO: Implement head count asserts for ulysses, TP, etc. 
## TODO: We get 53B not 70B. Reconcile this? Layernorm, swiglu, etc. Does this also suggest that our TFlop formula is also underestimated? 
## TODO: Implement peak memory size. Also memory saving compared to DP?

## Lamma 70B
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", "-B", type=int, default=1, help="") ## Q. What was the global batch size in Llama-3 80B?
parser.add_argument("--seq_length", "-s", type=int, default=128_000, help="") ## Q. 128K for llamma3 pretraining? https://arxiv.org/pdf/2407.21783 
                                                                              ## "In the final stages of pre-training, we train on long sequences to support context windows of up to 128K tokens"
parser.add_argument("--hidden_size", "-hid", type=int, default=8192, help="")
parser.add_argument("--ffn_hidden_size", "-hid_", type=int,default=28672,  help="")
parser.add_argument("--num_layers", "-l", type=int, default=80, help="")
# parser.add_argument("-hc", "--num_heads", type=int, default=16, help="")
parser.add_argument("--parallelism", "-P", type=int, default=16, help="model parallelism degree") ## Arbitrary default value
args = parser.parse_args()

B = args.batch_size ## Global Batch Size
s = args.seq_length ## sequence length
h = args.hidden_size ## emb hidden dimension
h_ = args.ffn_hidden_size ## ffnn hidden dimension
l = args.num_layers ## num att. layer
P = args.parallelism

def get_num_flop_per_layer(B, s, h, h_):
    '''
        Get total flop count of the transformer layers.
        Note: 
            - Flop count is solely dominated by Matrix multiplication.
            - We disregard layers before or after transformer layers such as patchification (VIT), Classification linear head (LLM), etc.
    '''
    Att_computation = 4 * B * s**2 * h ## QK^T, (NxN) @ V
    Lin_Transform = 8 * B * s * h**2 ## Q, K, V, O
    MLP = 4 * B * s * h * h_ ## 2 Lin Transform
    return 3 * (Att_computation +  Lin_Transform + MLP) ## x3 for fwd + bwd(2x)

def get_num_param_per_layer(h, h_):
    '''
        Ignored bias and norm layers. Probably insignificant. Can be added later
        ## layer_norm =  seq * h
    '''
    QKVO = h * 4*h
    FFNN = 2 * h * h_
    return (QKV + FFNN)

# def get_total_num_param(h, h_):
#     return l * get_num_param_per_layer(h, h_)

# def peak_memory_footprint():
#     '''
#         TODO: Difficult than I thought due to memory from Activations
#     '''
#     total_num_param = get_total_num_param(h, h_)
#     memory_from_param = 16 * total_num_param ## 2 (Gradient) 2 (Fwd Weight) 4 (Momentum1) 4 (Momentum2) 4 (Master Weight)
#     memory_from_data = B * s * h_ ## Highest activation size? 
#     # activation_from_norm
#     # activation_from_att
#     # activation_from_att_
#     # memory_from_activation = (B * s * h + B * s * h_ + B * s * h_) * l ## 

def get_comm_size_per_layer(B, s, h, P):
    '''
        P: model parallelism degree
        return comm size per Transformer layer (att + FFNN)

        Comm Formulas:
        All-Reduce (Ring-Algorithm) = 2 * MSG / P * (P-1) 
        All-Gather = MSG / P * (P-1)
        Reduce-Scatter = MSG / P * (P-1)
        All2All = MSG / P * (P-1)
    '''
    ## SPU
    # 4 all2all (2fwd + 2bwd)
    # divided by P because data is constantly partitioned
    qkv_all2all = B * s/P * 3 * h ## [B, s, 3 * h]
    att_out_all2all = B * s/P * h ## [B, s, h]
    SPU_comm = 2 * (qkv_all2all + att_out_all2all) / P * (P-1) ## Keep one partition for yourself

    ## SPR
    ## TODO

    ## TP
    # 4 all-reduce (2fwd + 2bwd)
    allreduce = B * s * h / P * (P-1) ## identical for att_out and MLP_out
    TP_comm = 4 * allreduce

    ## TP-SP
    TPSP_comm = TP_comm ## Same comm size as argued in the Sequence Parallelism paper: https://arxiv.org/pdf/2309.14509

    return SPU_comm, TP_comm, TPSP_comm

def get_async_comm_size_per_layer(h, h_):
    ## DP
    MSG = get_num_param_per_layer(h, h_)
    DP = 2 * MSG / P * (P-1)

    ## ZERO
    ZERO1 = ZERO2 = DP ## same as argued in the paper: https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/
    ZERO3 = 1.5 * ZERO1 ## extra all-gather of parameters during backward

    ## TODO: MICS

    return DP, ZERO1, ZERO2, ZERO3

## Zero is a bit nuanced, if Zero1 then there is this fwd and bwd all-gathers that we need to do asynchrnoulsy


## Q. Does everyone use mixed-precision always now? 

if __name__ == "__main__":
    num_flops_per_layer = round(get_num_flop_per_layer(B, s, h, h_) / 1_000_000_000_000, 2)
    num_param_per_layer = get_num_param_per_layer(h, h_) // 1_000_000
    
    print("\nAssuming Mixed-Precision ON")
    float_byte = 2 
    SPU_comm, TP_comm, TPSP_comm = [round(comm * float_byte / 1_000_000_000, 2) for comm in get_comm_size_per_layer(B, s, h, P)]
    DP, ZERO1, ZERO2, ZERO3 = [round(async_comm * float_byte / 1_000_000_000, 2) for async_comm in get_async_comm_size_per_layer(h, h_)]

    print(f"num params per layer (M): {num_param_per_layer}")
    print(f"total num params (M): {num_param_per_layer * l}")
    print(f"Tflop count per layer (fp16): {num_flops_per_layer}")
    # print(f"total Tflop count: {num_flops_per_layer * l}")
    print(f"\n")

    print(f"Comm size per layer (GB): ")
    print(f"    Ulysses comm: {SPU_comm}")
    print(f"    TP comm: {TP_comm}")
    print(f"    TPSP comm: {TPSP_comm}")
    print(f"\n")

    print(f"Async comm size per layer (GB): ")
    print(f"    DP(ZERO0): {DP}")
    print(f"    DP/SP + ZERO1: {ZERO1}")
    print(f"    DP/SP + ZERO2: {ZERO2}")
    print(f"    DP/SP + ZERO3: {ZERO3}")
    print(f"\n")