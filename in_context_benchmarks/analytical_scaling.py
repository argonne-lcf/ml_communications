# iPort torch
import argparse

## TODO: Implement combination of different parallelisms
## TODO: Implement head count asserts for ulysses, TP, etc. 
## TODO: Implement peak memory size. Also memory saving compared to DP?
## TODO: GQA
## TODO: need to reorganizing.

## Lamma 70B
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", "-B", type=int, default=1, help="") ## Q. What was the global batch size in Llama-3 70B
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
        LLM includes: 2 * h * vocab_size (look up table and Linear head)
    '''
    QKVO = h * 4*h
    FFNN = 2 * h * h_
    return (QKVO + FFNN)

# def get_total_num_param(h, h_):
#     return l * get_num_param_per_layer(h, h_)

def peak_memory_footprint():
    '''
        TODO: Difficult than I thought due to memory from Activations
    '''
    # total_num_param = get_total_num_param(h, h_)
    # memory_from_param = 16 * total_num_param ## 2 (Gradient) 2 (Fwd Weight) 4 (Momentum1) 4 (Momentum2) 4 (Master Weight)
    # memory_from_data = B * s * h_ ## Highest activation size? 
    # # activation_from_norm
    # # activation_from_att
    # # activation_from_att_
    # # memory_from_activation = (B * s * h + B * s * h_ + B * s * h_) * l ## 

    ## Borrowing from: https://arxiv.org/html/2411.06465v1

    # D = P ## DP degree
    # DP_memory_fpt = (6 + 12/D)(2*h*)

def get_comm_size_per_layer(B, s, h, P):
    '''
        P: model parallelism degree
        return comm size per Transformer layer (att + FFNN)
        ## NOTE: we compute the total message size across the entire system not buffer size per GPU. 

        Comm Formulas:
        All-Reduce (Ring-Algorithm) = 2 * MSG / P * (P-1) 
        All-Gather = MSG / P * (P-1)
        Reduce-Scatter = MSG / P * (P-1)
        All2All = MSG / P * (P-1)
    '''
    ## SPU
    # 4 all2all (2fwd + 2bwd)
    # divided by P because data is constantly partitioned
    qkv_all2all = B * s/P * 3 * h ## [B, s, 3 * h] - buffer size
    att_out_all2all = B * s/P * h ## [B, s, h] 
    SPU_comm = (qkv_all2all + att_out_all2all) * (P-1) ## Keep one partition for yourself
                                                       ## / P cancels out as Message Size = buffer * P

    ## SPR
    ## TODO

    ## TP
    # 4 all-reduce (2fwd + 2bwd)
    allreduce = B * s * h  ## identical for att_out and MLP_out
    TP_comm = 8 * allreduce * (P-1) ## / P cancels out, 8 = num_all_reduce * all_reduce_cost_coeff

    ## TP-SP
    TPSP_comm = TP_comm ## Same comm size as argued in the Sequence Parallelism paper: https://arxiv.org/pdf/2309.14509

    return SPU_comm, TP_comm, TPSP_comm

def get_async_comm_size_per_layer(h, h_):
    ## DP
    grad_buffer = get_num_param_per_layer(h, h_)
    DP = 2 * grad_buffer * (P-1) ## / P cancels out

    ## ZERO
    ## Zero is a bit nuanced, some comm are completely async while some are async only per layer
    ZERO1 = ZERO2 = DP ## same as argued in the paper: https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/
    ZERO3 = 1.5 * ZERO1 ## extra all-gather of parameters during backward

    ## TODO: MICS / ZERO++

    return DP, ZERO1, ZERO2, ZERO3

def memory_fpt_vit(s, b, h, h_ffn, L, pa, k, hc, v, t, d, c, p, verbose=True):
    ## TODO: separate vit and llm? currently it should be vit only
    ## TODO: combine formula to get the simplified form but comment on each componenets that totals to the simplied form.
    ## NOTE: Assuming ZeRO3 is used with DP, CP/SP

    ## for 3dvit, replace pa**2 with pa**3
    qkvo = 2*h**2 + 2*h**2*(k/hc)
    norm = 2*s ## negligible
    ffnn = 2*h*h_ffn
    num_parameters_att = qkvo + norm + ffnn
    patchify = pa**3 * h ## TODO: Change pa to cubed? 
    final_norm = 2*s
    # head = h*cl ## ignorable cl <<  h
    num_parameters_vit = (num_parameters_att * L/p + patchify + final_norm) / t

    swiglu = (h * h_ffn)
    num_parameters_llm_specific = (2*h*v + h)
    num_parameters_llm = ((num_parameters_att + swiglu) * L/p + num_parameters_llm_specific)/t
    num_parameters_llm_without_swiglu = (num_parameters_att * L/p) + (num_parameters_llm_specific/t)

    # activation_transformer = s*b*h*(6+4*k/a) + 2*s*b*(h+4*h_ffn) + 4*s*b*h ## w/swiglu but w/out embedding and lm head
    # activation_transformer = 2*s*b*h+2*s*b*h_ffn + 6*s*b*h + 4*s*b*h ## w/out swiglu
    att = (1 + 4/t) * b*s*h   # x, qkvo (activation of x is not parallelized)
    norm = 2*b*s*h # need two because standardization, element-wise linear but easy to recompute? Am I using RMSNorm or LayerNorm?
    # att = qkvo
    ffnn = b*s*h + b*s*h_ffn/t ## only the intermediate input is paralleized by TP
    num_act_elements = att + ffnn + norm ## 
    # dropout = b*s*h ## Don't we need this?
    num_bytes = 2
    activation_transformer = num_bytes*num_act_elements #+ dropout
    num_patchify = b*s*pa**2
    # num_head = b*h ## ignorable
    activation_vit_specific = num_bytes*num_patchify
    activation_vit = (activation_transformer * L + activation_vit_specific) / (d * c)

    ## llm (from the paper)
    # term1 = (6 + 12/(d*c)) * (h*v/t + (2*L/p)*h**2 * ((1 + k/a + (3/2)*h_ffn/h) / t + 1/h))
    # term2 = s*b*h/(t*c) * ((12 + 4*k/a + 8*h_ffn/h)*L + 8*p + 4*(1+v/h))
    # term1 = 18 / (d*c) * num_parameters_llm

    if hpz == 1:
        term1 = 18 / (d*c) * num_parameters_vit 
    else:
        term1 = (16/(d*c) + 2/hpz) * num_parameters_vit
    term2 = activation_vit
    # term2 = s*b*h*(16 + 4*k/a + 8*h_ffn/h) ## from the paper
    # result = term1 + term2 + peak_data_or_act_size
    result = term1 + term2
    # print(f"peak_data_or_act_size: {peak_data_or_act_size / 1024**3:.1f}")
    num_gpus = t*d*p*c
    # print(f"mem fpt per GPU (GiB): {result / 1000**3:.1f}")
    mem_fpt_per_gpu = result / 1024**3
    model_memory = term1 / 1024**3
    activation_memory = term2 / 1024**3
    if verbose:
        print(f"num_parameters_llm (M): {num_parameters_llm / 1000**2:.1f}")
        print(f"num_parameters_llm_without_swiglu (M): {num_parameters_llm_without_swiglu / 1000**2:.1f}")

        print(f"num_parameters_vit (M): {num_parameters_vit / 1000**2 :.1f}")
        print(f"term1 (model related memory in GiB)       : {term1 / 1024**3:.1f}")
        print(f"term2 (activationi related memory in GiB) : {term2 / 1024**3:.1f}")
        print(f"Total {num_gpus} GPUs, TP:{t}, DP{d}:, CP:{p}, PP:{c}")
        print(f"mem fpt per GPU (GiB): {mem_fpt_per_gpu:.1f}")
    return mem_fpt_per_gpu, model_memory, activation_memory

if __name__ == "__main__":
    # h = 64**2
    # h_ffn = h*4
    # L = 32
    b = 24
    d = b ## MBS=1, change otherwise

    s = 64**2 + 1
    pa = 16
    k = hc = 16
    v = 0
    t = 1
    c = 1 ## context or ulysses..
    p = 1
    hpz = 1 ## Secondary Paritioning. Probably not working probably? 1: REGULAR else 
    # cl = 10 ## num_classes

    # ## ViT-TINY (10M)
    # NLAYERS=6
    # HSIZE=512
    # FFN_HSIZE=512
    ## ViT-BASE (86M)
    # NLAYERS=12
    # HSIZE=768
    # FFN_HSIZE=3072
    # ## VIT-LARGE (307M)
    # NLAYERS=24
    # HSIZE=1024
    # FFN_HSIZE=4096
    ## VIT-HUGE (632M)
    # NLAYERS=32
    # HSIZE=1280
    # FFN_HSIZE=5120
    # ## GIANT
    # NLAYERS=48
    # HSIZE=1664
    # FFN_HSIZE=8192
    ## ENORMOUS
    # NLAYERS=56
    # HSIZE=1792
    # FFN_HSIZE=15360
    # NUM_HEADS=16

    ## 2.7B
    # NLAYERS = 24
    # HSIZE = 3072
    # FFN_HSIZE = 4 * HSIZE

    ## 5B
    # NLAYERS = 28
    # HSIZE = 64 * 60
    # FFN_HSIZE = 4 * HSIZE

    ## 5.6B
    # NLAYERS = 28
    # HSIZE = 64 * 64
    # FFN_HSIZE = 4 * HSIZE

    ## 6.4B
    # NLAYERS=32
    # HSIZE=4096
    # FFN_HSIZE=4*HSIZE

    ## 8.2B
    # NLAYERS=32
    # HSIZE=64 * 72
    # HSIZE=12000
    # FFN_HSIZE=4*HSIZE

    ## 9.2B
    # NLAYERS=36
    # HSIZE=64 * 72
    # FFN_HSIZE=4*HSIZE

    ## 10.2B
    # NLAYERS=36
    # HSIZE=64 * 76
    # FFN_HSIZE=4*HSIZE

    ## 11.4B
    # NLAYERS=38
    # HSIZE=64 * 78
    # FFN_HSIZE=4*HSIZE
    
    ## VIT 13B (12.6)
    # NLAYERS=40
    # HSIZE=5120
    # FFN_HSIZE=4*HSIZE
    
    ## 14B (13.8)
    # NLAYERS=44
    # HSIZE=64 * 80
    # FFN_HSIZE=4*HSIZE

    ## 16.7
    # NLAYERS=44
    # HSIZE=64 * 88
    # FFN_HSIZE=4*HSIZE

    ## 19.9
    # NLAYERS=44
    # HSIZE=64 * 96
    # FFN_HSIZE=4*HSIZE

    ## 21.8B
    # NLAYERS=48
    # HSIZE=6144
    # FFN_HSIZE=24576

    ## 24.5B
    # NLAYERS=48
    # HSIZE=64 * 102
    # FFN_HSIZE= 4 * HSIZE

    ## 25.5B
    # NLAYERS=49
    # HSIZE=64 * 104
    # FFN_HSIZE= 4 * HSIZE

    ## 27.6B
    # NLAYERS=50
    # HSIZE=64 * 106
    # FFN_HSIZE= 4 * HSIZE

    # ## 29.7B
    # NLAYERS=50
    # HSIZE=64 * 110
    # FFN_HSIZE= 4 * HSIZE

    ## 36.6B
    NLAYERS=50
    HSIZE=64 * 122
    FFN_HSIZE= 4 * HSIZE

    ## 42.4B
    # NLAYERS=51
    # HSIZE=64 * 130
    # FFN_HSIZE= 4 * HSIZE

    ## 46.4B
    # NLAYERS=51
    # HSIZE=64 * 136
    # FFN_HSIZE= 4 * HSIZE

    # ## 56.0B 
    # NLAYERS=52
    # HSIZE=64 * 148
    # FFN_HSIZE= 4 * HSIZE

    # ## 64.0B
    # NLAYERS=53
    # HSIZE=64 * 156
    # FFN_HSIZE= 4 * HSIZE

    ## 112B
    # NLAYERS=56
    # HSIZE=64 * 202
    # FFN_HSIZE= 4 * HSIZE

    # s = 2**2
    L = NLAYERS
    h = HSIZE
    h_ffn = FFN_HSIZE

    # model_size=13
    # num_layers=40
    # hidden_size=
    # num_attn_heads=40
    # lr=1.0e-4
    # min_lr=1.0e-6
    # init_std=0.008

    # ## Llama 3.1 - 8.0B
    # pa = 1
    # s = 8192
    # h = 4096
    # h_ffn = 14336
    # L = 32
    # a = hc = 40
    # v = 128_256
    # k = 10
    # d = 1
    # t = 4
    # c = 1
    # p = 2
    # b = 1

    # L=32
    # h=4096
    # h_ffnh=4*h
    # hc=32
    # b=1024
        # return result

    # print(f"memory_fpt_vit(s, b, h, h_ffn, L, pa, a, k, hc, v, t, d, c, p): {memory_fpt_vit(s, b, h, h_ffn, L, pa, a, k, hc, v, t, d, c, p)}")

    memory_fpt_vit(s, b, h, h_ffn, L, pa, k, hc, v, t, d, c, p, verbose=True)

    ## Plotting scaling plot
    # # b = d = 12
    # # b = t = 12; d = 1
    # scaling_range = range(1, 20_000, 100)
    # mem_fpts = [memory_fpt_vit(s, b, h, h_ffn, L, pa, k, hc, v, t, d, c, p, verbose=False)[0] for h in scaling_range] ## overwite s
    # model_memories = [memory_fpt_vit(s, b, h, h_ffn, L, pa, k, hc, v, t, d, c, p, verbose=False)[1] for h in scaling_range] ## overwite s
    # activation_memories = [memory_fpt_vit(s, b, h, h_ffn, L, pa, k, hc, v, t, d, c, p, verbose=False)[2] for h in scaling_range] ## overwite s

    # import matplotlib.pyplot as plt
    # plt.plot(scaling_range, mem_fpts, label="Total Memory Footprint per GPU")
    # plt.plot(scaling_range, model_memories, label="Model Memory")
    # plt.plot(scaling_range, activation_memories, label="Activation Memory")

    # # Add labels, title, and legend
    # plt.xlabel("Hidden Dim")
    # plt.ylabel("Memory Footprint per GPU")
    # plt.axvline(x=4608, color='red', linestyle='--', linewidth=1.5)
    # plt.text(2800, max(mem_fpts) * 0.3, "8B model", color='red', ha='center')
    # # plt.text(4608, max(mem_fpts) * 0.9, "8B model", color='red', rotation=90, ha='center')
    # plt.title("TP=12 with 4K Sequence")
    # plt.legend()
    # plt.savefig("analytical_scaling.png", format="PNG")



    # print(f"num_parameters_w_swiglu: {num_parameters_w_swiglu // 1_000**2}")
    # print(f"mem fpt per GPU: {result // (t * d * p * c) // 1024**3}")

    ## term1 should be atleast 1? 
    ## term2 is too small? 


    # num_flops_per_layer = round(get_num_flop_per_layer(B, s, h, h_) / 1000**4, 2)
    # num_param_per_layer = round(get_num_param_per_layer(h, h_) / 1000**2, 2)
    # total_num_param = round(get_num_param_per_layer(h, h_) * l / 1000**3, 2)
    
    # print(f"\n")
    # print("Assuming Mixed-Precision ON") ## Q. Does everyone use mixed-precision always now? 
    # float_byte = 2 
    # SPU_comm, TP_comm, TPSP_comm = [round(comm * float_byte / 1000**3, 2) for comm in get_comm_size_per_layer(B, s, h, P)]
    # DP, ZERO1, ZERO2, ZERO3 = [round(async_comm * float_byte / 1000**3, 2) for async_comm in get_async_comm_size_per_layer(h, h_)]

    # print(f"num params per layer (M): {num_param_per_layer}")
    # print(f"total num params (B): {total_num_param}")
    # print(f"Tflop count per layer (fp16): {num_flops_per_layer}")
    # # print(f"total Tflop count: {num_flops_per_layer * l}")
    # print(f"\n")

    # print(f"Comm size per layer (GB): ")
    # print(f"    Ulysses comm: {SPU_comm}")
    # print(f"    TP comm: {TP_comm}")
    # print(f"    TPSP comm: {TPSP_comm}")
    # print(f"\n")

    # print(f"Async comm size per layer (GB): ")
    # print(f"    DP(ZERO0): {DP}")
    # print(f"    DP/SP + ZERO1: {ZERO1}")
    # print(f"    DP/SP + ZERO2: {ZERO2}")
    # print(f"    DP/SP + ZERO3: {ZERO3}")
    # print(f"\n")

    