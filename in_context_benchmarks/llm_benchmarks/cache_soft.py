#!/usr/bin/env python
# -----------------------------------------------------------------------
# This script is to transfer the package to the local drives.
# Rank 0 will load the data and then do the broadcast and write them back
# -----------------------------------------------------------------------
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
parser.add_argument("--d", action='store_true')
args = parser.parse_args()

import os
import time
comm = MPI.COMM_WORLD
_CHUNK_SIZE=1024*1024*128
def bcast_chunk(A):
    if (comm.rank==0):
        size=len(A)
        print(f"size of data {size}")        
    else:
        size=0
    size = comm.bcast(size, root=0)
    nc = size//_CHUNK_SIZE+1
    B = bytearray(size)
    for i in range(nc):
        if i*_CHUNK_SIZE < size:
            end = min(i*_CHUNK_SIZE+_CHUNK_SIZE, size)
            if (comm.rank==0):
                data = A[i*_CHUNK_SIZE:end]
            else:
                data =None
            B[i*_CHUNK_SIZE:end] = comm.bcast(data, root=0)            
    return B
def Transfer(src, dst, decompress=True):
    start_time = time.time()
    if comm.rank==0:
        start = time.time()
        with open(src, "rb") as f:
            data = f.read()
        end = time.time()
        print("====================")
        print(f"Rank-0 loading library {src} took {end - start} seconds")
    else:
        data=None
    start = time.time()    
    data = bcast_chunk(data)
    end = time.time()
    with open(dst, "wb") as f:
        f.write(data)
    if comm.rank==0:
        print(f"Broadcast took {end-start} seconds")
        print(f"Writing to the disk {dst} took {time.time() - end}")
    start = time.time()
    dirname=os.path.dirname(dst)
    assert(os.path.isfile(dst))
    if (decompress):
        os.system(f"tar -p -xf {dst} -C {dirname}")
    if comm.rank==0:
        if decompress:
            print(f"untar to {dirname} took {time.time() - start} seconds")
        print(f"Total time: {time.time() - start_time} seconds")
        print("------------------\n")
Transfer(args.src, args.dst, args.d)
