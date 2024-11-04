This is a collection of mini-applications ("mini-benchmarks") within the context 
of the ALCF-4 AI/ML benchmark applications. We have written these as 
representative of the critical operations performed by the applications. This
effort keeps the order of execution of operations as close to the full 
application as possible. 

### Content of the directory
Here we give a brief description of each benchmark.

#### sequence_parallelism_compute.py
This is a benchmark which focuses on the communication pattern of a sequence 
parallelism implementation. We assume here 
the degree of parallelisms (tensor and sequence) equalling 
to the number of total ranks this mini-application is deployed on. Based on
the user input of the sequence length we perform the following set of 
operations in a loop. 
  - Initializing all the GPU ranks with parts of the sequence.
  - `ALLGATHER` to a buffer in each GPU
  - Perform matrix multiplication of the GPU buffer, 
  mimicking tha application of a weight matrix
  - Perform another matrix multiplication, mimicking the application of 
  another weight matrix.
  - `REDUCE-SCATTER` the GPU buffer after the compute.

  
The FOM for this benchmark is the time to solution, the total time required
for the set of operations described above to be completed for the desired
number of iterations. We also measure the time taken by the individual steps.
The application supports `float32` precision. We are adding support for other
precision types (`TO DO - Merge`).
