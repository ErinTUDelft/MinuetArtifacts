import argparse
from typing import Optional

import numpy as np
import torch
import minuet

import minuet.nn as spnn
from torch.nn import Sequential
from fvcore.nn import FlopCountAnalysis
import torchsparse
import torchsparse.nn as tsnn

import random

# def cuda_pytorch_profiling(func, input):
#     # Warmup: Execute the function 5 times without profiling
#     # for _ in range(5):
#     #     func(input)

#     # Start profiling
#     with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#         profile_memory=False,  # Set to True to profile memory as well
#         record_shapes=False,
#         with_stack=False
#     ) as prof:
#         #with torch.profiler.record_function("model_inference"):
#         func(input)

#     # Print the results sorted by CUDA time (descending)
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))


def cuda_pytorch_profiling(func, input, num ):

    input.cuda()
    func.cuda()
    # Start profiling
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU, 
            torch.profiler.ProfilerActivity.CUDA],
        with_stack=True, with_flops=True, profile_memory=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
        #profile_memory=False,  # Set to True to profile memory as well
        #record_shapes=False,
        #with_stack=False
        
    ) as prof:
        #with torch.profiler.record_function("model_inference"):
        func(input)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))

    if num== 1:
      prof.export_stacks("/home/erin/sparsity_for_flow/MinuetArtifacts/sparsity_for_flow/results/profiler_stacks_minuet.txt", "self_cuda_time_total")
    if num== 2:
      prof.export_stacks("/home/erin/sparsity_for_flow/MinuetArtifacts/sparsity_for_flow/results/profiler_stacks_torchsparse.txt", "self_cuda_time_total")


    
  
    

def random_unique_points(ndim: int,
                         n: int,
                         c_min: int,
                         c_max: int,
                         dtype=np.int32):
  r"""
  Generate random coordinates without duplicates

  Args:
    ndim: the dimension of each coordinate
    n: the number of points to be generated
    c_min: the minimum coordinate
    c_max: the maximum coordinate
    dtype: the coordinate data type

  Returns:
    a numpy array consists of generated coordinates
  """
  tables = {}
  max_coords = pow(c_max - c_min + 1, ndim)
  if n > max_coords:
    raise ValueError(f"Cannot sample {n} points without replacement from a "
                     f"space with only {max_coords} possible points")

  points = []
  for i in range(n):
    number = random.randrange(i, max_coords)
    value = tables.get(number, number)
    tables[number] = tables.get(i, i)
    point = []
    for j in range(ndim):
      value, x = divmod(value, c_max - c_min + 1)
      point.append(x + c_min)
    points.append(point)
  return np.asarray(points, dtype=dtype)


class SimplePCNet(torch.nn.Module):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.conv1 = minuet.nn.SparseConv3d(
        in_channels=4,
        out_channels=32,
        kernel_size=3,
    )
    self.conv2 = minuet.nn.SparseConv3d(
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        stride=2,
    )
    self.conv3 = minuet.nn.SparseConv3d(
        in_channels=32,
        out_channels=2,
        kernel_size=3,
    )

  def forward(self, x: minuet.SparseTensor):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    return x
  
  class SimplePCNet_TS(torch.nn.Module):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.conv1 = tsnn.Conv3d(
          in_channels=4,
          out_channels=32,
          kernel_size=3,
      )
      self.conv2 = tsnn.Conv3d(
          in_channels=32,
          out_channels=32,
          kernel_size=3,
          stride=2,
      )
      self.conv3 = tsnn.Conv3d(
          in_channels=32,
          out_channels=2,
          kernel_size=3,
      )

  def forward(self, x: torchsparse.SparseTensor):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    return x



def make_dummy_point_cloud(
    batch_size: Optional[int],
    num_points: int,
    num_features: int = 2,
    # Minuet always requires sorted coordinates
    ordered: bool = True,
    c_min: int = 0,
    c_max: int = 200):
  if batch_size is None:
    coordinates = random_unique_points(ndim=3,
                                                           n=num_points,
                                                           c_min=c_min,
                                                           c_max=c_max)
    coordinates = torch.tensor(coordinates).cuda()
    batch_dims = None
  else:
    coordinates = [
        random_unique_points(ndim=3,
                                                 n=num_points,
                                                 c_min=c_min,
                                                 c_max=c_max)
        for _ in range(batch_size)
    ]
    batch_dims = [0]
    batch_dims.extend([len(c) for c in coordinates])
    batch_dims = np.cumsum(np.asarray(batch_dims))

    coordinates = torch.concat([torch.tensor(c) for c in coordinates])
    coordinates = coordinates.cuda()
    batch_dims = torch.tensor(batch_dims, device=coordinates.device)

  features = torch.randn(len(coordinates),
                         num_features,
                         device=coordinates.device)

  if ordered:
    index = minuet.nn.functional.arg_sort_coordinates(coordinates,
                                                      batch_dims=batch_dims)
    coordinates = coordinates[index]

    features = features[index]

    minuet_sparse =  minuet.SparseTensor(features=features,
                             coordinates=coordinates,
                             batch_dims=batch_dims)

    zeros = torch.zeros(47000, 1).cuda()

  # Concatenate the zeros tensor in front of the original tensor along dimension 1
    coordinates = torch.cat((coordinates, zeros), dim=1).int()

    torchsparse_sparse = torchsparse.SparseTensor(feats=features, coords=coordinates)


  return minuet_sparse, torchsparse_sparse
mult = 2
net_minuet = Sequential(
        spnn.SparseConv3d(in_channels=2,
                            out_channels=16*mult,
                            kernel_size=3,
                            stride=2),
                            
        #spnn.BatchNorm(num_features=num_channels[0]),
        spnn.ReLU(True),
        spnn.SparseConv3d(in_channels=16*mult,
                            out_channels=32*mult,
                            kernel_size=3,
                            stride=2),
        #spnn.BatchNorm(num_features=num_channels[0]),
        spnn.ReLU(True),
        spnn.SparseConv3d(in_channels=32*mult,
                            out_channels=16,
                            kernel_size=3,
                            stride=1),
                            #spnn.BatchNorm(num_features=num_channels[0]),
        spnn.ReLU(True),
        spnn.SparseConv3d(in_channels=16,
                            out_channels=2,
                            kernel_size=3,
                            stride=1),
                            
    ).cuda().eval()

net_ts = Sequential(
        tsnn.Conv3d(in_channels=2,
                            out_channels=16*mult,
                            kernel_size=3,
                            stride=2),
                            
        #spnn.BatchNorm(num_features=num_channels[0]),
        tsnn.ReLU(True),
        tsnn.Conv3d(in_channels=16*mult,
                            out_channels=32*mult,
                            kernel_size=3,
                            stride=2),
        #spnn.BatchNorm(num_features=num_channels[0]),
        tsnn.ReLU(True),
        tsnn.Conv3d(in_channels=32*mult,
                            out_channels=16,
                            kernel_size=3,
                            stride=1),
        tsnn.ReLU(True),
        tsnn.Conv3d(in_channels=16,
                            out_channels=2,
                            kernel_size=3,
                            stride=1),
    ).cuda().eval()



def main(args):
  tuning_data = [
      make_dummy_point_cloud(num_points=args.num_points,
                             batch_size=args.batch_size) for _ in range(5)
  ]

 

  # Autotuning is optional but it is better for performance
  #minuet.autotune(net, cache, data=tuning_data)

  # At the current moment, Minuet does not support training
  with torch.no_grad():
    for i in range(3):
      cache = minuet.nn.KernelMapCache(ndim=3, dtype=torch.int32, device="cuda:0")
      minuet.set_kernel_map_cache(module=net_minuet, cache=cache)  
      # Before each different input the model cache must be reset
      # Note that the minuet.autotune may touch the cache as will
      cache.reset()
      minuet_sparse, torchsparse_sparse = make_dummy_point_cloud(num_points=args.num_points,
                                           batch_size=args.batch_size)
      
      cache.reset()
      
      print("starting")

      
      
      cuda_pytorch_profiling(net_minuet, minuet_sparse, num=1)
      cache.reset()

      cuda_pytorch_profiling(net_ts, torchsparse_sparse, num=2)

      
  
      #cuda_pytorch_profiling(net, dummy_input) 
      

      #print(output.C)

      #print(net(dummy_input))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size",
                      "-B",
                      type=int,
                      default=1,
                      help="batch size for inference")
  parser.add_argument("--num_points",
                      "-N",
                      type=int,
                      default = 47000,
                      help="number of points for random generated point cloud")
  main(parser.parse_args())