# Filename: run_simplenet_inference.py
import argparse
from typing import Optional

import numpy as np
import torch
import minuet
from minuet import nn as spnn

import random



from torch.profiler import profile, ProfilerActivity




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

mult = 8

class SimplePCNet(torch.nn.Module):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.conv1 = minuet.nn.SparseConv3d(
        in_channels=2,
        out_channels=32*mult,
        kernel_size=[3,3,3],
        stride=1,
    )
    self.conv2 = minuet.nn.SparseConv3d(
        in_channels=32*mult,
        out_channels=32*mult,
        kernel_size=3,
        stride=1,
    )
    self.conv3 = minuet.nn.SparseConv3d(
        in_channels=32*mult,
        out_channels=32*mult,
        kernel_size=3,
        stride=1,
    )
    self.conv4 = minuet.nn.SparseConv3d(
        in_channels=32*mult,
        out_channels=2,
        kernel_size=3,
        stride=1,
    )

  def forward(self, x: minuet.SparseTensor):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    #x = minuet.nn.functional.global_avg_pool(x)
    return x


def make_dummy_point_cloud(
    batch_size: Optional[int],
    num_points: int,
    num_features: int = 2,
    # Minuet always requires sorted coordinates
    ordered: bool = True,
    c_min: int = 0,
    c_max: int = 127):
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
    torch.set_printoptions(threshold=10_0000)
    #print(coordinates)
    # Don't forget to permute your features
    # It doesn't matter for dummy inputs though
    features = features[index]

  return minuet.SparseTensor(features=features,
                             coordinates=coordinates,
                             batch_dims=batch_dims)


def main(args):
  tuning_data = [
      make_dummy_point_cloud(num_points=args.num_points,
                             batch_size=args.batch_size).to("cuda:0") for _ in range(5)
  ]
  
  net = SimplePCNet().to("cuda:0")
  print(net)
  net.eval()

  cache = minuet.nn.KernelMapCache(ndim=3, dtype=torch.int32, device="cuda:0")
  minuet.set_kernel_map_cache(module=net, cache=cache)

  # Autotuning is optional but it is better for performance
#  minuet.autotune(net, cache, data=tuning_data)

#  At the current moment, Minuet does not support training
 # with torch.no_grad(): # not needed for now
  for i in range(10):
    # Before each different input the model cache must be reset
    # Note that the minuet.autotune may touch the cache as well
    cache.reset()
    dummy_input = make_dummy_point_cloud(num_points=args.num_points,
                                          batch_size=args.batch_size).to("cuda:0")
    
    print("input", dummy_input.device)
    net.to("cuda:0")
    print("net", net)
    
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:

    
      output = net(dummy_input)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    # trace_filename = f"/home/erin/sparsity_for_flow/MinuetArtifacts/TraceData/trace_{i}.json"
  #  prof.export_chrome_trace(trace_filename)
    




    # torch.cuda.synchronize()
    # event1 = torch.cuda.Event(enable_timing=True)
    # event2 = torch.cuda.Event(enable_timing=True)
    # event1.record()
    # output = net(dummy_input)
    # event2.record()
    # event2.synchronize()
    # print("time", event1.elapsed_time(event2))
  
#   dummy_inputs = [
#   make_dummy_point_cloud(num_points=args.num_points, batch_size=args.batch_size).to("cuda:0")
#   for _ in range(6)
# ]
    
#   with profile(
#   activities=[ProfilerActivity.CUDA], with_stack=False,
#   schedule=torch.profiler.schedule(
#       wait=0,
#       warmup=5,
#       active=1),
#   on_trace_ready=trace_handler
# ) as p:
#     for i in range(6):
#       cache.reset()
#       net(dummy_inputs[i].to("cuda:0"))
#       p.step()

# def trace_handler(p):
#     output = p.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=15,)
#     print("output cuda", output)
#     p.export_chrome_trace("/home/erin/sparsity_for_flow/MinuetArtifacts/TraceData/trace_" + str(p.step_num) + ".json")


      


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size",
                      "-B",
                      type=int,
                      default=4,
                      help="batch size for inference")
  parser.add_argument("--num_points",
                      "-N",
                      type=int,
                      default = 10000,
                      help="number of points for random generated point cloud")
  main(parser.parse_args())

