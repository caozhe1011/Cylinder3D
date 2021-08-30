import spconv
import torch
import numpy as np

conv = spconv.SubMConv3d(3, 3, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False, indice_key='test')
voxel_features = torch.tensor([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],dtype=torch.float32)  # 总共分8个7个有数据

# conv = spconv.SubMConv3d(1, 3, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False, indice_key='test')
# voxel_features = torch.tensor([[1], [2], [3], [1], [2], [3], [1]], dtype=torch.float32)  # 总共分8个7个有数据
coors = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]])
coors = coors.int()
sparse_shape = np.asarray([2, 2, 2])
ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size=1)
print(conv.weight.shape)
conv.weight.data = torch.ones((1,3,3,3,3))
# conv.weight.data = torch.tensor(([[[[[1., 1., 1.]],
#
#                                     [[2., 2., 2.]],
#
#                                     [[3., 3., 3.]]],
#
#                                    [[[1., 1., 1.]],
#
#                                     [[2., 2., 2.]],
#
#                                     [[3., 3., 3.]]],
#
#                                    [[[1., 1., 1.]],
#
#                                     [[2., 2., 2.]],
#
#                                     [[3., 3., 3.]]]]]))
print(conv.weight.data[:,:,:,0,1])
res = conv(ret)
print(res.dense().data, res.dense().shape)
