import h5py
import torch
import numpy as np
filename="PCA_activitynet_v1-3.hdf5"
h5 = h5py.File(filename,'r')

for key in h5["data"]:
    print(key,h5["data"][key].shape)
#print(h5["data"]["U"][:500].shape)

feat = h5py.File('sub_activitynet_v1-3.c3d.hdf5','r')

#print(feat['v_zyylgHTPUS8']["c3d_features"].shape)


test = np.random.randn(391,4096)
x_mean = h5['data/x_mean']
U = h5['data/U'][:]
S = np.fill_diagonal(np.zeros((U.shape[0],test.shape[0])),h5['data/S'][:])
print(S.shape)
t2 = np.matmul(S,U)
print(t2.shape,"HERE")
print(np.matmul(np.transpose(t2),t2).sum(),"HERE2")
#t1 = np.dot(test-x_mean,np.transpose(U))
print(np.matmul(np.transpose(U),U).sum())
print(U.shape)
print(test.shape)
t1 = np.dot(U,test)
t1 = np.dot(np.transpose(U),t1)
print(t1.shape)
print(t1.sum())
print(test.sum())
print(t1/test)
'''
U = np.expand_dims(U, axis=2)

diff_feat = test - x_mean #subtrac PCA mean from each feature
diff_feat = np.expand_dims(diff_feat, axis=1)
dot_prod = np.squeeze(np.dot(diff_feat, U))
print(np.allclose(t1,dot_prod))

print(t1.shape)
print(np.dot(U,np.transpose(U)).shape)
revert = np.dot(t1,np.transpose(np.transpose(U))) + x_mean
#print(revert)
print()
print()
#print(test)
#print(revert.shape)
#print(test.shape)
#print(np.transpose(U).shape)
#print(np.dot(dot_prod,U).shape)
#print(np.array_equal(test,))
'''
