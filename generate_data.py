#Idea: Generate 2-dim. vectors (elements (x,y) in the unit cube [0,1)x[0,1)) 
#the MLP decides if a vector is in the unit circle, i.e. if x^2+y^2 < 1

import torch



def random_vectors(n):
    '''
    returns a pytorch tensor with n (uniformly dirstributed) random vectors (x,y) where 0 <= x,y < 1, those vectors are also tensors
    '''
    return torch.rand(n,2)



if __name__=='__main__':
    print(random_vectors(4))
