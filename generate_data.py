#Idea: Generate 2-dim. vectors (elements (x,y) in the unit cube [0,1)x[0,1)) 
#the MLP decides if a vector is in the unit circle, i.e. if x^2+y^2 < 1

import torch



def random_vectors(n):
    '''
    returns a pytorch tensor with n (uniformly dirstributed) random vectors (x,y) where 0 <= x,y < 1, those vectors are also tensors
    '''
    return torch.rand(n,2,requires_grad=True)

def test_if_in_circle(tensor):
    '''
    takes a 2D tensor as input. If the coordinates are in the unit circle, returns [1,0], if they are not it returns [0,1]
    '''
    if tensor[0]**2 + tensor[1]**2 < 1:
        return [1.0,0]#we need floats for requires_grad=True in main
    return [0,1.0]
    

if __name__=='__main__':
    #print(random_vectors(4))
    x = torch.tensor([0.5,0])
    print(test_if_in_circle(x))
