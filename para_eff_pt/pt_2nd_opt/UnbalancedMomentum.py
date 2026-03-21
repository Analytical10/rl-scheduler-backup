# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

# 从UNM_projector.py复制的辅助函数和投影器类
@torch.no_grad()
def PowerIter(mat_g, error_tolerance=1e-6, num_iters=10):
    """Power iteration.
    Compute the maximum eigenvalue of mat, for scaling.
    v is a random vector with values in (-1, 1)

    Args:
    mat_g: the symmetric PSD matrix.
    error_tolerance: Iterative exit condition.
    num_iters: Number of iterations.

    Returns:
    eigen vector, eigen value, num_iters
    """
    original_type=mat_g.data.dtype
    mat_g=mat_g.to(torch.float32) #切高精度
    v = torch.rand(list(mat_g.shape)[0], device=mat_g.get_device()) * 2 - 1
    error = 1
    iters = 0
    singular_val = 0.0
    while error > error_tolerance and iters < num_iters:
        v = v / torch.norm(v)
        mat_v = torch.mv(mat_g, v)
        s_v = torch.dot(v, mat_v)
        error = torch.abs(s_v - singular_val)
        v = mat_v
        singular_val = s_v
        iters += 1
    return singular_val, v / torch.norm(v), iters #输出也是高精度

@torch.no_grad()
def MatPower(mat_m, p):
    """Computes mat_m^p, for p a positive integer.

      Args:
        mat_m: a square matrix
        p: a positive integer

      Returns:
        mat_m^p
    """
    if p in [1, 2, 4, 8, 16, 32]:
        p_done = 1
        res = mat_m
        while p_done < p:
            res = torch.matmul(res, res)
            p_done *= 2
        return res

    power = None
    while p > 0:
        if p % 2 == 1:
            power = torch.matmul(mat_m, power) if power is not None else mat_m
        p //= 2
        mat_m = torch.matmul(mat_m, mat_m)
    return power

@torch.no_grad()
def ComputePower(mat_g, 
                 p=2,
                 iter_count=10,
                 ire=1.0,
                 error_tolerance=1e-6,
                 ridge_epsilon=1e-6,
                 mode='newton'):
    """A method to compute G^{-1/p} using a coupled Newton iteration.

      See for example equation 3.2 on page 9 of:
      A Schur-Newton Method for the Matrix p-th Root and its Inverse
      by Chun-Hua Guo and Nicholas J. Higham
      SIAM Journal on Matrix Analysis and Applications,
      2006, Vol. 28, No. 3 : pp. 788-804
      https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

      Args:
        mat_g: A square positive semidefinite matrix
        p: a positive integer
        iter_count: Stop iterating after this many rounds.
        error_tolerance: Threshold for stopping iteration
        ridge_epsilon: We add this times I to G, to make is positive definite.
                       For scaling, we multiply it by the largest eigenvalue of G.
      Returns:
        (mat_g + rI)^{-1/p} (r = ridge_epsilon * max_eigenvalue of mat_g).
    """
    original_type=mat_g.data.dtype
    mat_g=mat_g.to(torch.float32) #切高精度
    if mode=='newton':
        identity = torch.eye(mat_g.size(0), device=mat_g.get_device())
        alpha = -1.0/p
        max_ev, _, _ = PowerIter(mat_g)
        ridge_epsilon *= max_ev
        mat_g += ridge_epsilon * identity
        z = (1 + p) / (2 * torch.norm(mat_g))
        mat_root = identity * torch.pow(z, 1.0/p)
        mat_m = mat_g * z
        error = torch.max(torch.abs(mat_m - identity))
        count = 0
        while error > error_tolerance and count < iter_count:
            tmp_mat_m = (1 - alpha) * identity + alpha * mat_m
            new_mat_root = torch.matmul(mat_root, tmp_mat_m)
            mat_m = torch.matmul(MatPower(tmp_mat_m, p), mat_m)
            new_error = torch.max(torch.abs(mat_m - identity))
            if new_error > error * 1.2:
                break
            mat_root = new_mat_root
            error = new_error
            count += 1
        
        if ire<1:
            eigenvalues, eigenvectors = torch.linalg.eigh(mat_g)
            k=int(0.05*mat_g.size(0))+3
            P=eigenvectors[:,-k:]
            Gamma=P.t()@mat_root@P
            mat_root=mat_root-(1-ire)*P@Gamma@P.t()
        else: 
            eigenvalues=None
         
    elif mode=='svd':
        eigenvalues, eigenvectors = torch.linalg.eigh(mat_g)
        k=int(0.05*mat_g.size(0))+3
        
        topk_indices = torch.arange(len(eigenvalues) - k, len(eigenvalues))
        ev_inv=torch.clamp(eigenvalues, min=0.0)**(1/p)+5e-3
        ev_inv[topk_indices] *= (1.0/ire) #topk子空间用0.2倍学习率, ire<=1
        ev_inv=torch.diag_embed(1.0/ev_inv)
        
        mat_root=eigenvectors@ev_inv@ eigenvectors.t()
        

    return mat_root.type(original_type), eigenvalues

@torch.no_grad()
def matrix_power(matrix,pow=-0.5,type='svd'):
    #默认matrix为正定阵
    if type=='svd':
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        pow_eigenvalues =  torch.pow(eigenvalues+ 1e-10 ,pow)
        result = (eigenvectors * pow_eigenvalues) @ eigenvectors.t()
    elif type=='power_approx':
        result=ComputePower(matrix, p=2,iter_count=10)
        eigenvalues=None 
        
    return eigenvalues, result


@torch.no_grad()
def Q_proj(M, P, side='right'): #M(I-P P^T), (I-P P^T) M
    if side=='right':
        return M-M@P@P.t()
    else: return M-P@(P.t()@M)

class UNM_Projector_default:
    def __init__(self, max_subspace_ratio,mode='right', initial_eps=1e-6,ire=1.0,verbose=False,log_E=False):
        self.max_subspace_ratio = max_subspace_ratio #一般取0.01-0.05
        self.verbose = verbose
        self.log_E=log_E
        self.precondition= None
        self.momentum= None
        self.initial_eps=initial_eps
        self.ire=ire

        self.diag=None
        self.P=None
        self.mode=mode
    
    def update_P(self, full_rank_grad):
        # small batch full_rank_grad   
        m,n=full_rank_grad.size(0),full_rank_grad.size(1)

        full_rank_grad_high=full_rank_grad.to(torch.float32) #切高精度

        if self.momentum is None: 
            self.momentum=full_rank_grad_high
        
        if self.mode=='right' :
            k=int(self.max_subspace_ratio*min(m,n))+1
            predcon_high=self.momentum.t()@self.momentum
            
        else:  
            k=int(self.max_subspace_ratio*min(m,n))+1
            predcon_high=self.momentum@self.momentum.t()

        eigenvalues, eigenvectors = torch.linalg.eigh(predcon_high)
        
        if k>0:
            self.P = eigenvectors[:, -k:]
        else: #这里取k<0方便输入，代表自适应
            k=int(0.02*min(m,n))+1
            while True:                
                total_sum = torch.sum(eigenvalues[-k:])               
                if total_sum > 0.9*torch.sum(eigenvalues) :
                    self.P = eigenvectors[:, -k:]
                    break
                else:
                    k += int(0.02*min(m,n))+1
                    k=min(k,min(m,n)-1)
            

        
        return None
    
    
    def update_direction_sharp(self, full_rank_grad,theta1,pows,RMS_pow=1): #若参数ndim=1，需要提前设置full_rank_grad为(n,1)形状         
        original_type=full_rank_grad.data.dtype
        m,n=full_rank_grad.size(0),full_rank_grad.size(1)
        pow_diag,pow=pows
        full_rank_grad_high=full_rank_grad.to(torch.float32) #切高精度

          
        
        if self.mode=='right' :
            P_grad= full_rank_grad_high@self.P
            P_momentum=(1-theta1)*P_grad+theta1*self.momentum@self.P
            self.momentum=P_momentum@self.P.t()+Q_proj(self.momentum, self.P, side='right')
            norm_square=torch.sum(self.momentum**2)


            diag=torch.sum(self.momentum**2, dim=1)*m/norm_square+1e-8
            diag=(diag**pow_diag).reshape(-1,1) #eg. pow=-0.5

            
            
            E,M_pow=matrix_power(n*P_momentum.t()@P_momentum/norm_square,pow)
            RMS_inv=torch.sqrt(m*n/norm_square)
            update= RMS_inv**RMS_pow * (diag*P_momentum)@M_pow @self.P.t()
            
            
        else: 
            P_grad= self.P.t()@full_rank_grad_high 
            P_momentum=(1-theta1)*P_grad+theta1*self.P.t()@self.momentum   
            self.momentum=self.P@P_momentum+Q_proj(self.momentum, self.P, side='left')
            norm_square=torch.sum(self.momentum**2)


            diag=torch.sum(self.momentum**2, dim=0)*n/norm_square+1e-8
            diag=(diag**pow_diag).reshape(1,-1) #eg. pow=-0.5


            
            E,M_pow=matrix_power(m*P_momentum@P_momentum.t()/norm_square,pow)
            RMS_inv=torch.sqrt(m*n/norm_square)
            update= RMS_inv**RMS_pow * self.P@ ( M_pow@(P_momentum*diag)  )
            
        return  update.to(original_type)
    

    def update_direction_flat(self, full_rank_grad,theta1,pows,RMS_pow=1): #若参数ndim=1，需要提前设置full_rank_grad为(n,1)形状  
        m,n= full_rank_grad.size(0), full_rank_grad.size(1)
        pow_diag,pow=pows
        k=self.P.size(1)
        original_type=full_rank_grad.data.dtype        
        
        full_rank_grad_high=full_rank_grad.to(torch.float32) #切高精度
        Q_momentum=(1-theta1)*full_rank_grad_high+theta1* self.momentum

        if self.mode=='right':
            Q_momentum=Q_proj(Q_momentum, self.P, side='right')
                
            self.momentum=Q_momentum+self.momentum@self.P@self.P.t()
            norm_square=torch.sum(self.momentum**2)
            
            diag=torch.sum(self.momentum**2, dim=1)*m/norm_square+1e-8
            diag=(diag**pow_diag).reshape(-1,1) #eg. pow=-0.5


            
            lambda_flat=n*torch.sum(Q_momentum**2)/( (n-k)*norm_square )  
            RMS_inv=torch.sqrt(m*n/norm_square) #测试 
            update= RMS_inv**RMS_pow * torch.pow(lambda_flat+1e-8,pow)*diag*Q_momentum  

        else:
            Q_momentum=Q_proj(Q_momentum, self.P, side='left')
                
            self.momentum=Q_momentum+self.P@(self.P.t()@self.momentum)  
            norm_square=torch.sum(self.momentum**2)

            diag=torch.sum(self.momentum**2, dim=0)*n/norm_square+1e-8
            diag=(diag**pow_diag).reshape(1,-1) #eg. pow=-0.5


            lambda_flat=m*torch.sum(Q_momentum**2)/ ( (m-k)*norm_square )

            RMS_inv=torch.sqrt(m*n/norm_square) #测试

            update= RMS_inv**RMS_pow * torch.pow(lambda_flat+1e-8,pow)*Q_momentum *diag
        
        return  update.to(original_type)



class UNM_Projector_qk:
    def __init__(self, max_subspace_ratio, initial_eps=1e-6,ire=1.0, verbose=False,n_head=8):
        self.max_subspace_ratio = max_subspace_ratio
        self.verbose = verbose
        
         
        self.precond= None
        self.precond_inv= None
        self.diag=None 
        self.momentum=None
        self.initial_eps=initial_eps
        self.ire=ire
        self.n_head=n_head
        self.head_dim=0 #初始化
        self.diag=None
        self.P=None
        
     
    def update_P(self, grad_blocks):
        # small batch full_rank_grad   
        original_type=grad_blocks.data.dtype
        device=grad_blocks.device.type
        m,n= grad_blocks.size(0), grad_blocks.size(1)
        self.head_dim=m//self.n_head #初始化，必须是整数,例如:512/8=64

        k=int(self.max_subspace_ratio*n)+1  
        grad_blocks_high=grad_blocks.to(torch.float32) #切高精度

        if self.momentum is None: 
            self.momentum=grad_blocks_high.view(self.n_head, self.head_dim, n) #例如：(8,64,512)
            
            self.P = [
                torch.zeros(self.head_dim, 1, device=device, dtype=grad_blocks_high.dtype)
                for _ in range(self.n_head)
            ]   # 初始化，形状: 长为8的list，(64, k_i )
        for i, momentum_block in enumerate(self.momentum):
            predcon_high=momentum_block@momentum_block.t()
        
            eigenvalues, eigenvectors = torch.linalg.eigh(predcon_high)
              
            

            if k>0:
                self.P[i]=eigenvectors[:, -k:]
            else: #这里取k<0方便输入，代表自适应
                sub_k=int(0.1*min(self.head_dim,n))+1
                while True:
                                    
                    total_sum = torch.sum(eigenvalues[-sub_k:])               
                    if total_sum > 0.9*torch.sum(eigenvalues) :
                        self.P[i] = eigenvectors[:, -sub_k:]
                        break
                    else:
                        sub_k += int(0.05*min(self.head_dim,n))+1
                        sub_k=min(sub_k,min(m,n)-1)
                 

        return None
   
    def update_direction_sharp(self, grad_blocks,theta1,pows,RMS_pow=1): #若参数ndim=1，需要提前设置full_rank_grad为(n,1)形状
        ###input中的grad_blocks不是三阶张量，而是个矩阵，这里只是为了记号简便
        pow_diag,pow=pows
        original_type=grad_blocks.data.dtype 
        m,n= grad_blocks.size(0), grad_blocks.size(1)
          
        
        self.head_dim=m//self.n_head    #必须是整数,例如:512/8=64
         
         
        update=self.momentum.clone()       #例如：(8,64,512)

        grad_blocks_high = grad_blocks.to(torch.float32) #切高精度
        grad_blocks_high = grad_blocks_high.view(self.n_head, self.head_dim, n) #例如：(8,64,512)
         

                
        for i, grad_block_high in enumerate(grad_blocks_high):

            P_grad= self.P[i].t()@grad_block_high
            P_momentum=(1-theta1)*P_grad+theta1*self.P[i].t()@self.momentum[i] 
            
            
            
            #注意 update[i]，self.momentum[i]更新的先后顺序
            self.momentum[i]=self.P[i]@P_momentum+Q_proj(self.momentum[i], self.P[i], side='left')
            norm_square=torch.sum(self.momentum[i]**2)

            diag=torch.sum(self.momentum[i]**2, dim=0)*n/norm_square+1e-8
            diag=(diag**pow_diag).reshape(1,-1) #eg. pow=-0.5


            
            E,M_pow=matrix_power(self.head_dim*P_momentum@P_momentum.t()/norm_square,pow)

            RMS_inv=torch.sqrt(self.head_dim*n/norm_square)  #测试
            update[i]= RMS_inv**RMS_pow * self.P[i]@ ( M_pow@(P_momentum*diag)  )

            
            
            
        return update.view(m,n).to(original_type)       

        
    def update_direction_flat(self, grad_blocks,theta1,pows,RMS_pow=1): #若参数ndim=1，需要提前设置full_rank_grad为(n,1)形状  
        #默认self.momentum已经有过赋值
        pow_diag,pow=pows
        original_type=grad_blocks.data.dtype       
        m,n= grad_blocks.size(0), grad_blocks.size(1)
        k=self.P[0].size(1)
        update=self.momentum.clone()       #例如：(8,64,512) 
        grad_blocks_high = grad_blocks.to(torch.float32) #切高精度
        grad_blocks_high = grad_blocks_high.view(self.n_head, self.head_dim, n) #例如：(8,64,512)


        
        for i, grad_block_high in enumerate(grad_blocks_high):
            Q_momentum_i=(1-theta1)*grad_block_high+theta1* self.momentum[i] #例如：(64,512)
            Q_momentum_i=Q_proj(Q_momentum_i, self.P[i], side='left')
                 
            
            
            #update,self.momentum[i] 更新顺序不能乱
            self.momentum[i]=Q_momentum_i+self.P[i]@(self.P[i].t()@self.momentum[i])
            norm_square=torch.sum(self.momentum[i]**2)

            diag=torch.sum(self.momentum[i]**2, dim=0)*n/norm_square+1e-8
            diag=(diag**pow_diag).reshape(1,-1) #eg. pow=-0.5

            lambda_flat=self.head_dim*torch.sum(Q_momentum_i**2)/ ( (self.head_dim-k)*norm_square )

            RMS_inv=torch.sqrt(self.head_dim*n/norm_square) #测试

            update[i]= RMS_inv**RMS_pow * torch.pow(lambda_flat+1e-8,pow)*Q_momentum_i *diag

            
        return  update.view(m,n).to(original_type)

class UNM_Projector_norm:
    def __init__(self,  initial_eps=1e-6,ire=1.0, verbose=False):
        
        self.verbose = verbose
        

         
        self.momentum= None
        
        
        
        
        self.initial_eps=initial_eps
        self.ire=ire


    def update_P(self, full_rank_grad):
        return None 
       
    def update_direction(self, full_rank_grad,theta1,pows,RMS_pow=1):   
        pow_diag,pow=pows       
        original_type=full_rank_grad.data.dtype 
        full_rank_grad_high=full_rank_grad.to(torch.float32)         
        
        if self.momentum is None: 
            self.momentum=full_rank_grad_high
        
        self.momentum=theta1*self.momentum+(1-theta1)*full_rank_grad_high

        norm_square=torch.sum(self.momentum**2)+1e-8
        m=self.momentum.numel()
          
        
        
        RMS_inv=torch.sqrt(m/norm_square)  
        update=RMS_inv**(RMS_pow+2*pow_diag)*torch.sign(self.momentum)*torch.abs(self.momentum)**(1+2*pow_diag)

        
        return  update.to(original_type)
    
    def update_direction_sharp(self, full_rank_grad,theta1,pows,RMS_pow):
        return self.update_direction(full_rank_grad,theta1,pows,RMS_pow)
    
    def update_direction_flat(self, full_rank_grad,theta1,pows,RMS_pow):
        return full_rank_grad*0.0



def zero_mask(tensor,threshold=100):
    # 创建一个布尔掩码，标记需要设置为 0 的元素
    mask = (tensor.abs() > threshold)  | torch.isinf(tensor) | torch.isnan(tensor) 
    
    
    tensor[mask] = 0
    
    return tensor

def neg_mask(tensor):
    # 创建一个布尔掩码，标记需要设置为 0 的元素
    mask = (tensor < 0.0)  
    
    
    tensor[mask] = 0
    
    return tensor

# 从UNM.py复制的优化器类，重命名为UnbalancedMomentum
class UnbalancedMomentum(Optimizer):
    """
    Implements shampoo algorithm.

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        thetas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            shp's thetas parameters (theta1 (moving avg for othor_proj) , theta2(decay for momemtum) ).
        eps (`float`, *optional*, defaults to 1e-03):
            shp's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        named_parameters : Iterable[Tuple[str, nn.Parameter]],
        #params: Iterable[nn.parameter.Parameter],
        update_proj_gap:int =10,
        subspace_ratio: float = 1e-2,
        lr: float = 1e-3,
        thetas: Tuple[float, float] = (0.95, 0.99),
        damping_threshold: float = 1e-6,
        weight_decay: float = 0.0,
        initial_eps: float = 1e-6,
        powers: Tuple[float, float] = (-0.5, -0.5),
        RMS_pow: float = 1.0,
        ire: float = 1.0,
        n_head: int = 8,
    ):
        
        self.named_parameters = named_parameters
        
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if powers[1] > 0.0:
            raise ValueError(f"Invalid power: {powers[1]} - should be < 0.0")
        if not 0.0 <= thetas[0] < 1.0:
            raise ValueError(f"Invalid thetas parameter: {thetas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= thetas[1] < 1.0:
            raise ValueError(f"Invalid thetas parameter: {thetas[1]} - should be in [0.0, 1.0)")
        
        defaults = {"lr": lr, "thetas": thetas, "damping_threshold": damping_threshold, "weight_decay": weight_decay}
        
        self.RMS_pow=RMS_pow
         
        self.pows=powers 
        self.initial_eps=initial_eps
        self.damping_threshold=damping_threshold 
        self.n_head=n_head
        self.ire=ire
        self.update_proj_gap=update_proj_gap
        optim_groups = []
        self.qk_modules_list = ["q_proj", "k_proj"]
        self.emb_modules_list = ["embed"]
        self.out_modules_list = ["head"]
        self.norm_modules_list= ["norm"]
        for param_name, param in named_parameters:
            param_name = param_name.lower()
            if not param.requires_grad:
                continue
            
            state = {}
            state["name"] = param_name
            state["params"] = param
            
            state["subspace_ratio"] = subspace_ratio
            if any(name in param_name for name in self.qk_modules_list):
                
                state["Projector"] = UNM_Projector_qk(
                    max_subspace_ratio=state["subspace_ratio"],
                    initial_eps=self.initial_eps,
                    ire=ire,
                    n_head=self.n_head,
                    )
                state["layer_type"] = 'S_F'
            elif any(name in param_name for name in self.emb_modules_list):
                
                state["Projector"] = UNM_Projector_default(
                    max_subspace_ratio=state["subspace_ratio"],
                    mode='right',
                    initial_eps=self.initial_eps,
                    ire=ire,
                    log_E=True
                    )
                state["layer_type"] = 'S_F'
            elif any(name in param_name for name in self.out_modules_list):
                
                state["Projector"] = UNM_Projector_default(
                    max_subspace_ratio=state["subspace_ratio"],
                    mode='right',
                    initial_eps=self.initial_eps,
                    ire=ire,
                    )
                state["layer_type"] = 'S_F'    
            elif any(name in param_name for name in self.norm_modules_list): 
                
                state["Projector"] = UNM_Projector_norm(initial_eps=self.initial_eps,ire=ire)
                state["layer_type"] = 'S'
            else:   
                m,n=param.size(0), param.size(1)
                if n<=m:
                    mode='right'
                    log_E=False
                else: 
                    mode='left'
                    log_E=True
                state["Projector"] = UNM_Projector_default(
                    max_subspace_ratio=state["subspace_ratio"],
                    mode=mode,
                    initial_eps=self.initial_eps,
                    ire=ire,
                    log_E=log_E
                    )
                state["layer_type"] = 'S_F'
                
            optim_groups.append(state)

        
        super().__init__(optim_groups, defaults)    
            
            
            
        #super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError("UNM does not support sparse gradients, please consider SparseAdam instead")
                    
                #注意：若参数ndim=1，需要提前设置梯度和动量为(n,1)形状
                if grad.ndim==1:
                    grad =grad.unsqueeze(1) 
                    
                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                
                
                theta1, theta2 = group["thetas"] 
                
                subroutine_t=state["step"]% self.update_proj_gap 

                if subroutine_t==0:
                    #子循环开始，更新投影阵
                    group["Projector"].update_P(grad)
                
                # if 0<= subroutine_t <self.T_sharp :
                #     update = group["Projector"].update_direction_sharp(
                #         grad, theta1) 
                # else:
                #     update = group["Projector"].update_direction_flat(
                #         grad, theta1) 
                update1 = group["Projector"].update_direction_sharp(
                         grad, theta1,self.pows,self.RMS_pow)
                update2 = group["Projector"].update_direction_flat(
                         grad, theta2,self.pows,self.RMS_pow)
                #不同子空间用不同decay
                update=update1+self.ire*update2
                if "update_norm" not in state:
                    state["update_norm"] = torch.norm(update)
                    
                # gamma=1.01
                # if torch.norm(update)>gamma*state["update_norm"]:
                #     update =update *gamma*state["update_norm"]/(torch.norm(update)+1e-8)

                state["update_norm"]=torch.norm(update)
                
                state["step"] += 1

                

                p.add_(update.squeeze(), alpha=-group["lr"])  #squeeze()是为了处理norm层

                with open("training_log_update.txt", "a") as f:
                    eigen_sum_ratio=torch.mean(update**2)
                    f.write(f"{update.size():}, {eigen_sum_ratio}\n")
               

                
                if group["weight_decay"] > 0.0:
                    
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

        return loss
