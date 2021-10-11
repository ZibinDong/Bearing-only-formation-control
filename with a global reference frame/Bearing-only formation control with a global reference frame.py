import numpy as np
import math as m
import matplotlib.pyplot as plt


def eye_matrix(list_m):
    n = 0
    index = 0
    for each in list_m:
        n += each.shape[0]
    eye_m = np.zeros((n,n))
    for each in list_m:
        m = each.shape[0]
        for i in range(m):
            for j in range(m):
                eye_m[index+i, index+j] = each[i,j]
        index += m
    return eye_m
        
def build_random_p(n, dim, r = 5, o = [0,0]):
    p = []
    for i in range(n*dim):
        p.append(r*2*(np.random.rand()-0.5))
    return np.array(p)


class controller:
    def __init__(self):
        self.dim = 2
        self.Id = np.eye(2)
        self.n = 4
        self.m = 5
        self.H = []
        self.H_bar = []
        self.p = []
        self.p_init = []
        self.e = []
        self.e_norm = []
        self.g = []
        self.g_exp = []
        self.centroid_init = []
        self.scale_init = []
        pass

    # points list is supposed to start from 0
    def build_H(self, start_points, end_points):
        n = max(start_points)+1
        m = len(start_points)
        H = np.zeros((m,n))
        for i in range(m):
            H[i, start_points[i]] = -1
            H[i, end_points[i]] = 1
        return H
            

    def set_dim(self, dim):
        self.dim = dim
        return

    # matrix H would form G
    def import_H(self, H):
        self.H = H
        (self.m, self.n) = H.shape
        self.Id = np.eye(self.dim)
        self.H_bar = np.kron(H,self.Id)
        return

    # import intial p 
    def import_initial_p(self, p):
        self.p = p
        self.p_init = p
        [self.centroid_init, _,_,self.scale_init] = self.compute_centroid_scale(p)
        return

    # compute e after import H and p
    def compute_e(self, p):
        e = np.matmul(self.H_bar, np.transpose(p))
        g = np.copy(e)
        e_norm = []
        for i in range(self.m):
            norm_e = np.linalg.norm([e[self.dim*i],e[self.dim*i+1]])
            e_norm.append(norm_e)
            for d in range(self.dim):
                g[self.dim*i+d] = e[self.dim*i+d]/norm_e
        return [e, e_norm, g]

    def P_x(self, x):
        d = x.size
        x = np.reshape(x, [d,1])
        P_x = np.eye(d) - (np.matmul(x,np.transpose(x)))/np.linalg.norm(x)/np.linalg.norm(x)
        return P_x

    # compute bearing rigidity matrix
    def R_p_and_diagP(self, p):
        tmp_diag = []
        [e, e_norm, g] = self.compute_e(p)
        for i in range(self.m):
            gk = np.array([g[self.dim*i+d] for d in range(self.dim)])
            Pgk = self.P_x(gk)
            tmp_diag.append(Pgk/e_norm[i])
        tmp_diag = eye_matrix(tmp_diag)
        R_p = np.matmul(tmp_diag, self.H_bar)
        return [R_p, tmp_diag]

    def import_g_exp(self, g_exp):
        self.g_exp = np.copy(g_exp)
        for i in range(self.m):
            norm_g = np.linalg.norm([g_exp[self.dim*i+d] for d in range(self.dim)])
            for d in range(self.dim):
                self.g_exp[self.dim*i+d] = g_exp[self.dim*i+d]/norm_g
        return

    def compute_v(self, diag_P):
        v = self.H_bar.T@diag_P@self.g_exp
        return v

    def p_exp2g_exp(self, p_exp):
        [_,_,g_exp] = self.compute_e(p_exp)
        [R_p_exp, _] = self.R_p_and_diagP(p_exp)
        is_infinitesimal_bearing_rigid = True
        if np.linalg.matrix_rank(R_p_exp) == self.dim*self.n - self.dim - 1:
            is_infinitesimal_bearing_rigid = True
        else:
            is_infinitesimal_bearing_rigid = False
        return [g_exp, is_infinitesimal_bearing_rigid]

    def plot2D(self, history):
        History = np.array(history)
        plt.figure()
        for i in range(self.n):
            plt.plot(History[:,self.dim*i], History[:,self.dim*i+1],'lightgray')
        init_p = [[] for _ in range(self.dim)]
        for i in range(self.n):
            for j in range(self.dim):
                init_p[j].append(History[0,self.dim*i+j])
        plt.plot(init_p[0], init_p[1],'o',color='lightgray')
        end_p = [[] for _ in range(self.dim)]
        for i in range(self.n):
            for j in range(self.dim):
                end_p[j].append(History[-1,self.dim*i+j])
        plt.plot(end_p[0], end_p[1],'o',color='steelblue')
        for i in range(self.m):
            hline = self.H[i,:]
            start_point = 0
            end_point = 0
            for j in range(self.n):
                if hline[j] == -1:
                    start_point = j
                if hline[j] == 1:
                    end_point = j
            plt.plot(
                [end_p[0][start_point], end_p[0][end_point]], 
                [end_p[1][start_point], end_p[1][end_point]], 
                'lightblue'
            )
        plt.axis('equal')
        plt.show()

    def compute_centroid_scale(self, p):
        centroid = np.zeros(self.dim)
        for i in range(self.n):
            centroid = (i/(i+1))*centroid + (1/(i+1))*p[self.dim*i:(self.dim*i+self.dim)]
        scale = 0
        for i in range(self.n):
            scale = ((i/(i+1))*scale) + (1/(i+1))*np.dot(p[self.dim*i:(self.dim*i+self.dim)]-centroid,p[self.dim*i:(self.dim*i+self.dim)]-centroid)
        scale = m.sqrt(scale)

        one = np.ones((self.n, 1))
        centroid2 = np.kron(one, self.Id).T@p/self.n
        scale2 = np.linalg.norm(np.reshape(p,[self.dim*self.n,1])-np.kron(one, np.reshape(centroid2, [self.dim, 1])))/m.sqrt(self.n)
        return [centroid, scale, centroid2, scale2]




model = controller()

# import H
start_points = [0,0,1,1,2,2,3,3,4,4,5,5]
end_points = [1,2,2,3,3,4,4,5,5,0,0,1]
H = model.build_H(start_points, end_points)
# H = np.array([[-1,1,0,0],[-1,0,1,0],[-1,0,0,1],[0,-1,0,1],[0,0,-1,1]])
model.import_H(H)
model.set_dim(2)

# build random initial p
p = build_random_p(6, 2)
model.import_initial_p(p)

# compute e
[model.e, model.e_norm, model.g] = model.compute_e(p)
[R_p,diag_P] = model.R_p_and_diagP(p)

# import g_exp
p_exp = [-1.732,1,0,2,1.732,1,1.732,-1,0,-2,-1.732,-1]
[g_exp, is_infinitesimal_bearing_rigid] = model.p_exp2g_exp(p_exp)
if is_infinitesimal_bearing_rigid:
    model.import_g_exp(g_exp)
else:
    print("The expected structure is not infinitesimal bearing rigid")
    exit(0)

diff = np.sum(np.abs(model.g-model.g_exp))

# simulation
epsilon = 10.0
history = [np.copy(model.p)]
while diff > 0.001:
# if True:
    v = model.compute_v(diag_P)
    new_p = model.p + epsilon*v
    model.p = np.copy(new_p)
    [model.e, model.e_norm, model.g] = model.compute_e(model.p)
    [R_p,diag_P] = model.R_p_and_diagP(model.p)
    diff = np.sum(np.abs(model.g-model.g_exp))
    history.append(np.copy(model.p))
    print(f'diff={diff}')

model.plot2D(history)
    
[centroid, scale, centroid2, scale2] = model.compute_centroid_scale(model.p)



