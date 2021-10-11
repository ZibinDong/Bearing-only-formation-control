# Bearing-Only formation control with a global reference frame

为了正常运行，首先需要给出编队有向图结构，也就是邻接矩阵H，还有初始位置向量p。

只需要给出有向图连接的起始点集和终点集，就可以调用controller类的build_H函数生成图的邻接矩阵H

```python
start_points = [0,0,1,1,2,2,3,3,4,4,5,5]
end_points = [1,2,2,3,3,4,4,5,5,0,0,1]
H = model.build_H(start_points, end_points)

# 导入H矩阵
model.import_H(H)
# 设定控制问题的维数，本例中进行平面编队，所以设置dim=2
model.set_dim(2)
```

当然，对应于现实问题时，结点的连接代表了这两个结点的相对方位是可以获知的，有向图的方向并不表示一定要从某个结点获知另一个结点的方位，由于采用纯方位控制，为了给出方位，首先要确定是从哪里指向哪里，因此程序后面所使用的e向量，g向量全都需要服从邻接矩阵给出的图的指向。

同时，图中的每一个边的方位信息，只需要由一对结点给出即可

![image-20211011171306128](./readme.assets/image-20211011171306128.png)

类似上图的五芒星结构，实际上每一个结点只需要获知两个方位向量就可以实现。



设置一个相对混乱的初始位置可以使用build_random_p函数，只需要给出点的数量和维数，就可以在中心为centroid，边长为r的方体内生成随机初始点向量p，其中centroid和r都是函数的可选参数

```python
# 可选参数r=5，centroid=[0,0]
p = build_random_p(6, 2)
model.import_initial_p(p)
```



下一步就可以输入期望的队形位置了，理论上只需要给出期望的相对方位向量g_exp，就可以通过控制率给出速度，但是对于很多图形，直接给出g_exp还是很麻烦的。可以先给出期望位置p_exp，再用controller的p_exp2g_exp函数计算出g_exp向量（这里的p_exp并不是队形最终会变化到的位置向量，只是为了描述编队结构方位用的，所以每个点相对位置的远近并不重要，只需要方位和你的预期相同即可）

```python
# import g_exp
p_exp = [-1.732,1,0,2,1.732,1,1.732,-1,0,-2,-1.732,-1]
[g_exp, is_infinitesimal_bearing_rigid] = model.p_exp2g_exp(p_exp)
if is_infinitesimal_bearing_rigid:
    model.import_g_exp(g_exp)
else:
    print("The expected structure is not infinitesimal bearing rigid")
    exit(0)
```

p_exp2g_exp函数会返回两个参数，一个是g_exp向量，还有一个是检验结构是否是infinitesimal bearing rigid的布尔值，只有满足这个条件时，结构才能被唯一确定。



```python
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
```

上面的代码用于进行模拟。

- epsilon变量表达类似于步长或者学习率的意义，因为控制律本质上是根据梯度下降设计的，所以这个变量能改变系统的运行速度
- history保存了每次迭代中的p向量，如果是进行2维的模拟，最后可以调用controller的plot2D函数观察运动轨迹
- diff变量表征了实际方位和期望方位的偏差，修改while循环条件能够改变要求的精度

![image-20211011171306128](./readme.assets/image-20211011171306128.png)

上图就是某一次模拟的运行结果，灰点表示初始位置，蓝点表示最终位置，灰线表示运动轨迹，蓝线则给出了图的连接关系。