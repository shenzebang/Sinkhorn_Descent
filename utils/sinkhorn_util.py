from geomloss.sinkhorn_samples import softmin_online, keops_lse, log_weights
from functools import partial
from geomloss.sinkhorn_samples import squared_distances
cost_formulas = {
    1 : "Norm2(X-Y)",
    2 : "(SqDist(X,Y) / IntCst(2))",
}

def softmin_tensorized(ε, C, f):
    return - ε * ( f.view(1,-1) - C/ε).logsumexp(dim=1).view(-1)

def sinkhorn_potential(α, x, β, y, b_x_particle, a_y_particle, blur, p, backend="online"):
    # a_y_particle & b_x_particle are sinkhorn potential vectors to the particle-approximated version of OT_\epsilon

    N, D_x = x.shape
    M, D_y = y.shape

    assert(D_x == D_y)
    if backend == "online":
        cost = cost_formulas[p]
        softmin = partial(softmin_online, log_conv=keops_lse(cost, D_x, dtype=str(x.dtype)[6:]))
        C_xy = (x, y.detach())
        ε = blur ** p
        β_log = log_weights(β)
        λ = 1
        b_x_neural = λ * softmin(ε, C_xy, (β_log + a_y_particle / ε).detach())
    elif backend == "tensorized":
        C_xy = squared_distances(x, y.detach())
        ε = blur ** p
        β_log = β.log()
        λ = 1
        b_x_neural = λ * softmin_tensorized(ε, C_xy, (β_log + a_y_particle / ε).detach())
        # print(b_x_neural.shape)

    else:
        b_x_neural = None
        print("unidentified backend")

    return b_x_neural

def potential_operator_grad(α, x, β, y, f_x_value, blur, p, backend="tensorized", niter=10):
    N, D_x = x.shape
    M, D_y = y.shape

    assert (D_x == D_y)
    if backend is not "tensorized":
        print("unidentified backend")
        return None
    ε = blur ** p
    β_log = β.log()
    α_log = α.log()
    C_xy = squared_distances(x, y)
    C_yx = squared_distances(y, x)
    f_x_value_grad = f_x_value
    for _ in range(niter):
        g_y_value_grad = softmin_tensorized(ε, C_yx, (α_log + f_x_value_grad / ε))
        f_x_value_grad = softmin_tensorized(ε, C_xy, (β_log + g_y_value_grad / ε))

    # perform update in parallel!
    f_x_value_grad, g_y_value_grad = softmin_tensorized(ε, C_xy, (β_log + g_y_value_grad / ε)), \
                                     softmin_tensorized(ε, C_yx, (α_log + f_x_value_grad / ε))

    return f_x_value_grad, g_y_value_grad

