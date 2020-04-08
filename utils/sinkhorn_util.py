from geomloss.sinkhorn_samples import softmin_online, keops_lse, log_weights
from functools import partial

cost_formulas = {
    1 : "Norm2(X-Y)",
    2 : "(SqDist(X,Y) / IntCst(2))",
}

def sinkhorn_potential(α, x, β, y, b_x_particle, a_y_particle, blur, p):
    # a_y_particle & b_x_particle are sinkhorn potential vectors to the particle-approximated version of OT_\epsilon

    N, D_x = x.shape
    M, D_y = y.shape

    assert(D_x == D_y)
    cost = cost_formulas[p]
    softmin = partial(softmin_online, log_conv=keops_lse(cost, D_x, dtype=str(x.dtype)[6:]))
    C_xy, C_yx = ( (x, y.detach()), (y, x.detach()) )

    ε = blur ** p

    α_log, β_log, = log_weights(α), log_weights(β)
    λ = 1
    # a_y_neural = λ * softmin(ε, C_yx, (α_log + b_x_particle / ε).detach())
    b_x_neural = λ * softmin(ε, C_xy, (β_log + a_y_particle / ε).detach())

    return b_x_neural