import torch
from geomloss import SamplesLoss

class FixSupportSinkhornBarycenter:
    def __init__(self, source_distributions_supports, source_distributions_weights, step_size, backend, blur=.05, p=2):
        # ==============================================================================================================
        #   initialize the supports for the source measures and the barycenter
        # ==============================================================================================================
        self.source_distributions_supports = source_distributions_supports
        self.min_max_range = self.find_min_max_range(source_distributions_supports)
        self.barycenter_support = self._generate_barycenter_support()
        # ==============================================================================================================
        #   initialize the weights for the source measures and the barycenter
        # ==============================================================================================================
        self.source_distributions_weights = source_distributions_weights
        self.barycenter_weight = ((1 / self.barycenter_support.shape[0]) * torch.ones(self.barycenter_support.shape[0], dtype=torch.float32, requires_grad=True)).cuda()
        # ==============================================================================================================
        #   miscellaneous
        # ==============================================================================================================
        self.n = len(self.source_distributions_supports)
        self.blur = blur
        self.p = p
        self.η = step_size
        self.ε = self.blur**self.p
        self.iter_count = 1
        if backend == "keops":
            sample_loss_backend = "online"
        else:
            sample_loss_backend = "auto"
        self.loss_operator = SamplesLoss(loss="sinkhorn", p=self.p, blur=self.blur, backend=sample_loss_backend,
                                         debias=True, scaling=0.9)
        self.potential_operator = SamplesLoss(loss="sinkhorn", p=self.p, blur=self.blur, backend=sample_loss_backend,
                                              potentials=True, debias=True, scaling=0.9)
        self.backend = backend

    def step(self):
        # ==============================================================================================================
        #   compute [∂ S(α(ω))/∂ ω], where ω is the weight of the barycenter, α(ω) is the barycenter with fixed support
        # ==============================================================================================================
        gradient_ω = 0
        report_loss = 0
        for i in range(self.n):
            loss = self.loss_operator(self.barycenter_weight, self.barycenter_support,
                                       self.source_distributions_weights[i], self.source_distributions_supports[i])
            gradient_ω += torch.autograd.grad(loss, self.barycenter_weight)[0]
            report_loss += loss.detach()

        gradient_ω_2 = 0
        for i in range(self.n):
            f, _ = self.potential_operator(self.barycenter_weight, self.barycenter_support,
                                       self.source_distributions_weights[i], self.source_distributions_supports[i])
            gradient_ω_2 += f
        print(torch.norm(gradient_ω - gradient_ω_2)/torch.norm(gradient_ω_2))
        value, index = torch.min(gradient_ω, 0)
        # ==============================================================================================================
        #   update the barycenter via Frank-Wolfe
        # ==============================================================================================================
        # print(value -  torch.dot(gradient_ω, self.barycenter_weight))
        step_size = self.η/(self.iter_count+2)
        self.barycenter_weight *= 1-step_size
        # print(torch.sum(self.barycenter_weight))
        self.barycenter_weight[index] += step_size
        # print(torch.sum(self.barycenter_weight))
        self.barycenter_weight /= torch.sum(self.barycenter_weight)  # ensure the weights sum up to one
        self.iter_count += 1
        return report_loss

    def find_min_max_range(self, source_distributions_supports):

        # number of distributions
        self.num_distributions = len(source_distributions_supports)

        # dimension of the ambient space
        self.d = source_distributions_supports[0].size(1)

        # save a tensor filled with all support points of all distributions
        self.full_support = torch.cat([nu for nu in source_distributions_supports],dim=0)

        # smallest cube containing all distributions (hence also the barycenter)
        # oriented as a 2 x d vector (first row "min" second row "max")
        return torch.cat((self.full_support.min(0)[0].view(-1,1),
                                        self.full_support.max(0)[0].view(-1,1)), dim=1).t()

    def _generate_barycenter_support(self, grid_step = 50, grid_margin_percentage=0.05):
        margin = (self.min_max_range[0,:] - self.min_max_range[1,:]).abs()*grid_margin_percentage

        tmp_ranges = [torch.arange((self.min_max_range[0,i]-margin[i]).item(),
                                   (self.min_max_range[1,i]+margin[i]).item(),
                                   ((self.min_max_range[1,i]-self.min_max_range[0,i]).abs()+2*margin[i]).item()/grid_step,
                                   dtype=torch.float32)
                      for i in range(self.d)]

        tmp_meshgrid = torch.meshgrid(*tmp_ranges)

        return torch.cat([mesh_column.reshape(-1,1) for mesh_column in tmp_meshgrid],dim=1).cuda()
