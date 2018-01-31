import torch.nn as nn
from ..functions.gaterecurrent2dnoind import GateRecurrent2dnoindFunction

class GateRecurrent3Way(nn.Module):
    """docstring for ."""
    def __init__(self):
        super(GateRecurrent3Way, self).__init__()

    def forward(self, X, G, dim=1):
        G = self.generate_gate(G)
        out = self.forward_iteration(X, G)
        out = self.forward_iteration(out, G)

        return out

    def forward_iteration(self, X, G):
        out = self.forward_one_direction(X, G[0], True, False) # left->right
        out = torch.max(out, self.forward_one_direction(X, G[1], True, True)) # right->left
        out = torch.max(out, self.forward_one_direction(X, G[2], False, False)) # top->bottom
        out = torch.max(out, self.forward_one_direction(X, G[3], False, True)) # bottom->top
        return out

    def forward_one_direction(self, X, G, horizontal, reverse):
        return GateRecurrent2dnoindFunction(horizontal, reverse)(X, *G)

    def generate_gate(self, G, dim=1)
        _,c,_,_ = G.size()
        split_size_or_sections = c // 12
        G_list = torch.split(G, split_size_or_sections, dim=dim)
        outs = []
        for d in range(4):
            s_i = d * 3
            e_i = (d + 1) * 3
            G1, G2, G3 = G_list[s_i:e_i]

            sum_abs = G1.abs() + G2.abs() + G3.abs()
            mask_need_norm = sum_abs.ge(1)
            mask_need_norm = mask_need_norm.float()
            G1_norm = torch.div(G1, sum_abs)
            G2_norm = torch.div(G2, sum_abs)
            G3_norm = torch.div(G3, sum_abs)

            G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
            G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
            G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm

            outs.append([G1, G2, G3])

        return outs