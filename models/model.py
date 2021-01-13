from torch import nn
from models import HGNN_conv, HGraphConvolutionBS
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.layers import GCNII, Dense
import math

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))  # G*x*\theta
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


class GCNModel(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False,
                 args=None,
                 attn=True,
                 ):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(GCNModel, self).__init__()
        self.mixmode = mixmode
        self.dropout = dropout
        self.baseblock = baseblock
        self.nbaselayer = nbaselayer
        self.inputlayer = inputlayer
        self.outputlayer = outputlayer
        self.adj = None
        self.args = args
        self.attn = attn
        if baseblock == "gcnii":
            self.BASEBLOCK = GCNII
        elif baseblock == "mutigcn":
            self.BASEBLOCK = MultiLayerHGNN
        elif baseblock == "gcn":
            self.BASEBLOCK = MultiLayerHGNN
        # elif baseblock == "inceptiongcn":
        #     self.BASEBLOCK = InecptionGCNBlock
        # else:
        #     raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))
        if inputlayer == "gcn":
            # input gc
            self.ingc = HGraphConvolutionBS(nfeat, nhid, activation, withbn, withloop ,args=args, res=False,
                                            )
            baseblockinput = nhid
        elif inputlayer == "none":
            self.ingc = lambda x, y: x
            baseblockinput = nfeat
        else:
            # self.ingc = nn.Linear(nfeat, nhid)
            self.ingc = Dense(nfeat, nhid, activation)
            baseblockinput = nhid

        outactivation = lambda x: x
        if outputlayer == "gcn":
            self.outgc = HGraphConvolutionBS(nhid, nclass, outactivation, withbn, withloop,
                                              res=False
                                             ,args=args,
                                             )
        # elif outputlayer ==  "none": #here can not be none
        #    self.outgc = lambda x: x
        else:
            # self.outgc = nn.Linear(nhid, nclass)
            self.outgc = Dense(nhid, nclass, outactivation)

        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        # for i in xrange(nhidlayer):

        #

        for i in range(nhidlayer):
            if baseblock.lower() == 'gcnii':
                gcb = self.BASEBLOCK(nfeat=None,
                                     nlayers=nbaselayer,
                                     nhidden=nhid,
                                     nclass=None,
                                     dropout=dropout,
                                     lamda=args.lamda,
                                     alpha=args.alpha,
                                     variant=args.variant,
                                     args=args,
                                     )
            else:
                gcb = self.BASEBLOCK(in_features=baseblockinput,
                                     hidden_features=nhid,
                                     nbaselayer=nbaselayer,
                                     withbn=withbn,
                                     withloop=withloop,
                                     activation=activation,
                                     dropout=dropout,
                                     dense=False,
                                     aggrmethod=aggrmethod,
                                     args=args,
                                     res=args.residue,

                                     )
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        if baseblock.lower() == 'gcnii':
            # self.ingc = nn.Linear(nfeat, nhid)
            # self.outgc = nn.Linear(nhid, nclass)
            self.fcs = nn.ModuleList([self.ingc, self.outgc])
            self.params1 = list(self.midlayer.parameters())
            self.params2 = list(self.fcs.parameters())
        self.reset_parameters()
        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)

    def reset_parameters(self):
        pass

    def forward(self, fea, adj, G=None):

        if self.mixmode:
            x = self.ingc(fea, G.cpu())
        else:
            x = self.ingc(fea, adj=adj, G=G)

        x = F.dropout(x, self.dropout, training=self.training)
        if self.mixmode:
            x = x.to(device)

        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(input=x, adj=adj, G=G)

        # output, no relu and dropput here.
        self.embbeding = x
        # 修改2 x = self.outgc(x, adj=adj, G=G, num=(i + 1) * self.nbaselayer + 1)
        x = self.outgc(x, adj=adj, G=G)

        return x

    def embbed(self, fea, adj, G=None):
        _ = self.forward(fea, adj, G)
        return self.embbeding


class MultiLayerHGNN(nn.Module):
    """
        The base block for Multi-layer GCN / ResGCN / Dense GCN
        """

    def __init__(self, in_features, hidden_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=0.5,
                 aggrmethod="nores", dense=False, res=False, incidence_v=None, incidence_e=None,
                 args=None):
        """
        The base block for constructing DeepGCN model.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: enable dense connection
        """
        super(MultiLayerHGNN, self).__init__()
        self.in_features = in_features
        self.hiddendim = hidden_features
        self.nhiddenlayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.hiddenlayers = nn.ModuleList()
        self.baselayer = HGraphConvolutionBS
        # self.baselayer = HGNN_conv
        self.res = res
        self.incidence_v = incidence_v
        self.incidence_e = incidence_e


        self.args = args
        self.__makehidden()
        self.adj = None

    def __makehidden(self):
        # for i in xrange(self.nhiddenlayer):
        for i in range(self.nhiddenlayer):
            if i == 0:
                layer = self.baselayer(self.in_features, self.hiddendim, activation=self.activation, withbn=self.withbn,
                                       res=self.res,
                                        args=self.args)
                # layer = HGraphConvolutionBS(self.in_features, self.hiddendim, self.activation, self.withbn,
                #                             self.withloop)
            else:
                layer = self.baselayer(self.hiddendim, self.hiddendim, activation=self.activation, res=self.res,
                                        withbn=self.withbn
                                       , args=self.args)
                # layer = HGraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn, self.withloop)
            self.hiddenlayers.append(layer)

    def _doconcat(self, x, subx):
        if x is None:
            return subx
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx
        elif self.aggrmethod == "nores":
            return x

    def forward(self, input, adj=None, G=None):
        x = input
        denseout = None
        # Here out is the result in all levels.
        for num, gc in enumerate(self.hiddenlayers):
            denseout = self._doconcat(denseout, x)
            # 修改1 x = gc(input=x, adj=adj, G=G, num=num + 1)
            x = gc(input=x, adj=adj, G=G)
            # if num == self.nhiddenlayer - 1:
            #     continue
            # x = self.activation(x)
            x = F.dropout(x, self.dropout, training=self.training)

        if not self.dense:
            return self._doconcat(x, input)
        return self._doconcat(x, denseout)

    def get_outdim(self):
        return self.hiddendim

#
# class decoder(nn.Module):
#     def __init__(self,in_size,out_size,ffn_num):
#         self.ffn=nn.ModuleList()
#         self.in_size=
