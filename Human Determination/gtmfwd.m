function mix = gtmfwd(net)
%GTMFWD	Forward propagation through GTM.

net.gmmnet.centres = rbffwd(net.rbfnet, net.X);
mix = net.gmmnet;
