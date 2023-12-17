import model


def get_Net(args):
    if args.use_canny:
        print("use UNet_canny")
        net = model.UNet_canny()
        print(net)
        return net
    else:
        print("use UNet_ori")
        net = model.UNet_ori()
        print(net)
        return net

def get_align():
    net_Ori = model.UNet_ori()
    net_Canny = model.UNet_canny()
    return net_Ori, net_Canny