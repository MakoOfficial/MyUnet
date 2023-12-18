import model
import os

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


# 保存原始的print函数，以便稍后调用它。
rewrite_print = print


# 定义新的print函数。
def print(*arg):
    # 首先，调用原始的print函数将内容打印到控制台。
    rewrite_print(*arg)

    # 如果日志文件所在的目录不存在，则创建一个目录。
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开（或创建）日志文件并将内容写入其中。
    log_name = 'log.txt'
    filename = os.path.join(output_dir, log_name)
    rewrite_print(*arg, file=open(filename, "a"))
