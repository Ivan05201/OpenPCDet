import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

# 参数设置，允许用户从命令行输入，一共两个参数，一个是命令行参数args，一个是配置文件cfg
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser') # ArgumentParser对象，从命令行字符串中解析参数
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training') # 配置文件路径参数

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    

    args = parser.parse_args() # 解析命令行并存放参数

    cfg_from_yaml_file(args.cfg_file, cfg) # 从YAML文件中加载配置，并存放在cfg对象中
    cfg.TAG = Path(args.cfg_file).stem # 设置cfg对象的TAG属性，Path(args.cfg_file).stem功能是获取配置文件的名称，不包括文件的路径和扩展名，就是例如pointpillar
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml' # 设置cfg对象的EXP_GROUP_PATH属性，用于表示配置文件所在的目录，移除第一个和最后一个
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False) # 设置args参数的use_amp参数，处理混合精度训练选项，

    if args.set_cfgs is not None: # 这一行代码检查是否通过命令行提供了额外的配置选项，如果提供了，就将它们应用到配置对象cfg
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config() # 解析命令行参数，加载配置文件
    if args.launcher == 'none': # 处理分布式训练的设置，如果参数是None，则不适用分布式训练
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        # getattr函数获取一个函数来初始化分布式环境，函数名称是init_dist_，后面跟args.launcher的值；函数被调用并传到args.tcp_port，分布式通信的tcp端口；args.local_rank分布式环境中此进程排名，ncll作为后端
        dist_train = True

    if args.batch_size is None: # 检查命令行是否设置batch_size大小
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU # 如果没有设置，则从cfg中读取
    else: # 如果在命令行中设置了bs
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus' 
        # 这是一个断言语句，确保批量大小可以被GPU数量整除；在分布式训练中，批量大小通常会在多个GPU上均分，如果批量大小不能被GPU的数量整除，那么数据分配将不均匀可能会导致错误
        args.batch_size = args.batch_size // total_gpus # 将批量大小除以GPU的数量，得到每个GPU的批量大小

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs # 设置epoch的数量，一个epoch表示数据集整体通过模型一遍；检查命令行是否设置epoch大小

    if args.fix_random_seed: # 设置随机种子，保证随机的一致性
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag # 路径拼接得到一个存储输出结果的地址
    ckpt_dir = output_dir / 'ckpt' # 存储checkpoints文件
    output_dir.mkdir(parents=True, exist_ok=True) # 创建文件夹，检查文件夹是否已经存在
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 处理日志文件的创建和配置
    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')) # 文件的路径和时间信息
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK) # 调用common_utils中的create_logger函数，创建一个日志记录器

    # log to file
    logger.info('**********************Start logging**********************') # 使用info方法在日志中写入信息，表示日志记录开始
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL' # 检查环境变量CUDA_VISIBLE_DEVICES是否设置，如果设置将包含一个字符串，表示可用于训练的GPU的ID；如果没设置，则所有GPU都可用
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list) # 在日志中记载哪些GPU用于训练

    if dist_train: # 检查是否为分布式训练，并把信息记录到日志中
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')
        
    for key, val in vars(args).items(): # 遍历命令行参数，将每个参数都记录到日志文件中
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger) # 将配置文件的参数记录到日志文件中
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None # 创建一个TensorBoard日志记录器，如果本地进程的排序是0，则记录训练过程中的信息；TensorBoard是一个可视化工具，用于查看神经网络训练过程中的各种指标，如损失和准确性

    # 创建数据加载器、网络、优化器
    logger.info("----------- Create dataloader & network & optimizer -----------")
    # 数据加载器，使用build_dataloader函数
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, # 数据集的配置信息
        class_names=cfg.CLASS_NAMES, # 类别名称
        batch_size=args.batch_size, # 批量大小
        dist=dist_train, workers=args.workers, # 是否分布式
        logger=logger, # 记录数据加载过程中信息
        training=True, # 训练模式
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, # 确定是否将所有迭代合并到一个时期
        total_epochs=args.epochs, # 训练总轮次
        seed=666 if args.fix_random_seed else None # 设置随机种子
    )

    # 根据给定配置创建神经网络模型，模型参数cfg.MODEL，类别数，训练数据集
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn: # 是否使用同步批量归一化，在分布式训练中有帮助
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda() # 模型迁移到GPU
 
    optimizer = build_optimizer(model, cfg.OPTIMIZATION) # 创建优化器

    # load checkpoint if it is possible
    # 从预训练模型或checkpoint恢复训练
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None: # 检查命令行指令是否有预训练模型的地址
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None: # 检查命令行是否有checkpoint文件的指令
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth')) # 搜索指定路径ckpt_dir中所有.pth文件，存储在列表中
              
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime) # 排序，以得到最新的checkpoint文件
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    ) # 加载这个文件
                    last_epoch = start_epoch + 1
                    break # 跳出循环
                except:
                    ckpt_list = ckpt_list[:-1] # 如果在加载过程中发生异常，则删除这个文件并尝试读取前一个文件

    # 训练模式
    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()]) # 分布式训练，用于在多个GPU上进行前向和反向传播
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    # 动态调整学习率
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg
    )

    # 检查训练数据集是否有shared_memory，共享内存可以用于在多进程之间共享数据，从而提高数据加载的效率；在训练结束或完成之后，最好清理共享内存，以释放内存资源并防止潜在的内存泄漏
    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    # 加载测试数据集
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train' # 创建存储评价文件夹
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    # 设置开始评估的轮次
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

    # 调用函数对测试数据集进行评估
    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
