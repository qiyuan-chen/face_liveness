import argparse


def get_argparse():
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_sup", default=500, type=int, help="有监督样本数量")
    parser.add_argument("--train_max_seq_length",
                        default=512, type=int, help="训练集的最大长度")
    parser.add_argument("--per_gpu_train_batch_size",
                        default=8, type=int, help="训练Batch size的大小")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=20, type=int, help="验证Batch size的大小")
    # 训练的参数
    parser.add_argument("--model", type=str, help="使用哪一个增强模型")
    parser.add_argument("--do_distri_train",
                        action="store_true", help="是否用两个卡并行训练")
    parser.add_argument("--gpu_num", default=1, help="默认使用的GPU编号")
    parser.add_argument("--model_name_or_path",
                        default="/data/zhoujx/prev_trained_model/chinese_roberta_wwm_ext_pytorch", type=str, help="预训练模型的路径")
    parser.add_argument("--num_train_epochs", default=50.0,
                        type=float, help="训练轮数")
    parser.add_argument("--early_stop", default=4, type=int, help="早停")
    parser.add_argument("--learning_rate", default=1e-5,
                        type=float, help="transformer层学习率")
    parser.add_argument("--linear_learning_rate",
                        default=1e-3, type=float, help="linear层学习率")
    parser.add_argument("--weight_decay", default=0.01,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--output_dir", default="./output",
                        type=str, help="保存模型的路径")
    parser.add_argument("--parallel", default=False, help="是否并行训练")
    parser.add_argument("--epoch_max_step", type=int, default=150, help="")
    # UDA的参数
    parser.add_argument("--unsup_ratio", type=int, default=4, help="")
    parser.add_argument("--tsa", type=str, help="")
    parser.add_argument("--total_steps", type=int, default=10000, help="")
    parser.add_argument("--uda_confidence_thresh",
                        type=float, default=0, help="")
    parser.add_argument("--uda_softmax_temp", type=float, default=0.7, help="")
    parser.add_argument("--uda_coeff", type=float, default=1, help="")

    parser.add_argument("--dataset_path", default='./data/', help="数据集地址")
    parser.add_argument("--task_id", type=int, default=0, help="任务id")

    return parser


''' 
n_filters = 200
filter_sizes = [2, 3, 5, 7, 9, 11]
'''
