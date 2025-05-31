import argparse
import time
from copy import deepcopy
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


from models.mint import get_mint
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from utils.args import get_args_parser
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask, mini_imagenet_mask
from models.tta import test_time_adapt_eval


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def main():

    args = parser.parse_args()
    set_random_seed(args.seed)

    assert args.gpu is not None
    torch.cuda.set_device(args.gpu)
    print("=> Use GPU: {} for training".format(args.gpu))


    print("=> Model created: visual backbone {}".format(args.arch))
    model = get_mint(args.arch, args.test_sets, args, args.gpu, args.n_ctx, args.ctx_init)
    model = model.cuda(args.gpu)


    if args.load is not None: 
        # print("Use pre-trained soft prompt (CoOp) as initialization")
        pretrained_ctx = torch.load(args.load)['state_dict']['ctx'] 

        assert pretrained_ctx.size()[0] == args.n_ctx
        with torch.no_grad():
            model.text_prompt_learner.ctx.copy_(pretrained_ctx)
            model.text_prompt_learner.ctx_init_state = pretrained_ctx


    for name, param in model.named_parameters():
        if "text_prompt" in name :
            param.requires_grad_(True)
        elif "list" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
        

    trainable_param = list(model.text_prompt_learner.parameters())
    
    trainable_param.extend(model.bank.key_list)
    trainable_param.extend(model.bank.prompt_list)
    optimizer = torch.optim.AdamW(trainable_param, args.lr)
    optim_state = deepcopy(optimizer.state_dict())


    scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True


    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    
    datasets = args.test_sets.split("/")
    results = {}

    for set_id in datasets:

        print("=> evaluating: {}".format(set_id))

        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])])
        data_transform = AugMixAugmenter(base_transform, preprocess, 
                                         n_views=args.batch_size-1, 
                                         augmix=len(set_id)>1)
        batchsize = 1 
        
        if len(set_id) > 1: 
           
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id == 'A':
                label_mask = imagenet_a_mask
                classnames = [classnames_all[i] for i in label_mask]
            elif set_id =='R':
                label_mask = imagenet_r_mask
                classnames = [classnames_all[i] for i in label_mask]
            elif set_id =='V':
                label_mask = imagenet_v_mask
                classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all
        
        model.reset_classnames(classnames, args.arch)

   
        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        # print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
            

        results[set_id] = test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler, args)

        del val_dataset, val_loader
        

    print("\n--- Accuracy Results ---")
    id_width = 15 
    acc_width = 10 
    print(f"{'Set ID':<{id_width}} {'Top-1 Acc (%)':>{acc_width}} {'Top-5 Acc (%)':>{acc_width}}") 
    print(f"{'-'*id_width} {'-'*acc_width} {'-'*acc_width}")
    for id, (top1, top5) in results.items():
        print(f"{id:<{id_width}} {top1:>{acc_width}.2f} {top5:>{acc_width}.2f}") # .2f 表示保留2位小数


if __name__ == '__main__':
    parser = get_args_parser()
    main()