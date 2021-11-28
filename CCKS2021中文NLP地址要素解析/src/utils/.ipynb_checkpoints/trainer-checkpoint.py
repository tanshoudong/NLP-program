import os
import copy
import torch
import logging
from torch.cuda.amp import autocast as ac
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from src.utils.attack_train_utils import FGM, PGD
from src.utils.functions_utils import load_model_and_parallel, swa
from src.utils.evaluator import model_evaluate,model_evaluate_mrc
from src.utils.conlleval import calculte_metrics
from src.utils.dataset_utils import infer,infer_mrc
import pdb

logger = logging.getLogger(__name__)


def save_model(opt, model,epoch):
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info(f'Saving model & optimizer & scheduler checkpoint to {opt.output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(opt.output_dir, 'checkpoint_{}_epoch.pt'.format(epoch+1)))


def build_optimizer_and_scheduler(opt, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))


    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def train(opt, model, train_dataset,dev_dataset,test_dataset,ent2id):
    if opt.task_type in ['span','crf']:
        fn = train_dataset.collate_fn
    else:
        fn = train_dataset.collate_fn_mrc
        
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opt.train_batch_size,
                              num_workers=0,
                              collate_fn=fn,
                              shuffle=True)

    dev_loader = DataLoader(dataset=dev_dataset,
                              batch_size=opt.train_batch_size,
                              num_workers=0,
                              collate_fn=fn,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=opt.train_batch_size,
                            num_workers=0,
                            collate_fn=fn,
                            shuffle=False)

    scaler = None
    if opt.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    model, device = load_model_and_parallel(model, opt.gpu_ids)
    swa_raw_model = copy.deepcopy(model)

    use_n_gpus = False
    if hasattr(model, "module"):
        use_n_gpus = True

    t_total = len(train_loader) * opt.train_epochs

    optimizer, scheduler = build_optimizer_and_scheduler(opt, model, t_total)

    # Train
    logger.info("***** Running training *****")
    logger.info(f"  Num Examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {opt.train_epochs}")
    logger.info(f"  Total training batch size = {opt.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0

    model.zero_grad()

    fgm, pgd = None, None

    attack_train_mode = opt.attack_train.lower()
    if attack_train_mode == 'fgm':
        fgm = FGM(model=model)
    elif attack_train_mode == 'pgd':
        pgd = PGD(model=model)

    pgd_k = 3

    save_steps = t_total // opt.train_epochs
    eval_steps = save_steps

    logger.info(f'Save model in {save_steps} steps; Eval model in {eval_steps} steps')

    log_loss_steps = 10

    avg_loss = 0.

    best_f1 = 0
    best_model = None

    for epoch in range(opt.train_epochs):
        torch.cuda.empty_cache()

        for step, batch_data in enumerate(train_loader):

            model.train()

            del batch_data['raw_text']
            if opt.task_type=='span':
                del batch_data['labels']
            
            if opt.task_type=="mrc":
                del batch_data['entitys']
                del batch_data['querys']
                del batch_data['labels']
            
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)

            if opt.use_fp16:
                with ac():
                    model(**batch_data)
                    loss = model(**batch_data)[0]
            else:
                loss = model(**batch_data)[0]

            if use_n_gpus:
                loss = loss.mean()

            if opt.use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if fgm is not None:
                fgm.attack()

                if opt.use_fp16:
                    with ac():
                        loss_adv = model(**batch_data)[0]
                else:
                    loss_adv = model(**batch_data)[0]

                if use_n_gpus:
                    loss_adv = loss_adv.mean()

                if opt.use_fp16:
                    scaler.scale(loss_adv).backward()
                else:
                    loss_adv.backward()

                fgm.restore()

            elif pgd is not None:
                pgd.backup_grad()

                for _t in range(pgd_k):
                    pgd.attack(is_first_attack=(_t == 0))

                    if _t != pgd_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()

                    if opt.use_fp16:
                        with ac():
                            loss_adv = model(**batch_data)[0]
                    else:
                        loss_adv = model(**batch_data)[0]

                    if use_n_gpus:
                        loss_adv = loss_adv.mean()

                    if opt.use_fp16:
                        scaler.scale(loss_adv).backward()
                    else:
                        loss_adv.backward()

                pgd.restore()

            if opt.use_fp16:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

            # optimizer.step()
            if opt.use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            model.zero_grad()

            global_step += 1

            if global_step % save_steps == 0:
                avg_loss /= log_loss_steps
                logger.info('Step: %d / %d ,total loss: %.5f,best_f1:%.4f' % (epoch,opt.train_epochs,avg_loss,best_f1))
                avg_loss = 0.
            else:
                avg_loss += loss.item()
        
        if opt.task_type=="mrc":
            model_evaluate_mrc(model, dev_loader, opt, device, ent2id)
        else:
            model_evaluate(model, dev_loader, opt, device, ent2id)
        f1 = calculte_metrics(opt)
        if f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(model)
            
        if epoch >= opt.swa_start:
            save_model(opt, model, epoch)

    
    #模型权重指数平滑
    print('********************model weights smooth***************************')
    swa_model = swa(swa_raw_model, opt.output_dir)
    if opt.task_type=="mrc":
        model_evaluate_mrc(swa_model, dev_loader, opt, device, ent2id)
    else:
        model_evaluate(swa_model, dev_loader, opt, device, ent2id)
    calculte_metrics(opt)
    #测试集
    print('***************make test********************')
    if opt.task_type=="mrc":
        infer_mrc(swa_model,test_loader,opt,device,ent2id)
    else:
        infer(swa_model,test_loader,opt,device,ent2id)

    # clear cuda cache to avoid OOM
    torch.cuda.empty_cache()
    logger.info('Train done')