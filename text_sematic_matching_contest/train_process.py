from torch.utils.data import DataLoader
import numpy as np
import copy
import torch
import logging

def train_dev(config,model,train_data,dev_data):
    dev_acc = 0.
    predict_label = []

    # 加载模型
    model_example = copy.deepcopy(model).to(config.device)
    best_model = None
    convert_to_features, build_data_set, train_module, evaluate_module = MODEL_CLASSES[config.use_model]

    if train_data:
        config.train_num_examples = len(train_data)
        # 特征转化
        train_features = convert_to_features(
            examples=train_data,
            tokenizer=tokenizer,
            label_list=config.class_list,
            max_length=config.pad_size,
            data_type='train'
        )
        train_dataset = build_data_set(train_features)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        # dev 数据加载与转换
        if dev_data is not None:
            config.dev_num_examples = len(dev_data)
            dev_features = convert_to_features(
                examples=dev_data,
                tokenizer=tokenizer,
                label_list=config.class_list,
                max_length=config.pad_size,
                data_type='dev'
            )
            dev_dataset = build_data_set(dev_features)
            dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
        else:
            dev_loader = None

        best_model = train_module(config, model_example, train_loader, dev_loader)

        if dev_data is not None:
            dev_acc, dev_loss, total_inputs_err = evaluate_module(config, best_model, dev_loader)
            logger.info('classify error sentences:{}'.format(len(total_inputs_err)))
            # for idx, error_dict in enumerate(total_inputs_err):
            #     tokens = tokenizer.convert_ids_to_tokens(error_dict['sentence_ids'], skip_special_tokens=True)
            #     logger.info('## idx: {}'.format(idx+1))
            #     logger.info('sentences: {}.'.format(''.join(x for x in tokens)))
            #     logger.info('true label: {}'.format(error_dict['true_label']))
            #     logger.info('proba: {}'.format(error_dict['proba']))

            logger.info('evaluate: acc: {0:>6.2%}, loss: {1:>.6f}'.format(dev_acc, dev_loss))

    if test_examples is not None or dev_data is not None:
        if test_examples is None:
            test_examples = dev_data
        test_features = convert_to_features(
            examples=test_examples,
            tokenizer=tokenizer,
            label_list=config.class_list,
            max_length=config.pad_size,
            data_type='test'
        )
        test_dataset = build_data_set(test_features)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        predict_label = evaluate_module(config, model_example, test_loader, test=True)

    return best_model, dev_acc, predict_label