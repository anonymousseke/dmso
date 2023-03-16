from fastNLP import Tester, Vocabulary, EarlyStopCallback, logger
from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric

from fastNLP.io import CSVLoader
from fastNLP.embeddings import StaticEmbedding

from fastNLP.core.callback import WarmupCallback, GradientClipCallback
from fastNLP.core.metrics import ClassifyFPreRecMetric, ConfusionMatrixMetric, AccuracyMetric

import torch
import torch.nn as nn
from models.dmso import DMSO

class my_model(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = embed
        self.dmso = DMSO(embed=embed, hidden_size=200, num_classes=4, dropout=0.2)

    def forward(self, words1, words2):
        outputs = self.dmso(words1, words2)
        return {'pred':outputs} 

def func(text):
    text = text.replace('[','').replace(']','').replace("'",'')
    text = text.split(",")
    length = 0
    for i in range(len(text)):
        if i == 0:
            tmp = [text[i].split()[:60]]
            continue
        tmp.append(text[i].split()[:60])
        length += len(text[i])
        if length >= 250:
            break
    return tmp

for i in range(5): 
    print('loading data...')
    loader = CSVLoader(headers=('ku1','ku2','label'))
    data_bundle = loader.load("./data")
    
    print('preprocessing...')
    MAP = {'0':'duplicate', '1':'direct', '2':'indirect', '3':'isolated'}
    data_bundle.apply(lambda x: func(x['ku1']), new_field_name='words1', is_input=True)
    data_bundle.apply(lambda x: func(x['ku2']), new_field_name='words2', is_input=True)
    data_bundle.apply(lambda x: MAP[x['label']], new_field_name='target', is_target=True)

    train_data = data_bundle.get_dataset('train')
    dev_data = data_bundle.get_dataset('dev')
    test_data = data_bundle.get_dataset('test')
    print(train_data[:10])

    print('vocabulary...')
    vocab = Vocabulary()
    vocab.from_dataset(train_data, field_name=['words1', 'words2'], no_create_entry_dataset=[dev_data,test_data])
    vocab.index_dataset(train_data, dev_data, test_data, field_name=['words1', 'words2'], new_field_name=['words1', 'words2'])

    target_vocab = Vocabulary(padding=None, unknown=None)
    target_vocab.add_word_lst(['duplicate','direct','indirect','isolated'])
    target_vocab.index_dataset(train_data, dev_data, test_data, field_name='target')

    data_bundle.set_vocab(field_name='words', vocab=vocab)
    data_bundle.set_vocab(field_name='target', vocab=target_vocab)

    data_bundle.set_input('words1', 'words2')
    data_bundle.set_target('target')
    print(dev_data[:10])


    logger.add_file('./log/log.txt', level='INFO')
    print('embedding...')
    embed = StaticEmbedding(data_bundle.get_vocab('words'), model_dir_or_name="./embedding.txt", requires_grad=False, dropout=0.1, word_dropout=0.1)
    print('model...')
    model = my_model(embed=embed)

    metric = [AccuracyMetric(), ClassifyFPreRecMetric(f_type='macro', only_gross=True)]
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = 0 if torch.cuda.is_available() else 'cpu'
    callbacks = [WarmupCallback(warmup=0, schedule='linear'), EarlyStopCallback(5),
                GradientClipCallback(clip_value=5, clip_type='value')]
            
    trainer = Trainer(train_data=train_data, model=model, optimizer=optimizer,
                    loss=CrossEntropyLoss(), device=device, batch_size=32, dev_data=dev_data,
                    metrics=metric, n_epochs=30, update_every=1, callbacks=callbacks)
    trainer.train()

    tester = Tester(data=test_data, model=model, device=device, batch_size=32, 
                        metrics=[ConfusionMatrixMetric(), ClassifyFPreRecMetric(f_type='macro', only_gross=False)])
    tester.test()

    torch.save(model.dmso.state_dict(), "./save/dmso.pkl")


