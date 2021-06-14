import time
import torch
import random
import datetime
import numpy as np
import pandas as pd

from typing import List
from logging import INFO
from logger_factory import logger_factory
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup

logger = logger_factory('finetuner', INFO)


class FineTuner():
    def __init__(
            self,
            sentences: List[str], 
            labels: List[int],
            model='bert-base-uncased',
            num_labels=2,
            epochs=4, 
            seed_val=42
        ) -> None:

        """ Class for fine-tuning BERT for classification tasks. 
        This class is based on the Transformers library as well as the run_glue.py script 
        which describes fine-tuning BERT for classification tasks and is kindly provided 
        by HuggingFace here:
        https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py

        Args:
            sentences: list if N sentences

            labels: list of N labels

            model: string which determines the huggingface transformer model that 
            should be finetuned. The default is the 12-layer standard BERT model.

            num_labels: int which defines how many different output labels exist in 
            the provided dataset. The default is 2 for binary classification.

            epochs: int which specifies amount of training epochs. 
            BERT authors recommend 2-4 epochs for fine-tuning

            seed_val: int which determines the initial seed
        """

        if torch.cuda.is_available():    
            # utilize GPU
            self.device = torch.device("cuda")
            logger.info(f'{torch.cuda.device_count()} GPU(s) available')
        else:
            logger.info(f'no GPU available. carrying out finetuning on CPU.')
            self.device = torch.device("cpu")

        self.sentences = sentences
        self.labels = labels
        self.epochs = epochs

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Load BertForSequenceClassification, the pretrained BERT model 
        # with a single linear classification layer on top. 
        self.model = BertForSequenceClassification.from_pretrained(
            model,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )

        # run model on the GPU.
        self.model.cuda()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr = 5e-5,
            eps = 1e-8
        )


        # setting seed values
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

    @staticmethod
    def pad_tokens(input_ids, max_len=64):
        """ Pads and truncates sequences to obtain same length

        Args:
            input_ids: vocabulary ids for each token in each sentence as list 
            of list of ints

            max_len: maximum sentence length, should correspond to the longest 
            expected sentence in the dataset but less than 512
        
        Returns:
            input_ids: vocabulary ids with same length where each element has length max_len
        """
    
        logger.info('padding and truncating all sentences to {max_len}')

        # pads input tokens with 0
        # "post" indicates that we want to pad and truncate at the end of the sequence,
        # as opposed to the beginning.
        return pad_sequences(
            input_ids, 
            value=0, 
            maxlen=max_len, 
            dtype="long",
            padding="post",
            truncating="post"
        )

    @staticmethod
    def get_attention_masks(input_ids):
        """ Create attention masks
        
        Args:
            input_ids: vocabulary ids for each token in each sentence as list of list of ints

        Returns:
            attention_masks: mask which defines which tokens are actual words as list of list of ints 
        """


        attention_masks = []

        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)
        
        return attention_masks

    @staticmethod
    def flat_accuracy(preds, labels):
        ''' calculates accuracy for training loop '''
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
    @staticmethod
    def format_time(elapsed):
        ''' converts a time in seconds and returns a string hh:mm:ss '''
    
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def tokenize(self, sentences: List[str]):
        """ Tokenizes all sentences and maps the tokens to thier word IDs
        
        Args:
            sentences: list of strings

        Returns:
            input_ids: vocabulary ids for each token in each sentence as list of list of ints
        """

        input_ids = []

        for sent in sentences:
            # encode = tokenize + prepend_CLS + append_SEP + map_tokens_to_vocab_ids
            encoded_sent = self.tokenizer.encode(sent, add_special_tokens = True)
            input_ids.append(encoded_sent)
        
        return input_ids

    def get_data_loader(self):
        """ Provides training and validation dataloader for training loop.
        These torch dataloaders care for the batching and yield subbatches 
        of the dataset to meet memory requirements.

        Returns:
            train_dataloader
            validation_dataloader
        """
        tokenized_sentences = self.tokenize(self.sentences)
        
        input_ids = self.pad_tokens(tokenized_sentences)

        att_masks = self.get_attention_masks(input_ids)

        # 9:1 train-test split
        train_inputs, validation_inputs, train_labels, validation_labels = \
            train_test_split(input_ids, self.labels, random_state=42, test_size=0.1)
        train_masks, validation_masks, _, _ = \
            train_test_split(att_masks, self.labels, random_state=42, test_size=0.1)

        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)
      
        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        # BERT authors recommend 16 or 32 as batch size for fine-tuning 
        batch_size = 32

        # training data loader
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # validation data loader
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        return train_dataloader, validation_dataloader

    def train(self):
        loss_values = []
        train_dataloader, validation_dataloader = self.get_data_loder(sentences, labels)

        # training_steps = batches * epochs
        num_training_steps = len(train_dataloader) * self.epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps = 0, # default value in run_glue.py
            num_training_steps = num_training_steps
        )

        for epoch_i in range(0, self.epochs):
            
            # ========================================
            #               Training
            # ========================================

            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training ...')

    
            # save start time and reset total loss for epoch
            t0 = time.time()
            total_loss = 0

            # switch model to train mode
            self.model.train()

            for step, batch in enumerate(train_dataloader):

                # progress update
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # copy tensor to GPU
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # clear previously calculated gradients before backward pass
                self.model.zero_grad()        

                # perform forward pass = evaluate the model on training batch
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = self.model(
                    b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels
                )
                
                loss = outputs[0]

                # aggregate training loss
                total_loss += loss.item()

                # backward pass to calculate gradients
                loss.backward()

                # gradient clipping to avoid exploding gradients / instability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # update parameters 
                self.optimizer.step()

                # update learning rate
                scheduler.step()

            # calculate average loss & store for plotting
            avg_train_loss = total_loss / len(train_dataloader)            
            loss_values.append(avg_train_loss)

            print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(self.format_time(time.time() - t0)))
                
            # ========================================
            #               Validation
            # ========================================

            print("\nRunning Validation ...")

            t0 = time.time()

            # switch to eval mode (turns off dropout layers)
            self.model.eval()

            # track variables 
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for batch in validation_dataloader:
                
                # adding batch to GPU
                batch = tuple(t.to(self.device) for t in batch)
                
                # unpack inputs from dataloader
                b_input_ids, b_input_mask, b_labels = batch
                
                # don't compute or store gradients for validation
                with torch.no_grad():        

                    # forward pass: calculates logit predictions
                    outputs = self.model(
                        b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask
                    )
                
                # logits = output values prior to applying an activation function
                logits = outputs[0]

                # move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                
                # calculate accuracy for batch of test sentences
                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
                
                # aggregate total accuracy
                eval_accuracy += tmp_eval_accuracy

                # track number of batches
                nb_eval_steps += 1

            # report final accuracy for validation run
            logger.info("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
            logger.info("  Validation took: {:}".format(self.format_time(time.time() - t0)))

            logger.info("Training complete!")


if __name__== '__main__':
    df = pd.read_csv(
        "../data/cola_public/raw/in_domain_train.tsv", 
        delimiter='\t', 
        header=None, 
        names=['sentence_source', 'label', 'label_notes', 'sentence']
    )

    sentences = df.sentence.values
    labels = df.label.values

    f = FineTuner(sentences, labels)
    f.train()