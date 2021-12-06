import datetime
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForTokenClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split


class BertModel:
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

        # If there's a GPU available
        if torch.cuda.is_available():    
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))

        # If none GPU device was found
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")
        
        self.train_data = None
        self.train_sentences = []
        self.train_doc_ids = []
        self.train_labels = []
        self.train_input_ids = []
        self.train_attention_masks = []
        self.train_bert_labels = []
        self.train_black_lists = []
        self.train_dataloader = None

        self.valid_data = None
        self.valid_sentences = []
        self.valid_doc_ids = []
        self.valid_labels = []
        self.valid_input_ids = []
        self.valid_attention_masks = []
        self.valid_bert_labels = []
        self.valid_black_lists = []
        self.valid_dataloader = None

        self.test_data = None
        self.test_sentences = []
        self.test_doc_ids = []
        self.test_labels = []
        self.test_input_ids = []
        self.test_attention_masks = []
        self.test_bert_labels = []
        self.test_black_lists = []
        self.test_dataloader = None

        self.labels_map = {'B':0, 'I':1, 'L':2, 'U':3, 'O':4}
        self.reversed_labels_map = {0: 'B', 1: 'I', 2: 'L', 3: 'U', 4: 'O'}
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def init_sentences_with_labels(self, data, doc_ids):
        new_sentences = []
        new_labels = []
        for doc_id in doc_ids:
            sentence_tokens = [str(token) for token in data['Token'].loc[data['DocID'] == doc_id].to_list()]
            sentence_labels = [str(tag) for tag in data['Tag'].loc[data['DocID'] == doc_id].to_list()]
            index = 0
            new_sentence_tokens = []
            new_sentence_labels = []
            for index in range(len(sentence_tokens)):
                new_sentence_tokens.append(sentence_tokens[index])
                new_sentence_labels.append(sentence_labels[index])
                if (sentence_tokens[index]) == '.':
                    new_sentences.append(new_sentence_tokens)
                    new_sentence_tokens = []
                    new_labels.append(new_sentence_labels)
                    new_sentence_labels = []
            if len(new_sentence_tokens) != 0:
                new_sentences.append(new_sentence_tokens)
                new_labels.append(new_sentence_labels)
        return new_sentences, new_labels

    def init_input_ids_with_attention_masks(self, sentences, sententce_max_length):
        print('Generating input ids for all the tokens according to BERT ...')
        input_ids = []
        attention_masks = []
        black_lists = []

        for sentence in sentences:
            sentence_black_list = []
            for token in sentence:
                sentence_black_list.append(0)
                bert_tokens = self.tokenizer.tokenize(token)
                for i in range(1, len(bert_tokens)):
                    if bert_tokens[i][0:2] != '##':
                        sentence_black_list.append(1)
            black_lists.append(sentence_black_list)

            sentence = ' '.join(sentence)
            encoded_output = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=sententce_max_length,
                return_attention_mask=True,
                pad_to_max_length=True,
                return_tensors='pt',
                truncation=True,
            )
            input_ids.append(encoded_output['input_ids'][0])
            attention_masks.append(encoded_output['attention_mask'][0])

        print('Done!')
        return input_ids, attention_masks, black_lists

    def init_bert_labels(self, input_ids, original_labels, black_lists):
        new_labels = []
        null_label_id = -100

        for (sentence, tags, sentence_black_list) in zip(input_ids, original_labels, black_lists):
            sentence_labels = []
            tag_index = 0
            black_list_index = 0
            for token_id in sentence:
                # Get the value of the token.
                token_id = token_id.numpy().item()

                if (token_id == self.tokenizer.cls_token_id) or (token_id == self.tokenizer.pad_token_id) or (token_id == self.tokenizer.sep_token_id):
                    sentence_labels.append(null_label_id)

                elif self.tokenizer.ids_to_tokens[token_id][0:2] == '##':
                    sentence_labels.append(null_label_id)

                elif sentence_black_list[black_list_index] == 1:
                    sentence_labels.append(null_label_id)
                    black_list_index += 1

                else:
                    sentence_labels.append(self.labels_map[tags[tag_index]])
                    tag_index += 1
                    black_list_index += 1
            
            assert(len(sentence) == len(sentence_labels)) 
            
            new_labels.append(sentence_labels)
        return new_labels

    def analyze_train_sentences_lengths(self):
        print('Calculating the lengths of sentences according to BERT ...')
        sentences_lengths = []
        for sentence in self.train_sentences:
            sentence = ' '.join(sentence)
            encoded_sent = self.tokenizer.encode(
                sentence, 
                add_special_tokens=True
            )
            sentences_lengths.append(len(encoded_sent))
        print('   Min length: {:,} tokens'.format(min(sentences_lengths)))
        print('   Max length: {:,} tokens'.format(max(sentences_lengths)))
        print('Median length: {:,} tokens'.format(int(np.median(sentences_lengths))))
        print('Done!')

    def load_train_data(self, file_name, valid_file_name=None):
        self.train_data = pd.read_csv(file_name)
        if valid_file_name != None:
            self.train_data = self.train_data.append(pd.read_csv(valid_file_name))
        self.train_doc_ids = self.train_data['DocID'].drop_duplicates().to_list()
        self.train_sentences, self.train_labels = self.init_sentences_with_labels(self.train_data, self.train_doc_ids)

    def prepare_train_data(self, sententce_max_length):
        self.train_input_ids, self.train_attention_masks, self.train_black_lists = self.init_input_ids_with_attention_masks(self.train_sentences, sententce_max_length)
        self.train_bert_labels = self.init_bert_labels(self.train_input_ids, self.train_labels, self.train_black_lists)

    def get_train_tensor_dataset(self):
        return TensorDataset(
            torch.stack(self.train_input_ids, dim=0), 
            torch.stack(self.train_attention_masks, dim=0), 
            torch.tensor(self.train_bert_labels, dtype=torch.long)
        )

    def load_valid_data(self, file_name):
        self.valid_data = pd.read_csv(file_name)
        self.valid_doc_ids = self.valid_data['DocID'].drop_duplicates().to_list()
        self.valid_sentences, self.valid_labels = self.init_sentences_with_labels(self.valid_data, self.valid_doc_ids)

    def prepare_valid_data(self, sententce_max_length):
        self.valid_input_ids, self.valid_attention_masks, self.valid_black_lists = self.init_input_ids_with_attention_masks(self.valid_sentences, sententce_max_length)
        self.valid_bert_labels = self.init_bert_labels(self.valid_input_ids, self.valid_labels, self.valid_black_lists)

    def get_valid_dataloader(self, batch_size):
        valid_dataset = TensorDataset(
            torch.stack(self.valid_input_ids, dim=0), 
            torch.stack(self.valid_attention_masks, dim=0)
        )
        valid_sampler = SequentialSampler(valid_dataset)
        return DataLoader(
            valid_dataset, 
            sampler=valid_sampler, 
            batch_size=batch_size
        )
    
    def load_test_data(self, file_name):
        self.test_data = pd.read_csv(file_name)
        self.test_data = self.test_data.assign(Tag='N')
        self.test_doc_ids = self.test_data['DocID'].drop_duplicates().to_list()
        self.test_sentences, self.test_labels = self.init_sentences_with_labels(self.test_data, self.test_doc_ids)

    def prepare_test_data(self, sententce_max_length):
        self.test_input_ids, self.test_attention_masks, self.test_black_lists = self.init_input_ids_with_attention_masks(self.test_sentences, sententce_max_length)
        self.test_bert_labels = self.init_bert_labels(self.test_input_ids, self.test_labels, self.test_black_lists)

    def get_test_dataloader(self, batch_size):
        test_dataset = TensorDataset(
            torch.stack(self.test_input_ids, dim=0), 
            torch.stack(self.test_attention_masks, dim=0)
        )
        test_sampler = SequentialSampler(test_dataset)
        return DataLoader(
            test_dataset, 
            sampler=test_sampler, 
            batch_size=batch_size
        )

    def init_model(self, batch_size, dataset):
        self.train_dataloader = DataLoader(
            dataset,
            sampler = RandomSampler(dataset),
            batch_size = batch_size
        )
        # Load BertForTokenClassification
        self.model = BertForTokenClassification.from_pretrained(
            self.model_name,
            num_labels = len(self.labels_map) + 1,
            output_attentions = False,
            output_hidden_states = False,
        )
        self.model.cuda()

    def train_model(self, epochs):
        # Load the AdamW optimizer
        optimizer = AdamW(self.model.parameters(),
            lr = 5e-5, 
            eps = 1e-8 
        )
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
            num_warmup_steps = 0,
            num_training_steps = len(self.train_dataloader) * epochs
        )
        return train_model(self.model, epochs, self.train_dataloader, optimizer, scheduler, self.device)

    def get_predictions(self, prediction_dataloader):
        print('Starting computing predictions ...')
        self.model.eval()
        predictions = []

        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            
            # Unpack the inputs from our dataloader
            input_ids, input_mask = batch
            
            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                result = self.model(input_ids, 
                                token_type_ids=None, 
                                attention_mask=input_mask,
                                return_dict=True)

            logits = result.logits

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            
            # Store predictions and true labels
            predictions.append(logits)

        print('Done!')
        return predictions

    def convert_bert_predictions_to_precited_labels(self, predictions, labels):
        all_predictions = np.concatenate(predictions, axis=0)
        predicted_label_ids = np.argmax(all_predictions, axis=2)
        predicted_label_ids = np.concatenate(predicted_label_ids, axis=0)
        all_true_labels = np.concatenate(labels, axis=0)
        print("Before filtering out `null` tokens, length = {:,}".format(len(predicted_label_ids)))
        real_predictions = []
        for index in range(len(all_true_labels)):
            if not all_true_labels[index] == -100:
                real_predictions.append(predicted_label_ids[index])
        print("After filtering out `null` tokens, length = {:,}".format(len(real_predictions)))
        return [self.reversed_labels_map[pred] for pred in real_predictions]

    def post_process_predictions(self, predictions):
        final_predictions = []
        for sentence in predictions:
            sentence_final_predictions = []
            for index_token in range(len(sentence)):
                if index_token != 0:
                    if sentence_final_predictions[index_token - 1] == self.labels_map['B']:
                        sentence[index_token][self.labels_map['U']] = float('-inf')
                        sentence[index_token][self.labels_map['O']] = float('-inf')
                    if sentence_final_predictions[index_token - 1] == self.labels_map['I']:
                        sentence[index_token][self.labels_map['B']] = float('-inf')
                    if sentence_final_predictions[index_token - 1] == self.labels_map['O']:
                        sentence[index_token][self.labels_map['L']] = float('-inf')
                        sentence[index_token][self.labels_map['I']] = float('-inf')
                result = np.where(sentence[index_token] == np.amax(sentence[index_token]))
                sentence_final_predictions.append(result[0][0])
            final_predictions.append(sentence_final_predictions)
        return final_predictions

def write_to_submission_file(file_name, token_ids, predictions):
    sumbission_dataframe = pd.DataFrame({'TokenID': token_ids, 'Tag': predictions})
    sumbission_dataframe.to_csv(file_name, index=False)

# def analyze_prediction(all_predictions, index):
#     possibilities = [
#         {
#             'prev': [2, 3, 4],
#             'next': [1, 2, 3]
#         },
#         {
#             'prev': [0, 1, 3],
#             'next': [1, 2, 3]
#         },
#         {
#             'prev': [0, 1, 3],
#             'next': [0, 3, 4]
#         },
#         {
#             'prev': [2, 3, 4],
#             'next': [0, 4, 3]
#         },
#         {
#             'prev': [2, 3, 4],
#             'next': [0, 3, 4]
#         },
#         {
#             'prev': [0, 1, 2, 3, 4],
#             'next': [0, 1, 2, 3, 4]
#         }
#     ]
#     token_prediction = all_predictions[index].copy()
#     if index != 0 and index + 1 < len(all_predictions):
#         next_token_prediction = all_predictions[index + 1]
#         prev_token_prediction = all_predictions[index - 1]
#         prev_next_predictions_sum = sum(next_token_prediction) + sum(prev_token_prediction)
#         for label in range(len(token_prediction)):
#             new_prediction = 0
#             for prev_element in possibilities[label]['prev']:
#                 new_prediction += prev_token_prediction[prev_element] / prev_next_predictions_sum
#             for next_element in possibilities[label]['next']:
#                 new_prediction += next_token_prediction[next_element] / prev_next_predictions_sum
#             token_prediction[label] = new_prediction * token_prediction[label]
#         return np.where(token_prediction == np.amax(token_prediction))[0][0]

# The code below is taken from the tutorial presented in class.
def print_model_loss_values_plot(loss_values):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(loss_values, 'b-o')

    # Label the plot.
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train_model(model, epochs, train_dataloader, optimizer, scheduler, device):
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # In PyTorch, calling `model` will in turn call the model's `forward` 
            # function and pass down the arguments. The `forward` function is 
            # documented here: 
            # https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.TokenClassifierOutput
            result = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            
            loss = result.loss

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)            
        
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    print("")
    print("Training complete!")
    return loss_values