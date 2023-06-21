from transformers import AutoTokenizer, RwkvModel,RwkvPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from typing import Optional, List

def recurrent_model_forward(model :RwkvModel, input_ids :torch.Tensor,context_length :int):
    if input_ids.shape[1] <= context_length:
        output = model(input_ids)
        return output.last_hidden_state,output.state,output.attentions
    else:
        current_index = 0
        state = None
        last_hidden_state = None
        attentions = None
        while current_index  < input_ids.shape[1]:
            length = context_length if current_index + context_length < input_ids.shape[1] else input_ids.shape[1] - current_index
            inputs = input_ids[:, current_index:length + current_index]

            if current_index == 0:
                outputs = model(inputs,use_cache=True)
                last_hidden_state = outputs.last_hidden_state
                state = outputs.state
                attentions = outputs.attentions
            else:
                outputs = model(inputs,state=state,use_cache=True)
                state = outputs.state
                last_hidden_state = outputs.last_hidden_state
                attentions = outputs.attentions
            current_index += length
        return last_hidden_state,state,attentions

class RwkvModelForSequenceClassification(RwkvPreTrainedModel):
    
    def __init__(self, config,pad_token_id=0):
        print(config)
        print(pad_token_id)
        super().__init__(config)
        print(config)
        self.num_labels = len(config.label2id.keys())
        self.pad_token_id = pad_token_id
        self.rwkv = RwkvModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

    def forward_for_inference(self, 
             input_ids: Optional[torch.LongTensor] = None
        ):
        context_length = self.rwkv.config.context_length
        last_hidden_state,state,attentions = recurrent_model_forward(self.rwkv,input_ids,context_length)
        logits = self.score(last_hidden_state[:, -1, :])
        return logits,last_hidden_state,state
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        sequence_lengths : Optional[torch.LongTensor] = None,
    ) :
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        context_length = self.rwkv.config.context_length
        rwkv_output = self.rwkv(input_ids,
                                inputs_embeds,
                                state,
                                use_cache,
                                output_attentions,
                                output_hidden_states,
                                return_dict)
        last_hidden_state = rwkv_output[0]
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        logits = self.score(last_hidden_state)
        if sequence_lengths is None:
            sequence_lengths = (torch.ne(input_ids, self.pad_token_id).sum(-1) - 1).to(logits.device)
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                # print(pooled_logits)
                # print(labels)
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + rwkv_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=rwkv_output.state,
            hidden_states=rwkv_output.last_hidden_state,
            attentions=rwkv_output.attentions,
        )