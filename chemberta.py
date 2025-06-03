import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer



class pre_chemBERTa(nn.Module):
    def __init__(self, pre_hidden=879, d_hidden=256):
        super().__init__()

        self.pre_hidden = pre_hidden

        self.tokenizer = AutoTokenizer.from_pretrained("ChemBERTa_zinc250k_v2_40k")
        self.encoder = AutoModelForMaskedLM.from_pretrained("ChemBERTa_zinc250k_v2_40k")

        self.encoder.resize_token_embeddings(self.pre_hidden)


    def forward(self, smiles):

        with torch.no_grad():
            tokens = self.tokenizer.tokenize(smiles)
            string = ''.join(tokens)
            inputs = self.tokenizer(string[0:min(len(string), self.pre_hidden)], return_tensors='pt', max_length=512)

            chem_outputs = self.encoder(**inputs)
            chem_outputs = chem_outputs.logits.squeeze(0)

        return chem_outputs

























    # def get_smiles_embeddings(dir_path, device):
    #     tokenizer = AutoTokenizer.from_pretrained("ChemBERTa-77M-MLM")
    #     model = AutoModelForMaskedLM.from_pretrained("ChemBERTa-77M-MLM")
    #
    #     # device = 'cuda:' if torch.cuda.is_available() else 'cpu'
    #     model.to(device)
    #     model = model.eval()
    #     compoundsloader, N = preparedataset(dir_path)
    #     compounds = []
    #     i = 0
    #     ln = nn.LayerNorm(768).to(device)
    #     for data in tqdm(compoundsloader):
    #         print(str(i + 1) + '/' + str(N))
    #         tokens = tokenizer.tokenize(data[0])
    #         string = ''.join(tokens)
    #         if len(string) > 512:
    #             j = 0
    #             flag = True
    #             output = torch.zeros(1, 384).to(device)
    #             while flag:
    #                 input = tokenizer(string[j:min(len(string), j + 511)], return_tensors='pt').to(device)
    #                 if len(string) <= j + 511:
    #                     flag = False
    #                 with torch.no_grad():
    #                     hidden_states = model(**input, return_dict=True, output_hidden_states=True).hidden_states
    #                     output_hidden_state = torch.cat([(hidden_states[-1] + hidden_states[1]).mean(dim=1),
    #                                                      (hidden_states[-2] + hidden_states[2]).mean(dim=1)],
    #                                                     dim=1)  # first last layers average add
    #                     output_hidden_state = ln(output_hidden_state)
    #                 output = torch.cat((output, output_hidden_state), dim=0)
    #                 j += 256
    #                 print(output.shape)
    #             output = output[1:-1].mean(dim=0).unsqueeze(dim=0).to('cpu').data.numpy()
    #         else:
    #             input = tokenizer(data[0], return_tensors='pt').to(device)
    #             with torch.no_grad():
    #                 hidden_states = model(**input, return_dict=True, output_hidden_states=True).hidden_states
    #                 output_hidden_state = torch.cat([(hidden_states[-1] + hidden_states[1]).mean(dim=1),
    #                                                  (hidden_states[-2] + hidden_states[2]).mean(dim=1)],
    #                                                 dim=1)  # first last layers average add
    #                 output_hidden_state = ln(output_hidden_state)
    #             output = output_hidden_state.to('cpu').data.numpy()
    #         compounds.append(output)
    #         i += 1
    #     compounds = np.array(compounds, dtype=object)
    #     np.save(dir_path + '/smilesembeddings', compounds, allow_pickle=True)
    #     print('The preprocess of dataset has finished!')
    #

# 原版chemberta
# class pre_chemBERTa(nn.Module):
#     def __init__(self, batch_size, d_hidden, pre_hidden, act_fn, dropout):
#         super().__init__()
#
#         # self.batch_size = batch_size
#         self.micro_batch = batch_size // 8
#         self.pre_hidden = pre_hidden
#
#         self.tokenizer = AutoTokenizer.from_pretrained("ChemBERTa-zinc-base-v1")
#         self.encoder = AutoModelForMaskedLM.from_pretrained("ChemBERTa-10M-MTR")
#
#         self.encoder.resize_token_embeddings(self.pre_hidden)  # 设置token_embedding_dim
#         # self.pub_filter_sizes = (11, 19, 27)  # pubmedbert卷积核尺寸
#
#         # self.convs_pub = nn.ModuleList(
#         #     [nn.Conv2d(1, pre_hidden, (k, pre_hidden)) for k in self.pub_filter_sizes])
#
#         # # 定义 LSTM
#         # self.lstm = nn.LSTM(input_size=pre_hidden, hidden_size=d_hidden, batch_first=True)
#         self.lstm = nn.LSTM(pre_hidden, d_hidden, 1,
#                             bidirectional=True, batch_first=True, dropout=dropout)
#         # self.chem_mlp = nn.Sequential(
#         #     nn.Linear(d_hidden, d_hidden * 2),
#         #     act_fn,
#         #     nn.Dropout(dropout),
#         #     nn.Linear(d_hidden * 2, d_hidden * 2),
#         #     act_fn,
#         #     nn.Dropout(dropout),
#         #     nn.Linear(d_hidden * 2, d_hidden),
#         # )
#         self.chem_norm = LayerNorm(d_hidden)
#
#         # 通过线性层将拼接后的维度变换为128
#         # self.linear_layer = nn.Linear(self.pre_hidden, d_hidden)
#         # 对token embedding，经过encoder，得到的是MaskedLM Object，需要经过ouput.logits得到变量
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2).unsqueeze(1)
#         return x
#
#     def forward(self, batch_smiles):
#         # 存储所有输出
#         all_outputs = []
#
#         # 对输入进行微批处理
#         for i in range(0, len(batch_smiles), self.micro_batch):
#             with torch.no_grad():
#                 # 存储微批输出
#                 micro_outputs = []
#                 # 计算当前微批的结束索引
#                 end_index = min(i + self.micro_batch, len(batch_smiles))
#                 # 提取当前微批的数据
#                 micro_batch_smile = batch_smiles[i:end_index]  # 移动到设备（CPU/GPU）
#
#                 tokenized_inputs = self.tokenizer(micro_batch_smile, return_tensors="pt", padding=True, truncation=True)
#                 input_ids = tokenized_inputs["input_ids"].to(self.encoder.device)
#                 attention_mask = tokenized_inputs["attention_mask"].to(self.encoder.device)
#
#                 inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
#
#                 outputs = self.encoder(**inputs)
#
#                 feature_vectors = outputs.logits
#                 micro_outputs.append(feature_vectors)
#
#             # lstm，取最后一个时间步
#             micro_outputs = torch.cat(micro_outputs, dim=0)
#             lstm_out, (h_n, c_n) = self.lstm(micro_outputs)     # 6,256,128
#             micro_outputs = h_n[-1].squeeze(0)
#
#             # micro_outputs = micro_outputs.mean(-2).squeeze(-2)        # 对可变维度求平均
#
#             # 多尺度二维卷积
#             # chem_features = micro_outputs.unsqueeze(1)
#             # chem_cnn_out = torch.cat([self.conv_and_pool(chem_features, conv) for conv in self.convs_pub], 1)
#             # out_p, _ = self.lstm(chem_cnn_out)
#             all_outputs.append(micro_outputs)
#
#         all_outputs = torch.cat(all_outputs, dim=0)
#         # all_outputs = self.chem_mlp(all_outputs)
#         # all_outputs = self.linear_layer(all_outputs)
#         all_outputs = self.chem_norm(all_outputs)
#         return all_outputs.unsqueeze(-2)
