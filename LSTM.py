from lightning.pytorch.loops.optimization.automatic import OutputResult
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L
from torch.utils.data import DataLoader, TensorDataset

class LSTM(L.LightningModule):
    #Init method
    def __init__(self) -> None:
        super().__init__()
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)
        
        #Bias and Weights in Forget gate
        self.wfg1 = nn.parameter.Parameter(torch.normal(mean=mean,std=std),requires_grad=True)
        self.wfg2 = nn.parameter.Parameter(torch.normal(mean=mean,std=std),requires_grad=True)
        self.bfg = nn.parameter.Parameter(torch.tensor(0.0),requires_grad=True)
        
        #Bias and Weights in Input gate
        self.wig11 = nn.parameter.Parameter(torch.normal(mean=mean,std=std),requires_grad=True)
        self.wig12 = nn.parameter.Parameter(torch.normal(mean=mean,std=std),requires_grad=True)
        self.big1 = nn.parameter.Parameter(torch.tensor(0.0),requires_grad=True)

        self.wig21 = nn.parameter.Parameter(torch.normal(mean=mean,std=std),requires_grad=True)
        self.wig22 = nn.parameter.Parameter(torch.normal(mean=mean,std=std),requires_grad=True)
        self.big2 = nn.parameter.Parameter(torch.tensor(0.0),requires_grad=True)
        
        #Bias and Weights in Output gate
        self.wog1 = nn.parameter.Parameter(torch.normal(mean=mean,std=std),requires_grad=True)
        self.wog2 = nn.parameter.Parameter(torch.normal(mean=mean,std=std),requires_grad=True)
        self.bog = nn.parameter.Parameter(torch.tensor(0.0),requires_grad=True)

     
    #Conduce lstm math in gates
    def lstm_unit(self,input,long_term,short_term):
        percent_forget_long_term = torch.sigmoid(input*self.wfg2 + short_term*self.wfg1 + self.bfg)
        percent_potential_long_term = torch.sigmoid(short_term*self.wig11+input*self.wig12+self.big1)
        potential_long_term = torch.tanh(short_term*self.wig21 + input*self.wig22 + self.big2)
        
        #new long_term 
        new_long_term = long_term*percent_forget_long_term + (percent_potential_long_term*potential_long_term)

        percent_potential_short_term = torch.sigmoid(short_term*self.wog1+input*self.wog2+self.bog)
        new_short_term = torch.tanh(new_long_term)*percent_potential_short_term
        
        return ([new_long_term,new_short_term])
    def forward(self,input):
        long_memory = 0.0
        shor_memory = 0.0 
        for value in input:
            long_memory,shor_memory = self.lstm_unit(value,long_memory,shor_memory)
        return shor_memory
    def configure_optimizers(self):
        return Adam(self.parameters())

    #Calculating Loss
    def training_step(self, batch,batch_idx):
        input_i,label_i = batch
        output_i = self.forward(input_i[0])
        loss = (label_i - output_i)**2
            
        self.log("train_loss",loss)

        if (label_i == 0):
            self.log("out_0",output_i)
        else:
            self.log("out_1",output_i)

        return loss


model = LSTM()


#Training Data
inputs = torch.tensor([[0.,0.5,0.25,1.0],[1.,0.5,0.25,1.]])
labels = torch.tensor([0.,1.])

dataset = TensorDataset(inputs,labels)
dataloader = DataLoader(dataset)

#Trainer in Lightning
trainer = L.Trainer(max_epochs=20000)
trainer.fit(model=model,train_dataloaders=dataloader)



#Result
print("A observed=0, Predict=",model(torch.tensor([0.,0.5,0.25,1.])).detach())
print("B observed=1, Predict=",model(torch.tensor([1.,0.25,0.25,1.])).detach())


