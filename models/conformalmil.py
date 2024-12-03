import os
import math
import random
import time
import numpy as np
import pickle



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from models.inceptiontime import InceptionTimeFeatureExtractor
#from models.nystrom_attention import NystromAttention
from models.lookhead import Lookahead


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from scipy.stats.mstats import mquantiles
from scipy.optimize import brentq

class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = nn.BCEWithLogitsLoss()(input, target)
        return loss

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    
    attn_bias += -math.log(S)
    
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.sigmoid(attn_weight)#torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight


class CrossAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = decoder_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(decoder_dim, decoder_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """
        query from decoder (x), key and value from encoder (y)
        """
        B, N, C = x.shape
        Ny = y.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, Ny, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        x, attn = scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop,
        )
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class MILPooling(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dropout=0.2, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = CrossAttention(dim, dim, num_heads=16, attn_drop=dropout)
        
    def forward(self, class_tokens, input_tokens):
        class_tokens, attn = self.attn(class_tokens, self.norm(input_tokens))
        return class_tokens, attn

### Define Wavelet Kernel
def mexican_hat_wavelet(size, scale, shift, device='cuda'): #size :d*kernelsize  scale:d*1 shift:d*1
    """
    Generate a Mexican Hat wavelet kernel.

    Parameters:
    size (int): Size of the kernel.
    scale (float): Scale of the wavelet.
    shift (float): Shift of the wavelet.

    Returns:
    torch.Tensor: Mexican Hat wavelet kernel.
    """
  
    x = torch.linspace(-( size[1]-1)//2, ( size[1]-1)//2, size[1]).to(device)
    x = x.reshape(1,-1).repeat(size[0],1)
    x = x - shift  # Apply the shift

    # Mexican Hat wavelet formula
    C = 2 / ( 3**0.5 * torch.pi**0.25)
    wavelet = C * (1 - (x/scale)**2) * torch.exp(-(x/scale)**2 / 2)*1  /(torch.abs(scale)**0.5)

    return wavelet #d*L

class WaveletEncoding(nn.Module):
    def __init__(self, dim=512, max_len = 256, hidden_len = 512,dropout=0.0):
        super().__init__()
        #n_w =3
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_3 = nn.Linear(dim, dim)
        
    def forward(self, x, wave1, wave2, wave3):
        x = x.transpose(1, 2)
        
        D = x.shape[1]
        scale1, shift1 =wave1[0,:],wave1[1,:]
        wavelet_kernel1 = mexican_hat_wavelet(size=(D,19), scale=scale1, shift=shift1, device=x.device)
        scale2, shift2 =wave2[0,:],wave2[1,:]
        wavelet_kernel2 = mexican_hat_wavelet(size=(D,19), scale=scale2, shift=shift2, device=x.device)
        scale3, shift3 =wave3[0,:],wave3[1,:]
        wavelet_kernel3 = mexican_hat_wavelet(size=(D,19), scale=scale3, shift=shift3, device=x.device)
        
         #Eq. 11
        pos1= torch.nn.functional.conv1d(x,wavelet_kernel1.unsqueeze(1),groups=D,padding ='same')
        pos2= torch.nn.functional.conv1d(x,wavelet_kernel2.unsqueeze(1),groups=D,padding ='same')
        pos3= torch.nn.functional.conv1d(x,wavelet_kernel3.unsqueeze(1),groups=D,padding ='same')
        x = x.transpose(1, 2)   #B*N*D
        # print(x.shape)
        #Eq. 10
        x = x + self.proj_1(pos1.transpose(1, 2)+pos2.transpose(1, 2)+pos3.transpose(1, 2))# + mixup_encording
        
        # mixup token information
        return x


class MILModel(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400, dropout=0.):
        super().__init__()
        # define backbone Can be replace here
        self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=in_features)
            
        # define WPE    
        self.wave1 = torch.randn(2, mDim,1)
        self.wave1[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 )  #make sure scale >0
        self.wave1 = nn.Parameter(self.wave1)
        
        self.wave2 = torch.zeros(2, mDim,1)
        self.wave2[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave2 = nn.Parameter(self.wave2)
        
        self.wave3 = torch.zeros(2, mDim,1 )
        self.wave3[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave3 = nn.Parameter(self.wave3)    
        
        hidden_len = 2 * max_seq_len
        
        self.pos_layer =  WaveletEncoding(mDim, max_seq_len, hidden_len) 
        
        # define class token      
        self.cls_token = nn.Parameter(torch.randn(1, n_classes, mDim))
        
        # define mil classifier
        self.mil_pooling = MILPooling(dim=mDim, dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        
        self.clf = nn.Sequential(
            nn.Linear(mDim, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, 1)
        ) 
        
        self.initialize_weights()
    
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x1 = self.feature_extractor(x.transpose(1, 2))
        x1 = x1.transpose(1, 2)
        x = x1

        B, seq_len, D = x.shape
        
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)    #B * 1 * d
        
        x = self.pos_layer(x,self.wave1,self.wave2,self.wave3)
        
        cls_tokens, attn = self.mil_pooling(cls_tokens, x) 
 
        logits = self.clf(self.norm(cls_tokens)).squeeze(dim=2)
            
        return logits, attn.detach()

class ConformalMIL(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.configs = configs
        
        self.device = configs.device
        
        assert configs.n_classes > 1, "number of classes should be more than one"
        
        milnet = MILModel(configs.in_features, configs.n_classes, configs.embed, configs.seq_len, configs.dropout)
        
        if configs.use_multi_gpu:
            self.milnet = torch.nn.DataParallel(milnet.to(self.configs.device))
        else:
            self.milnet = milnet.to(self.configs.device)
        
    def load_dataset(self, flag='train'):
        if flag == "train":
            data = torch.load(f"./datasets/{self.configs.dataset}/train.pt", weights_only=False)
        if flag == "val":
            data = torch.load(f"./datasets/{self.configs.dataset}/val.pt", weights_only=False)
        if flag == "test":
            data = torch.load(f"./datasets/{self.configs.dataset}/test.pt", weights_only=False)
    
        dataset = TensorDataset(data['samples'].float(), data['labels'].long())
        
        return dataset
    
    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  
    
    def train(self):
        ckpt = f"./checkpoints/{self.configs.dataset}/{self.configs.seed}"
        os.makedirs(ckpt, exist_ok=True)
        
        train_dataset = self.load_dataset("train")
        
        if self.configs.cal_fraction > 0: #if cal_fraction is 0, calibrating model on validation set
            train_dataset, cal_dataset = torch.utils.data.random_split(train_dataset, [1-self.configs.cal_fraction, self.configs.cal_fraction])

            train_loader = DataLoader(train_dataset, batch_size=min(self.configs.batch_size, len(train_dataset)), 
                                      shuffle=True, drop_last=True, pin_memory=True, num_workers=self.configs.num_workers)
        else:
            train_loader = DataLoader(train_dataset, batch_size=min(self.configs.batch_size, len(train_dataset)), 
                                      shuffle=True, drop_last=False, pin_memory=True, num_workers=self.configs.num_workers)
            
        optimizer = torch.optim.AdamW(self.milnet.parameters(), lr=self.configs.lr, weight_decay=self.configs.weight_decay)
        
        criterion = LabelSmoothingBCEWithLogitsLoss()#nn.BCEWithLogitsLoss()
        
        time_now = time.time()
        train_steps = len(train_loader)
        
        best_val_f1 = 0
        
        for epoch in range(self.configs.epochs):
            self.milnet.train()
            iter_count = 0
            train_loss = []
            
            epoch_time = time.time()
            
            for batch_id, (feats, bag_label) in enumerate(train_loader):
                bag_feats = feats.to(self.device)
                bag_label = bag_label.to(self.device)
                
                bag_label = F.one_hot(bag_label, num_classes=self.configs.n_classes).float()
                
                # window-based random masking
                if self.configs.dropout_patch > 0:
                    selecy_window_indx = random.sample(range(10), int(self.configs.dropout_patch*10))
                    inteval = int(len(bag_feats)//10)
                    
                    for idx in selecy_window_indx:
                        bag_feats[:,idx*inteval:idx*inteval+inteval,:] = torch.randn(1).to(self.device)
                        
                optimizer.zero_grad()
                bag_prediction, attn  = self.milnet(bag_feats)
                
                bag_loss = criterion(bag_prediction, bag_label)

                train_loss.append(bag_loss.item())
                
                bag_loss.backward()
                nn.utils.clip_grad_norm_(self.milnet.parameters(), max_norm=1.0)
                optimizer.step()
                
                iter_count+=1
                
                if (batch_id + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(batch_id + 1, epoch + 1, bag_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.configs.epochs - epoch) * train_steps - batch_id)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            vali_loss, val_metrics_dict = self.validation("val")
            test_loss, test_metrics_dict = self.validation("test")
            
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {test_metrics_dict['Precision']:.5f}, "
                f"Recall: {test_metrics_dict['Recall']:.5f} "
                f"F1: {test_metrics_dict['F1']:.5f}, "
                f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
            )
            
            if best_val_f1 < test_metrics_dict['F1']:
                print(f"Best validation Accuracy score update: {best_val_f1} ---> {val_metrics_dict['F1']}")
                best_val_f1 = val_metrics_dict['F1']
                
                if self.configs.use_multi_gpu:
                    torch.save(self.milnet.module.state_dict(), os.path.join(ckpt,"conformal_mil.pth"))
                else:
                    torch.save(self.milnet.state_dict(), os.path.join(ckpt,"conformal_mil.pth"))
    
    def post_train_evaluation(self):
        
        train_dataset = self.load_dataset("train")
        
        if self.configs.cal_fraction > 0.0: #if cal_fraction is 0, calibrating model on validation set
            _, cal_dataset = torch.utils.data.random_split(train_dataset, [1-self.configs.cal_fraction, self.configs.cal_fraction])
            cal_loader = DataLoader(cal_dataset, batch_size=min(self.configs.batch_size, len(cal_dataset)), 
                                      shuffle=False, drop_last=False, pin_memory=True, num_workers=self.configs.num_workers)
        else:
            cal_dataset = self.load_dataset('val')
            cal_loader = DataLoader(cal_dataset, batch_size=min(self.configs.batch_size, len(cal_dataset)), 
                                      shuffle=False, drop_last=False, pin_memory=True, num_workers=self.configs.num_workers)
        
        ckpt = f"./checkpoints/{self.configs.dataset}/{self.configs.seed}"
        print('loading best val F1 model')
        self.milnet.load_state_dict(torch.load(os.path.join(ckpt,"conformal_mil.pth"), weights_only=False))    
        
        test_loss, test_metrics_dict = self.validation("test")
        print(
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f} "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        
        cal_logits, cal_labels = self.get_logits_targets(cal_loader)
        
        self.T = self.set_temperature(cal_logits, cal_labels)
        
        cal_sgmd = torch.sigmoid(cal_logits/self.T.detach().cpu()).numpy()
        
        cal_logits, cal_labels = cal_logits.numpy(), cal_labels.numpy()
        
        for alpha in [0.1, 0.05, 0.025, 0.01]:
            pth_results = f"./results/{self.configs.dataset}/{self.configs.seed}/{alpha}"
            os.makedirs(pth_results, exist_ok=True)
            
            lamhat = self.calibration(cal_sgmd, cal_labels, alpha)
            print(lamhat)
            scores_list, preds, trues, inputs, attns = self.evaluate_predictions(lamhat)
            
            save_dict = {
                'Lambdahat': lamhat,
                'Scores': scores_list,
                'Predictions': preds,
                'Targets': trues,
                'Inputs': inputs,
                'Attentions': attns
                }
            
            #np.save(pth_results+"/results.npy", save_dict, allow_pickle=True)
            with open(pth_results + "/results.npy", 'wb') as f:
                pickle.dump(save_dict, f, protocol=4)
            
    def validation(self, flag="val"):
        val_dataset = self.load_dataset(flag)
        val_loader = DataLoader(val_dataset, batch_size=min(self.configs.batch_size, len(val_dataset)), 
                                  shuffle=False, drop_last=False, pin_memory=True, num_workers=self.configs.num_workers)
        
        criterion = nn.BCEWithLogitsLoss()
        
        total_loss = []
        preds = []
        trues = []
        
        self.milnet.eval()
        
        with torch.no_grad():
            for batch_id, (feats, bag_label) in enumerate(val_loader):
                bag_feats = feats.to(self.device)
                bag_label = bag_label.to(self.device)
                truth = bag_label.detach().cpu()
                
                bag_label = F.one_hot(bag_label, num_classes=self.configs.n_classes).float()
                    
                bag_prediction, attn  = self.milnet(bag_feats)
                
                pred = torch.sigmoid(bag_prediction).detach().cpu()
                loss = criterion(pred, bag_label.detach().cpu())
                total_loss.append(loss.item())

                preds.append(pred)
                trues.append(truth)
                
            total_loss = np.average(total_loss)
            preds = torch.cat(preds, 0)
            trues = torch.cat(trues, 0)
            
            probs = preds#F.softmax(preds, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
            
            trues_onehot = F.one_hot(trues.reshape(-1,), num_classes=self.configs.n_classes).float().cpu().numpy()
            predictions = (torch.argmax(probs, dim=1).cpu().numpy())
            
            probs = probs.cpu().numpy()
            trues = trues.flatten().cpu().numpy()
            
            metrics_dict = {
                "Accuracy": accuracy_score(trues, predictions),
                "Precision": precision_score(trues, predictions, average="macro", zero_division=1),
                "Recall": recall_score(trues, predictions, average="macro", zero_division=1),
                "F1": f1_score(trues, predictions, average="macro"),
                "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
                "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
            }
            
        return total_loss, metrics_dict
    
    def get_logits_targets(self, loader):
        
        print(f"Calibrate on {len(loader.dataset)} data points")
        logits = []
        labels = []
        self.milnet.eval()
        with torch.no_grad():
            for feats, targets in loader:
                feats = feats.to(self.device)
                targets = targets.to(self.device)
                
                batch_logits = self.milnet(feats)[0].detach().cpu()
                logits.append(batch_logits)
                labels.append(targets.detach().cpu())
        
        logits = torch.cat(logits, 0)
        labels = torch.cat(labels, 0)

        return logits, labels
    
    def set_temperature(self, cal_logits, cal_labels):
        criterion = nn.BCEWithLogitsLoss()
        
        T = torch.tensor([1.3]*self.configs.n_classes, device=self.device, requires_grad=True)
        
        optimizer = torch.optim.SGD([T], lr=0.1)
        
        logits = cal_logits.detach().to(self.device)
        targets = F.one_hot(cal_labels.detach().reshape(-1,), num_classes=self.configs.n_classes).float().to(self.device)
        
        for iter in range(100):
            optimizer.zero_grad()
            logits.requires_grad = True
            out = logits/T

            loss = criterion(out, targets)
            
            loss.backward()
            optimizer.step()
        
        print("Temperature:", T)
        
        return T
    
    def calibration(self, cal_sgmd, cal_labels, alpha):
        print(f"Calibrating MIL Model with tolerance {alpha:.4f}")
        n_classes = self.configs.n_classes
        
        # Initialize list to hold thresholds for each class
        class_thresholds = []
        
        def false_negative_rate(class_preds, threshold):
            # Predict as positive where sigmoid >= threshold
            predicted_positive = class_preds >= threshold
            # FNR is 1 - (TP / (TP + FN))
            fnr = 1 - np.mean(predicted_positive)
            return fnr
        
        def lamhat_threshold(lam, class_preds, n):
            return false_negative_rate(class_preds, lam) - ((n + 1) / n * alpha - 1 / (n + 1))
        
        # Calculate threshold for each class
        for class_idx in range(n_classes):
            cal_sgmd_cls = cal_sgmd[cal_labels == class_idx][:, class_idx]
            n = len(cal_sgmd_cls)  # Length should be for the specific class
            lamhat = brentq(lamhat_threshold, 0, 1, args=(cal_sgmd_cls, n))
            class_thresholds.append(lamhat)
        
        return np.array(class_thresholds)

    def conformal_inference(self, X, lamhat): 
        X = X.to(self.device)
        logits, attn = self.milnet(X)
    
        scores = torch.sigmoid(logits.detach()/self.T.detach()).cpu().float().numpy()
        binary_predictions = scores >= lamhat
        
        # Ensure each prediction set contains at least one element
        prediction_set = []
        for i in range(binary_predictions.shape[0]):
            if binary_predictions[i].sum() == 0:
                # If no element is above the threshold, add the most confident prediction
                top_prediction = np.argmax(scores[i])
                prediction_set.append([top_prediction])
            else:
                # Otherwise, add the elements that meet the threshold
                prediction_set.append(np.where(binary_predictions[i] == 1)[0].tolist())
    
        return scores, prediction_set, attn
        
    def evaluate_predictions(self, lamhat):
        test_dataset = self.load_dataset('test')
        
        test_loader = DataLoader(test_dataset, batch_size=min(self.configs.batch_size, len(test_dataset)), 
                                 shuffle=False, drop_last=False, pin_memory=True, num_workers=self.configs.num_workers)
        
        scores_list = []
        preds = []
        trues = []
        inputs = []
        attns = []
        self.milnet.eval()
        with torch.no_grad():
            for batch_id, (feats, bag_label) in enumerate(test_loader):
                scores, prediction_set, attn  = self.conformal_inference(feats, lamhat)
                bag_label = F.one_hot(bag_label, num_classes=self.configs.n_classes).float()
                
                preds+=prediction_set
                
                scores_list.append(scores)
                attns.append(attn.detach().cpu())
                inputs.append(feats.detach().cpu())
                trues.append(bag_label.detach().cpu())
        
        scores_list = np.concatenate(scores_list, 0)
        inputs = torch.cat(inputs, 0).numpy()
        attns = torch.cat(attns, 0).numpy()
        trues = torch.cat(trues, 0).numpy()
        
        sample_coverage = np.array([np.argmax(trues[i]) in preds[i] for i in range(len(trues))])
        marg_coverage = np.mean(sample_coverage)
        length = np.mean([len(preds[i]) for i in range(len(trues))])
        
        print('Marginal coverage:{:2.3%}'.format(marg_coverage))
        print('Average size:            {:2.3f}'.format(length))
        
        gt_labels = np.argmax(trues, axis=1)  # Convert one-hot encoded trues to class labels
        for class_label in range(self.configs.n_classes):
            # For the current class, find where the ground truth equals the class label
            class_indices = gt_labels == class_label
            # True Positives are where the ground truth equals the class label and the sample is correctly classified
            true_positives = np.sum(sample_coverage[class_indices])
            # False Negatives are where the ground truth equals the class label but the sample is misclassified
            false_negatives = np.sum(~sample_coverage[class_indices])
            # FNR = False Negatives / (True Positives + False Negatives)
            fnr = false_negatives / (false_negatives + true_positives)
            
            print(f'Class {class_label}: FNR = {fnr:.3%}')
        
        return scores_list, preds, trues, inputs, attns
