
import os
import torch
import time
from torch.utils.data import DataLoader

def sanitize_class_name(class_name):
    illegal_chars = '\\/*?:"<>|'
    sanitized = class_name.replace(' ', '_')
    for char in illegal_chars:
        sanitized = sanitized.replace(char, '')
    return sanitized

def NLL(y_pred, Y):
    muX = y_pred[..., 0]
    muY = y_pred[..., 1]
    sigX = y_pred[..., 2]
    sigY = y_pred[..., 3]
    rho = y_pred[..., 4]
    
    x = Y[..., 0]
    y = Y[..., 1]
    
    # 保证sigma是正数，避免数值问题
    eps = 1e-6
    sigX = sigX.clamp(min=eps)
    sigY = sigY.clamp(min=eps)
    rho = rho.clamp(min=-1+eps, max=1-eps)  # 保证rho在(-1,1)范围内
    
    one_minus_rho2 = 1 - rho**2
    
    # 计算各项
    norm_x = (x - muX) / sigX
    norm_y = (y - muY) / sigY
    
    # NLL公式的二次型部分
    z = (norm_x**2) + (norm_y**2) - 2 * rho * norm_x * norm_y
    
    # 负对数似然损失（不含常数项）
    nll_element = 0.5 * torch.log(2 * torch.pi * sigX * sigY * torch.sqrt(one_minus_rho2)) + \
                  (z / (2 * one_minus_rho2))

    return nll_element.mean()

class EasyInstructor:
    def __init__(
        self,
        net,
        train_dataset,
        device,
        epoch=1,
        lr=1e-4,
        batch_size=128,
        test_dataset=None,
        val_dataset=None,
        log_title='',
        test_per_epoch=-1,
        lossf='ce',
        num_works=1,
        opt='adam',
        noise=False,
        early_stop=-1,
        loader_sampler=None,
        clip_grad=False,
        momentum=0,
        weight_decay=0,
        minibatch=False
    ):
        self.net = net
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.log_title = log_title
        self.test_per_epoch = test_per_epoch
        self.print_train_acc = True
        self.num_works = num_works
        self.selectopt(opt)
        self.noise = noise
        self.loader_sampler = loader_sampler
        self.gradhandle = None
        self.updatedhandle = None
        self.ontrainhandle = None
        self.clip_grad = clip_grad
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.minibatch = minibatch
        self.test_record = []
        self.best_val_loss = float('inf')
        os.makedirs('./trained_models', exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        # self.best_model_path = f'./trained_models/{timestamp}/best_model.pth'
        # # 新增：日志文件路径
        # self.log_path = f'./trained_models/{timestamp}/train_log.txt'

    def selectopt(self, opt):
        if isinstance(opt, str):
            if opt.lower() == 'sgd':
                self.optc = torch.optim.SGD
            elif opt.lower() == 'adamw':
                self.optc = torch.optim.AdamW
            elif opt.lower() == 'adam':
                self.optc = torch.optim.Adam
        else:
            self.optc = opt

    @staticmethod
    @torch.no_grad()
    def _test_model_(net:torch.nn.Module, dataset, batch_size=64, device='cuda:0', num_workers=4, miss_threshold=2.0):
        if dataset is None:
            return None
        mse_loss_fn = torch.nn.MSELoss()
        EasyInstructor.to(net, device)
        loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)
        total_mse_loss, total_ADE, total_FDE, miss_count, total_samples = 0.0, 0.0, 0.0, 0, 0
        net.eval()
        for batch in loader:
            for key in batch:
                batch[key] = batch[key].to(device)
            output, lon, lat = net(batch)
            target = batch['Y']
            output = output[:, :, :2]
            loss = mse_loss_fn(output, target)
            total_mse_loss += loss.item()
            batch_size_curr = output.shape[0]
            total_samples += batch_size_curr
            l2_distances = torch.norm(output - target, dim=2)
            total_ADE += l2_distances.mean(dim=1).sum().item()
            total_FDE += l2_distances[:, -1].sum().item()
            miss_count += (l2_distances[:, -1] > miss_threshold).sum().item()

        net = net.cpu()
        mean_mse_loss = total_mse_loss / len(loader)
        mean_ADE = total_ADE / total_samples
        mean_FDE = total_FDE / total_samples
        mean_MR = miss_count / total_samples
        return {
            'mse_loss': mean_mse_loss,
            'ADE': mean_ADE,
            'FDE': mean_FDE,
            'MR': mean_MR
        }

    @staticmethod
    def to(md,device):

        if md == None:
            return md

        if type(device) == str:
            device = torch.device(device)
        elif issubclass(type(device),torch.nn.Module):
            for pm in device.parameters():
                device = pm.device
                break
        if hasattr(md,'device'):
            md_device = md.device
        else:
            for pm in md.parameters():
                md_device = pm.device
                break

        if md_device != device:
            if md_device == torch.device('cpu'):
                md.to(device)
            else:
                md.cpu().to(device)

        return md

    def log(self, text):
        path = os.path.join('./trained_models',self.timestamp)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join('./trained_models',self.timestamp,'train.log'), 'a', encoding='utf-8') as f:
            f.write(text + '\n')

    def fit(self):
        net, device = self.net, self.device
        net = net.to(self.device)
        if self.loader_sampler is not None:
            loader = DataLoader(self.train_dataset, batch_sampler=self.loader_sampler)
        else:
            loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, pin_memory=True)
        opt = self.optc(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        MSELoss = torch.nn.MSELoss()
        CrsLoss = torch.nn.CrossEntropyLoss()
        losses = []
        count = 0
        self.log(f"开始训练，总 Epoch: {self.epoch}")
        for epoch in range(1, self.epoch + 1):
            start_time = time.time()
            net.train()
            mean_loss = 0
            for batch in loader:
                opt.zero_grad()
                for key in batch:
                    batch[key] = batch[key].to(device)
                output,lon,lat = net(batch)
                
                loss = 0
                if count %10 == 0:
                    nll_loss = NLL(output,batch['Y'])
                else:
                    loss = MSELoss(output[:,:,:2],batch['Y'])
                    nll_loss = 0

                if lon != None:
                    loss = loss + CrsLoss(lon,batch['LC'])
                
                if lat != None:
                    loss = loss + CrsLoss(lat,batch['LI'])
            
                if hasattr(net,'loss'):
                    loss = loss + net.loss

                (loss + nll_loss).backward()
                count += 1
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.clip_grad)
                opt.step()
                mean_loss += loss.item() if type(loss) != int else loss
            mean_loss /= len(loader)
            losses.append(mean_loss)

            # 验证集评估
            val_info = ""
            if self.val_dataset is not None:
                val_metrics = self._test_model_(net, self.val_dataset, batch_size=self.batch_size, device=device)
                val_loss = val_metrics['mse_loss']
                val_info = f" | Val Loss: {val_loss:.6f}, ADE: {val_metrics['ADE']:.6f}, FDE: {val_metrics['FDE']:.6f}, MR: {val_metrics['MR']:.6f}"
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if not os.path.exists(os.path.join('./trained_models',self.timestamp)) or not hasattr(self,'best_model_path'):
                        path = os.path.join('./trained_models/',self.timestamp)
                        os.makedirs(path, exist_ok=True)
                        self.best_model_path = os.path.join(path,sanitize_class_name(net.__class__.__name__) + '.pth')
                    torch.save(net.state_dict(), self.best_model_path)
                    self.log(f"[Epoch {epoch}] 新最佳验证集Loss: {val_loss:.6f}，模型已保存。")
                net = net.to(device)

            elapsed = time.time() - start_time
            log_line = f"[Epoch {epoch}] Train Loss: {mean_loss:.6f}{val_info} | Time: {elapsed:.2f}s"
            print(log_line)
            self.log(log_line)

        # 加载最佳模型并测试
        if self.test_dataset is not None and os.path.exists(self.best_model_path):
            net.load_state_dict(torch.load(self.best_model_path))
            test_metrics = self._test_model_(net, self.test_dataset, batch_size=self.batch_size, device=device)
            test_log = f"测试集结果: Loss: {test_metrics['mse_loss']:.6f}, ADE: {test_metrics['ADE']:.6f}, FDE: {test_metrics['FDE']:.6f}, MR: {test_metrics['MR']:.6f}"
            print(test_log)
            self.log(test_log)

        net.cpu()
        return net, losses, self.test_record
